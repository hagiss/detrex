#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging
import os
import sys
import time
import torch
import copy
from tqdm import tqdm
import pandas as pd
from torch.nn.parallel import DataParallel, DistributedDataParallel

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from detectron2.utils.events import (
    CommonMetricPrinter, 
    JSONWriter, 
    TensorboardXWriter
)
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from fvcore.common.config import CfgNode as _CfgNode
import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data import build_detection_test_loader, build_detection_train_loader

from detrex.utils import WandbWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

logger = logging.getLogger("detrex")


class Trainer(SimpleTrainer):
    """
    We've combine Simple and AMP Trainer together.
    """

    def __init__(
        self,
        model,
        ema_model,
        dataloader,
        optimizer,
        amp=False,
        clip_grad_params=None,
        grad_scaler=None,
    ):
        super().__init__(model=model, data_loader=dataloader, optimizer=optimizer)

        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        if amp:
            if grad_scaler is None:
                from torch.cuda.amp import GradScaler

                grad_scaler = GradScaler()
            self.grad_scaler = grad_scaler

        # set True to use amp training
        self.amp = amp

        # gradient clip hyper-params
        self.clip_grad_params = clip_grad_params

        self.ema_model = ema_model
        for p in self.ema_model.parameters():
            p.requires_grad = False

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[Trainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        with autocast(enabled=self.amp):
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()

        if self.amp:
            self.grad_scaler.scale(losses).backward()
            if self.clip_grad_params is not None:
                self.grad_scaler.unscale_(self.optimizer)
                self.clip_grads(self.model.parameters())
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            losses.backward()
            if self.clip_grad_params is not None:
                self.clip_grads(self.model.parameters())
            self.optimizer.step()

        # update ema
        m = 0.996
        for current_params, ma_params in zip(self.model.parameters(), self.ema_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = old_weight * m + (1 - m) * up_weight

        self._write_metrics(loss_dict, data_time)

    def clip_grads(self, params):
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return torch.nn.utils.clip_grad_norm_(
                parameters=params,
                **self.clip_grad_params,
            )


def do_test(cfg, model):
    # if "evaluator" in cfg.dataloader:
    #     ret = inference_on_dataset(
    #         model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
    #     )
    #     print_csv_format(ret)
    #     return ret
    model.eval()
    prediction_strings = []
    file_names = []

    test_loader = instantiate(cfg.dataloader.test)

    for data in tqdm(test_loader):
        # data = data[0]
        prediction_string = ''
        # print("data", data)
        with torch.no_grad():
            outputs = model(data)[0]['instances']
        # print("outputs", outputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        targets = outputs.pred_classes.cpu().tolist()
        boxes = [i.cpu().detach().numpy() for i in outputs.pred_boxes]
        scores = outputs.scores.cpu().tolist()

        for target, box, score in zip(targets, boxes, scores):
            prediction_string += (str(target) + ' ' + str(score) + ' ' + str(box[0]) + ' '
                                  + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' ')
        prediction_strings.append(prediction_string)
        # print(data[0]['file_name'].replace('/data/home/user/Data/upstage/dataset/', ''))
        file_names.append(data[0]['file_name'].replace('/data/home/user/Data/upstage/dataset/', ''))

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(os.path.join(cfg.train.output_dir, f'submission_det2.csv'), index=None)
    submission.head()


def do_test_ensemble(cfg, model1, model2):
    # if "evaluator" in cfg.dataloader:
    #     ret = inference_on_dataset(
    #         model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
    #     )
    #     print_csv_format(ret)
    #     return ret
    model1.eval()
    model2.eval()
    # model3.eval()
    prediction_strings = []
    file_names = []

    test_loader = instantiate(cfg.dataloader.test)

    for data in tqdm(test_loader):
        # data = data[0]
        prediction_string = ''
        # print("data", data)
        with torch.no_grad():
            multi_level_feats1, multi_level_masks, multi_level_position_embeddings1, query_embeds, attn_mask, dn_meta, images = model1.get_backbone_feature(data)
            multi_level_feats2, _, multi_level_position_embeddings2, _, _, _, _ = model2.get_backbone_feature(data)

            # print("feat", multi_level_feats1)
            # print("pos", multi_level_position_embeddings1)
            # print("query", query_embeds1)

            multi_level_feats = [(i + j) / 2 for i, j in zip(multi_level_feats1, multi_level_feats2)]
            multi_level_position_embeddings = [(i + j) / 2 for i, j in zip(multi_level_position_embeddings1, multi_level_position_embeddings2)]
            # query_embeds = [(i+j)/2 for i, j in zip(query_embeds1, query_embeds2)]
            outputs = model2.forward_with_feature(data, multi_level_feats, multi_level_masks, multi_level_position_embeddings, query_embeds, attn_mask, dn_meta, images)[0]['instances']
            # outputs3 = model3(data)[0]['instances']
        # print("outputs", outputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        targets = outputs.pred_classes.cpu().tolist()
        boxes = [i.cpu().detach().numpy() for i in outputs.pred_boxes]
        scores = outputs.scores.cpu().tolist()

        for target, box, score in zip(targets, boxes, scores):
            prediction_string += (str(target) + ' ' + str(score) + ' ' + str(box[0]) + ' '
                                  + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' ')
        prediction_strings.append(prediction_string)
        # print(data[0]['file_name'].replace('/data/home/user/Data/upstage/dataset/', ''))
        file_names.append(data[0]['file_name'].replace('/data/home/user/Data/upstage/dataset/', ''))

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(os.path.join(cfg.train.output_dir, f'submission_det2.csv'), index=None)
    submission.head()


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    ema_model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)
    ema_model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)
    # train_loader = build_detection_train_loader(
    #     cfg, mapper=MyMapper, sampler=None
    # )

    model = create_ddp_model(model, **cfg.train.ddp)
    ema_model = create_ddp_model(ema_model, **cfg.train.ddp)

    trainer = Trainer(
        model=model,
        ema_model=ema_model,
        dataloader=train_loader,
        optimizer=optim,
        amp=cfg.train.amp.enabled,
        clip_grad_params=cfg.train.clip_grad.params if cfg.train.clip_grad.enabled else None,
    )

    DetectionCheckpointer(model).resume_or_load(cfg.train.init_checkpoint, resume=False)
    checkpointer = DetectionCheckpointer(
        ema_model,
        cfg.train.output_dir,
        trainer=trainer,
    )

    if comm.is_main_process():
        # writers = default_writers(cfg.train.output_dir, cfg.train.max_iter)
        output_dir = cfg.train.output_dir
        PathManager.mkdirs(output_dir)
        writers = [
            CommonMetricPrinter(cfg.train.max_iter),
            JSONWriter(os.path.join(output_dir, "metrics.json")),
            TensorboardXWriter(output_dir),
        ]
        if cfg.train.wandb.enabled:
            PathManager.mkdirs(cfg.train.wandb.params.dir)
            writers.append(WandbWriter(cfg))

    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                writers,
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    cfg.model.num_classes = 10

    if args.eval_only:
        # model = instantiate(cfg.model)
        # model.to(cfg.train.device)
        # model = create_ddp_model(model)
        # DetectionCheckpointer(model).resume_or_load(cfg.train.init_checkpoint, resume=False)
        # do_test(cfg, model)
        ###########################################################
        model1 = instantiate(cfg.model)
        model1.to(cfg.train.device)
        model1 = create_ddp_model(model1)
        DetectionCheckpointer(model1).resume_or_load(cfg.train.init_checkpoint1, resume=False)

        model2 = instantiate(cfg.model)
        model2.to(cfg.train.device)
        model2 = create_ddp_model(model2)
        DetectionCheckpointer(model2).resume_or_load(cfg.train.init_checkpoint2, resume=False)
        #
        # model3 = instantiate(cfg.model)
        # model3.to(cfg.train.device)
        # model3 = create_ddp_model(model3)
        # DetectionCheckpointer(model3).resume_or_load(cfg.train.init_checkpoint3, resume=False)
        #
        do_test_ensemble(cfg, model1, model2)
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
