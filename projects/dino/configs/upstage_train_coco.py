from detrex.config import get_config
from .models.dino_vitdet import model
from fvcore.common.param_scheduler import MultiStepParamScheduler
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler

# get default config
dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
# lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_upstage
train = get_config("common/train.py").train


# modify training config
# train.init_checkpoint = "../checkpoints/dino_vitdet_large_4scale_50ep.pth"
# train.init_checkpoint = "../checkpoints/model_final.pth"
train.init_checkpoint = "./output/coco_large_ema/model_0001999.pth"
# train.init_checkpoint2 = "./output/coco_large/model_0018999.pth"
# train.init_checkpoint3 = "./output/coco2/model_0011999.pth"
train.output_dir = "./output/coco_large_ema"

# max training iterations
train.max_iter = 2000

# run evaluation every 5000 iters
train.eval_period = 21000

# log training infomation every 20 iters
train.log_period = 100

# save checkpoint every 5000 iters
train.checkpointer.period = 2000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device

# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 4

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 8

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir

# convert vitdet-base to vitdet-large
model.backbone.net.embed_dim = 1024
model.backbone.net.depth = 24
model.backbone.net.num_heads = 16
model.backbone.net.drop_path_rate = 0.4
# 5, 11, 17, 23 for global attention
model.backbone.net.window_block_indexes = (
    list(range(0, 5)) + list(range(6, 11)) + list(range(12, 17)) + list(range(18, 23))
)

# use warmup lr scheduler
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[0.1],
        milestones=[2000],
    ),
    # warmup_length=250 / train.max_iter,
    warmup_length=0,
    warmup_factor=0.001,
)
