from omegaconf import OmegaConf
import copy
import torch
import array

from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detrex.data import DetrDatasetMapper

# Register Dataset
try:
    register_coco_instances('coco_trash_train', {}, '/data/home/user/Data/upstage/dataset/train.json', '/data/home/user/Data/upstage/dataset/')
except AssertionError:
    pass

try:
    register_coco_instances('coco_trash_test', {}, '/data/home/user/Data/upstage/dataset/test.json', '/data/home/user/Data/upstage/dataset/')
except AssertionError:
    pass

MetadataCatalog.get('coco_trash_train').thing_classes = ["General trash", "Paper", "Paper pack", "Metal",
                                                         "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]


def MyMapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')

    transform_list = [
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3)
    ]

    image, transforms = T.apply_transform_gens(transform_list, image)

    dataset_dict['image'] = torch.as_tensor(image.transpose(2,0,1).astype('float32'))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop('annotations')
        if obj.get('iscrowd', 0) == 0
    ]

    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict['instances'] = utils.filter_empty_instances(instances)

    return dataset_dict


def TestMapper(dataset_dict):

    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')

    dataset_dict['image'] = torch.tensor(deepcopy.copy(image))

    return dataset_dict


dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="coco_trash_train"),
    # mapper=L(DetrDatasetMapper)(
    #     augmentation=[
    #         L(T.RandomFlip)(),
    #         L(T.ResizeShortestEdge)(
    #             short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
    #             max_size=1333,
    #             sample_style="choice",
    #         ),
    #     ],
    #     augmentation_with_crop=[
    #         L(T.RandomFlip)(),
    #         L(T.ResizeShortestEdge)(
    #             short_edge_length=(400, 500, 600),
    #             sample_style="choice",
    #         ),
    #         L(T.RandomCrop)(
    #             crop_type="absolute_range",
    #             crop_size=(384, 600),
    #         ),
    #         L(T.ResizeShortestEdge)(
    #             short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
    #             max_size=1333,
    #             sample_style="choice",
    #         ),
    #     ],
    #     is_train=True,
    #     mask_on=False,
    #     img_format="RGB",
    # ),
    mapper=MyMapper,
    total_batch_size=4,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="coco_trash_test", filter_empty=False),
    # mapper=L(DetrDatasetMapper)(
    #     augmentation=[
    #         L(T.ResizeShortestEdge)(
    #             short_edge_length=800,
    #             max_size=1333,
    #         ),
    #     ],
    #     augmentation_with_crop=None,
    #     is_train=False,
    #     mask_on=False,
    #     img_format="RGB",
    # ),
    mapper=TestMapper,
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)
