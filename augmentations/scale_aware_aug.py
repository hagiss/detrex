import copy
import torch
import torchvision
from augmentations.image_level_augs.img_level_augs import Img_augs
from augmentations.box_level_augs.box_level_augs import Box_augs
from augmentations.box_level_augs.color_augs import color_aug_func
from augmentations.box_level_augs.geometric_augs import geometric_aug_func
from detectron2.utils.comm import get_world_size


class SA_Aug(object):
    def __init__(self):
        autoaug_list = (6, 9, 5, 3, 3, 4, 2, 4, 4, 4, 5, 2, 4, 1, 4, 2, 6, 4, 2, 2, 2, 6, 2, 2, 2, 0, 5, 1, 3, 0, 8, 5, 2, 8, 7, 5, 1, 3, 3, 3)
        num_policies = 5
        max_iters = 20000
        scale_splits = [2048, 10240, 51200]
        box_prob = 0.3

        img_aug_list = autoaug_list[:4]
        img_augs_dict = {'zoom_out':{'prob':img_aug_list[0]*0.05, 'level':img_aug_list[1]},
                         'zoom_in':{'prob':img_aug_list[2]*0.05, 'level':img_aug_list[3]}}
        self.img_augs = Img_augs(img_augs_dict=img_augs_dict)

        box_aug_list = autoaug_list[4:]
        color_aug_types = list(color_aug_func.keys())
        geometric_aug_types = list(geometric_aug_func.keys())
        policies = []
        for i in range(num_policies):
            _start_pos = i * 6
            sub_policy = [(color_aug_types[box_aug_list[_start_pos+0]%len(color_aug_types)], box_aug_list[_start_pos+1]* 0.1, box_aug_list[_start_pos+2], ), # box_color policy
                          (geometric_aug_types[box_aug_list[_start_pos+3]%len(geometric_aug_types)], box_aug_list[_start_pos+4]* 0.1, box_aug_list[_start_pos+5])] # box_geometric policy
            policies.append(sub_policy)

        _start_pos = num_policies * 6
        scale_ratios = {'area': [box_aug_list[_start_pos+0], box_aug_list[_start_pos+1], box_aug_list[_start_pos+2]],
                        'prob': [box_aug_list[_start_pos+3], box_aug_list[_start_pos+4], box_aug_list[_start_pos+5]]}

        box_augs_dict = {'policies': policies, 'scale_ratios': scale_ratios}

        self.box_augs = Box_augs(box_augs_dict=box_augs_dict, max_iters=max_iters, scale_splits=scale_splits, box_prob=box_prob)
        self.max_iters = max_iters

        self.count = 0
        self.start_iter = 0
        # num_gpus = get_world_size()
        self.batch_size = 1
        self.num_workers = 4
        if self.num_workers==0:
            self.num_workers += 1

    def __call__(self, dataset_dict):
        tensor_out = dataset_dict['image']
        target_out = dataset_dict['instances']._fields
        iteration = self.count // self.batch_size * self.num_workers
        # tensor_out, target_out = self.img_augs(tensor, target)
        tensor_out, target_out = self.box_augs(tensor_out, target_out, iteration=self.start_iter + iteration)
        self.count += 1

        dataset_dict['instances']._image_size = tensor_out.shape[1:]
        dataset_dict['instances']._fields = target_out
        # print(tensor_out.shape)
        dataset_dict['image'] = tensor_out #.transpose(2, 0, 1)

        return dataset_dict
