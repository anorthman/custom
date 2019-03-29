import _init_paths
import argparse

from mmcv import Config
import torch 
from torch import nn
from torch.nn import DataParallel

from backbone.fbnet import FBNet
from utils import _logger

parser = argparse.ArgumentParser(description="Train a model with data parallel for base net \
                                and model parallel for classify net.")
parser.add_argument('--batch-size', type=int, default=256,
                    help='training batch size of all devices.')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of training epochs.')
parser.add_argument('--log-frequence', type=int, default=4,
                    help='log frequence, default is 400')
parser.add_argument('--gpus', type=str, default='0',
                    help='gpus, default is 0')
parser.add_argument('--fb_cfg', type=str, help='fbnet_buildconfig')
args = parser.parse_args()

search_cfg = Config.fromfile(args.fb_cfg)

_space = search_cfg.search_space
depth = _space['depth']
space = _space['space']

class Config(object):
  alpha = 0.2
  beta = 0.6
  speed_f = './speed_cpu.txt'
  w_lr = 0.001
  w_mom = 0.9
  w_wd = 1e-4
  t_lr = 0.01
  t_wd = 5e-4
  t_beta = (0.9, 0.999)
  model_save_path = '/home1/nas/fbnet-pytorch/'
  start_w_epoch = 2
  train_portion = 0.8
  imgs_per_gpu = 2

lr_scheduler_params = {
  'logger' : _logger,
  'T_max' : 400,
  'alpha' : 1e-4,
  'warmup_step' : 100,
  't_mul' : 1.5,
  'lr_mul' : 0.95,
}

config = Config()

from model.classifer import Class
from model.two_stage import TwoStageDetector
#class fbnet_cls(nn.Module):
# class fbnet_cls(FBNet):
#     def __init__(self, depth, space, weight_opt_dict, theta_opt_dict, w_sche_cfg):
#         super(fbnet_cls, self).__init__(depth, space,weight_opt_dict, theta_opt_dict,w_cfg=w_sche_cfg)
#         self.module = nn.ModuleList()
#         #self.fbnet = FBNet(depth,space)
#         self._class = Class(classes=2)
#         #self.theta = self.fbnet.theta
#         self._criterion = nn.CrossEntropyLoss().cuda()
#         # print(self.step_w)

#     def forward(self, x, y):
#         x = self.forward_train(x)
#         x = self._class(x)
#         loss = self._criterion(x,y).mean()
#         print(loss)
#         return loss

# net = fbnet_cls(depth, space,
#               weight_opt_dict={'type':'SGD',
#                             'lr':config.w_lr,
#                             'momentum':config.w_mom,
#                             'weight_decay':config.w_wd},
#               theta_opt_dict={'type':'Adam',
#                              'lr':config.t_lr,
#                              'betas':config.t_beta,
#                              'weight_decay':config.t_wd},
#               w_sche_cfg=lr_scheduler_params
#               )


# model = DataParallel(net).cuda()
# print(net)
# img = torch.ones((4,8,24,24)).cuda()
# label = torch.cuda.LongTensor([1,1,1,1])
# input = [img,label]
# # for n, p in enumerate(net.parameters()):
# #     print(p.size())
# w_opt = model.module.get_wopt(net.parameters())
# t_opt = model.module.get_topt(model.module.theta)
# for i in range(20):
#     #model.module.step_t(t_opt, input=img, label=label)
#     model.module.step_t(t_opt, *input)
# for i in range(10):
#     model.module.step_w(w_opt, *input)

model_cfg = dict(
    type='FasterRCNN',
    pretrained='modelzoo://resnet50',
    neck=dict(
        type='FPN',
        # in_channels=[122, 128, 256, 256],
        in_channels=[16, 32, 64, 128], 
        out_channels=256,
        num_outs=5),    
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[4],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        use_sigmoid_cls=True),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=2, #81,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        smoothl1_beta=1 / 9.0,
        debug=False),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=True))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100)
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)

dataset_type = 'CocoDataset'
data_root = '/home/zhaoyu/workspace/pytorch1.0/mmdetection/data/coco/'
#data_root = '/home/zhouchangqing/git/mmdetection/data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data_cfg = dict(train=dict(
        # type=dataset_type,
        # ann_file=data_root + 'annotations/instances_train2017.json',
        ann_file='/home1/zhaoyu/dataset/testcar.pkl',
        img_prefix="" ,#data_root + 'val2017/',
        #img_prefix=data_root + 'train2017/',
        img_scale=(1333, 320),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=True,
        with_label=True))

from mmcv import Config as mmcv_config
from mmdet.apis.train import parse_losses

class fbnet_det(FBNet):
    def __init__(self, depth, space, weight_opt_dict, theta_opt_dict, w_sche_cfg, cfg, train_cfg, test_cfg):
        super(fbnet_det, self).__init__(depth, space,weight_opt_dict, theta_opt_dict,w_cfg=w_sche_cfg)

        self._detect = TwoStageDetector(cfg, train_cfg, test_cfg)
        self.init_weights()

    def forward(self, **input):
        input["x"] = self.forward_traindet(input.pop("img"), input.pop('temperature'))
        loss = self._detect(**input)
        # loss, log_vars = parse_losses(loss)
        # print(log_vars)
        return loss

from mmdet.datasets import CocoDataset, build_dataloader, CustomDataset
coco_dataset = CustomDataset(**data_cfg['train'])
coco_dataset = build_dataloader(coco_dataset, imgs_per_gpu=config.imgs_per_gpu,
                   workers_per_gpu=config.imgs_per_gpu,
                   dist=False,
                   num_gpus=len(args.gpus.split(',')))

net = fbnet_det(depth, space,
              weight_opt_dict={'type':'SGD',
                            'lr':config.w_lr,
                            'momentum':config.w_mom,
                            'weight_decay':config.w_wd},
              theta_opt_dict={'type':'Adam',
                             'lr':config.t_lr,
                             'betas':config.t_beta,
                             'weight_decay':config.t_wd},
              w_sche_cfg=lr_scheduler_params,
              cfg = mmcv_config(model_cfg),
              train_cfg = mmcv_config(train_cfg),
              test_cfg = mmcv_config(test_cfg)
              )
from mmcv.parallel import MMDataParallel
gpus = [int(x) for x in args.gpus.split(",")]

import logging
import time
from search.search import fbnet_search

searcher = fbnet_search(net, gpus, config.imgs_per_gpu, lr_scheduler_params)
searcher.search(
            train_w_ds = coco_dataset,
            train_t_ds = coco_dataset,
            epoch=args.epochs,
            start_w_epoch=config.start_w_epoch,
            log_frequence=args.log_frequence)

