from __future__ import division
import _init_paths
import argparse
import torch 
from torch import nn
from torch.nn import DataParallel

from backbone.fbnet import FBNet
from backbone import FBNet_sample
from utils import _logger

# parser = argparse.ArgumentParser(description="Train a model with data parallel for base net \
#                                 and model parallel for classify net.")
# parser.add_argument('--batch-size', type=int, default=256,
#                     help='training batch size of all devices.')
# parser.add_argument('--epochs', type=int, default=1000,
#                     help='number of training epochs.')
# parser.add_argument('--log-frequence', type=int, default=10,
#                     help='log frequence, default is 400')
# parser.add_argument('--gpus', type=str, default='0',
#                     help='gpus, default is 0')
# parser.add_argument('--fb-cfg', type=str, help='fbnet_buildconfig')
# parser.add_argument('--det-cfg', type=str, help='fbnet_buildconfig')
# args = parser.parse_args()



# det_cfg = Config.fromfile(args.det_cfg)

from model.classifer import Class
from model.two_stage import TwoStageDetector


from mmcv import Config as mmcv_config
from mmdet.apis.train import parse_losses
from mmdet.models.backbones import ResNet
import logging
import time
from search.search import fbnet_search
from mmdet.datasets import CocoDataset, build_dataloader, CustomDataset
from mmcv.parallel import MMDataParallel

class detection(nn.Module):
    def __init__(self, cfg, train_cfg, test_cfg, base, depth, space, theta_txt):
        super(detection, self).__init__()
        #self.resnet50 = ResNet(50, num_stages=3, out_indices=[2], strides=(1,2,2), dilations=(1,1,1))
        #self.fbnet = FBNet_sample(base, depth, space, theta_txt="theta/epoch_49_end_arch_params.txt")
        self.fbnet = FBNet_sample(base, depth, space, theta_txt)#,weight_opt_dict, theta_opt_dict,w_cfg=w_sche_cfg)
        self.detect = TwoStageDetector(cfg, train_cfg, test_cfg)
        #self.init_weights = self.fbnet.init_weights()
        self.init_weights()
    def forward(self, **input):
        #input["x"] = self.resnet50(input.pop('img'))
        #input["x"] = self.fbnet(input.pop('img'),input.pop('temp'))
        input["img"] = self.fbnet(input.pop('img'))
        #print(input)
        loss = self.detect(**input)
        #loss, self.loss_vars = parse_losses(loss)
        return loss
    def init_weights(self):
        return self.fbnet.init_weights()

import argparse

from mmdet import __version__
from mmdet.datasets import get_dataset
from mmdet.apis import (train_detector, init_dist, get_root_logger,
                        set_random_seed)
from mmdet.models import build_detector

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    #parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument('--fb_cfg', help='fbnet search unit')
    parser.add_argument('--model_cfg', help='model arch')
    parser.add_argument('--theta_txt', help='choose block')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    #search_cfg = Config.fromfile("search_config/config.py")
    fb_cfg = mmcv_config.fromfile(args.fb_cfg)
    _space = fb_cfg.search_space
    base = _space['base']
    depth = _space['depth']
    space = _space['space']

    model_cfg = mmcv_config.fromfile(args.model_cfg)
    #cfg = Config.fromfile("cascade_mask_rcnn_r50_fpn_1x.py")
    # set cudnn_benchmark
    if model_cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        model_cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        model_cfg.resume_from = args.resume_from
    model_cfg.gpus = args.gpus
    if model_cfg.checkpoint_config is not None:
        # save mmdet version in checkpoints as meta data
        model_cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__, config=model_cfg.text)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **model_cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(model_cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)
    model = detection(mmcv_config(model_cfg['model_cfg']), 
                        mmcv_config(model_cfg['train_cfg']), 
                        mmcv_config(model_cfg['test_cfg']), 
                        base, depth, space,
                        args.theta_txt)
    print(model)
    # model = build_detector(
    #     cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    train_dataset = get_dataset(model_cfg.data.train)
    # img_norm_cfg = dict(
    #     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    # w_data_cfg = dict(train=dict(
    #     #ann_file=data_root + 'annotations/instances_train2017.json',
    #     ann_file='/home1/zhaoyu/dataset/car_w.pkl',
    #     #img_prefix="" ,#data_root + 'val2017/',
    #     img_prefix="",
    #     #img_prefix=data_root + 'train2017/',
    #     img_scale=(1000, 320),
    #     img_norm_cfg=img_norm_cfg,
    #     size_divisor=32,
    #     flip_ratio=0.5,
    #     with_mask=False,
    #     with_crowd=True,
    #     with_label=True))    
    # w_dataset = CustomDataset(**w_data_cfg['train'])
    # w_dataset = build_dataloader(w_dataset, imgs_per_gpu=16,
    #                workers_per_gpu=2,#config.imgs_per_gpu,
    #                dist=False,
    #                num_gpus=len(gpus))
    train_detector(
        model,
        train_dataset,
        model_cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger)


if __name__ == '__main__':
    main()
