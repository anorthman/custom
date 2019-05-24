from __future__ import division
import _init_paths
import argparse
import torch 
from torch import nn
from torch.nn import DataParallel

from backbone.fbnet import FBNet
from backbone import FBNet_sample
from utils import _logger

from model.two_stage import TwoStageDetector

from mmcv import Config as mmcv_config
from mmdet.apis.train import parse_losses
from mmdet.models.backbones import ResNet
import logging
import time
from search.search import fbnet_search
from mmdet.datasets import CocoDataset, build_dataloader, CustomDataset
from mmcv.parallel import MMDataParallel
from register import DETECTORS 
from model.single_stage import SingleStageDetector
class detection(nn.Module):
    def __init__(self, cfg, train_cfg, test_cfg, search_cfg, theta_txt):
        super(detection, self).__init__()
        self.fbnet = FBNet_sample(search_cfg, theta_txt)
        self.detect = DETECTORS[cfg['type']](cfg, train_cfg, test_cfg)
        self.init_weights()
    def forward(self, **input):
        input["img"] = self.fbnet(input.pop('img'))
        loss = self.detect(**input)
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
    fb_cfg = mmcv_config.fromfile(args.fb_cfg)
    _space = fb_cfg.search_space
    # base = _space['base']
    # depth = _space['depth']
    # space = _space['space']

    model_cfg = mmcv_config.fromfile(args.model_cfg)
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
                        _space,
                        args.theta_txt)
    print(model)
    train_dataset = get_dataset(model_cfg.data.train)
    train_detector(
        model,
        train_dataset,
        model_cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger)


if __name__ == '__main__':
    main()
