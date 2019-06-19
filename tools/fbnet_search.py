import _init_paths
import argparse
import torch 
from torch import nn
import logging
import time
import os

from mmdet.datasets import CocoDataset, build_dataloader, CustomDataset
from mmcv.parallel import MMDataParallel
from mmcv import Config as mmcv_config
from mmdet.apis.train import parse_losses

from backbone.fbnet import FBNet
from utils import _logger
from model.two_stage import TwoStageDetector
from model.single_stage import SingleStageDetector
from search.search import fbnet_search
from tools.convert_custom import split_data
from register import DETECTORS 
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class detection(nn.Module):
    def __init__(self, cfg, train_cfg, test_cfg, search_cfg, speed_txt):
        super(detection, self).__init__()
        print(speed_txt)
        self.fbnet = FBNet(search_cfg, speed_txt=speed_txt)
        self.detect = DETECTORS[cfg['type']](cfg, train_cfg, test_cfg)
        self.theta = self.fbnet.theta
        self.temp = self.fbnet.temp
        self.temp_decay = self.fbnet.temp_decay
        self.init_weights()

    def forward(self, **input):
        input["img"], latloss = self.fbnet(input.pop('img'),input.pop('temp'))
        loss = self.detect(**input)
        return loss, latloss

    def init_weights(self):
        return self.fbnet.init_weights()

def main():    
    parser = argparse.ArgumentParser(description="Train a model with data parallel for base net \
                                    and model parallel for classify net.")
    #parser.add_argument('--batch-size', type=int, default=256,
    #                    help='training batch size of all devices.')
    #parser.add_argument('--epochs', type=int, default=1000,
    #                    help='number of training epochs.')
    #parser.add_argument('--log-frequence', type=int, default=10,
    #                    help='log frequence, default is 400')
    parser.add_argument('--gpus', type=str, default='0',
                        help='gpus, default is 0')
    parser.add_argument('--fb_cfg', type=str, help='fbnet_buildconfig')
    parser.add_argument('--model_cfg', type=str, help='fbnet_buildconfig')
    parser.add_argument('--speed_txt',type=str,help='block_time')

    args = parser.parse_args()

    search_cfg = mmcv_config.fromfile(args.fb_cfg)
    _space = search_cfg.search_space

    model_cfg = mmcv_config.fromfile(args.model_cfg)
    # # dataset settings
    classes = ['background', 'face']
    min_scale = 0
    w_data, t_data=split_data('./data/newlibraf_info/train_imglist', 
                    './newlibraf_info/newlibraf_face', classes, min_scale)
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    w_data_cfg = dict(train=dict(
            # ann_file='/home1/zhaoyu/dataset/car_w.pkl',
            ann_file=w_data,
            img_prefix="./data/",
            img_scale=(1000, 216),
            img_norm_cfg=img_norm_cfg,
            size_divisor=32,
            flip_ratio=0.5,
            with_mask=False,
            with_crowd=True,
            with_label=True))

    t_data_cfg = dict(train=dict(
            # ann_file='/home1/zhaoyu/dataset/car_t.pkl',
            ann_file=t_data,
            img_prefix="./data/",
            img_scale=(1000, 216),
            img_norm_cfg=img_norm_cfg,
            size_divisor=32,
            flip_ratio=0.5,
            with_mask=False,
            with_crowd=True,
            with_label=True))
    gpus = [int(x) for x in args.gpus.split(",")]
    w_dataset = CustomDataset(**w_data_cfg['train'])
    w_dataset = build_dataloader(w_dataset, imgs_per_gpu=16,
                       workers_per_gpu=4,
                       dist=False,
                       num_gpus=len(gpus))

    t_dataset = CustomDataset(**t_data_cfg['train'])
    t_dataset = build_dataloader(t_dataset, imgs_per_gpu=16,
                       workers_per_gpu=2,
                       dist=False,
                       num_gpus=len(gpus))

    det = detection(mmcv_config(model_cfg['model_cfg']), 
                    mmcv_config(model_cfg['train_cfg']), 
                    mmcv_config(model_cfg['test_cfg']),
                    _space,
                    speed_txt=args.speed_txt)
    print(det)
    save_result_path = "./theta/"+args.fb_cfg.split('/')[-1][:-3]+'_'+args.model_cfg.split('/')[-1][:-3]
    if not os.path.exists(save_result_path):
        os.makedirs(save_result_path)
    searcher = fbnet_search(det, gpus, imgs_per_gpu=8,              
                            weight_opt_dict={'type':'SGD',
                                            'lr':0.01,
                                            'momentum':0.9,
                                            'weight_decay':0.001},
                              theta_opt_dict={'type':'Adam',
                                             'lr':0.01,
                                             'betas':(0.9,0.99),
                                             'weight_decay':5e-4}, 
                              weight_lr_sche = { 
                                              'logger' : _logger,
                                              'T_max' : 400,
                                              'alpha' : 1e-4,
                                              'warmup_step' : 1000,
                                              't_mul' : 1.5,
                                              'lr_mul' : 0.95,
                                            },alpha=0.1,
                            save_result_path=save_result_path)
    searcher.search(
                train_w_ds = w_dataset,
                train_t_ds = t_dataset,
                epoch=100,
                start_w_epoch=5,
                log_frequence=10)
if __name__ == "__main__":
    main()
