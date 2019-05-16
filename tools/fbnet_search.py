import _init_paths
import argparse
import torch 
from torch import nn
from torch.nn import DataParallel

from backbone.fbnet import FBNet
from utils import _logger





# class fbnet_cls(FBNet):
#     def __init__(self, base, depth, space, weight_opt_dict, theta_opt_dict, w_sche_cfg):
#         super(fbnet_cls, self).__init__(base, depth, space,weight_opt_dict, theta_opt_dict,w_cfg=w_sche_cfg)
#         self._class = Class(classes=2)
#         #self.theta = self.fbnet.theta
#         self._criterion = nn.CrossEntropyLoss().cuda()
#         # print(self.step_w)

#     def forward(self, x, y):
#         x = self.forward_train(5.0, x)
#         x = self._class(x)
#         loss = self._criterion(x,y).mean()
#         print(loss)
#         return loss

# net = fbnet_cls(base, depth, space,
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
# img = torch.ones((4,3,24,24)).cuda()
# label = torch.cuda.LongTensor([1,1,1,1])
# input = {"x":img,"y":label}
# # for n, p in enumerate(net.parameters()):
# #     print(p.size())
# w_opt = model.module.get_wopt(net.parameters())
# t_opt = model.module.get_topt(model.module.theta)
# # for i in range(20):
# #     #model.module.step_t(t_opt, input=img, label=label)
# #     model.module.step_t(t_opt, **input)
# for i in range(10):
#     model.module.step_w(w_opt, **input)


# dataset_type = 'CocoDataset'
from model.two_stage import TwoStageDetector
from mmcv import Config as mmcv_config
from mmdet.apis.train import parse_losses
from mmdet.models.backbones import ResNet
import logging
import time
import os
from search.search import fbnet_search
from mmdet.datasets import CocoDataset, build_dataloader, CustomDataset
from mmcv.parallel import MMDataParallel
import sys

parser = argparse.ArgumentParser(description="Train a model with data parallel for base net \
                                and model parallel for classify net.")
parser.add_argument('--batch-size', type=int, default=256,
                    help='training batch size of all devices.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of training epochs.')
parser.add_argument('--log-frequence', type=int, default=10,
                    help='log frequence, default is 400')
parser.add_argument('--gpus', type=str, default='0',
                    help='gpus, default is 0')
parser.add_argument('--fb_cfg', type=str, help='fbnet_buildconfig')
parser.add_argument('--model_cfg', type=str, help='fbnet_buildconfig')
parser.add_argument('--speed_txt',type=str,help='block_time')

args = parser.parse_args()

search_cfg = mmcv_config.fromfile(args.fb_cfg)
_space = search_cfg.search_space
base = _space['base']
depth = _space['depth']
space = _space['space']

model_cfg = mmcv_config.fromfile(args.model_cfg)
# # dataset settings
data_root = '/home/zhaoyu/workspace/pytorch1.0/mmdetection_/data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
w_data_cfg = dict(train=dict(
        ann_file='/home1/zhaoyu/dataset/car_w.pkl',
        img_prefix="",
        img_scale=(1000, 320),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=True,
        with_label=True))

t_data_cfg = dict(train=dict(
        ann_file='/home1/zhaoyu/dataset/car_t.pkl',
        img_prefix="",
        img_scale=(1000, 320),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=True,
        with_label=True))
gpus = [0,1]
w_dataset = CustomDataset(**w_data_cfg['train'])
w_dataset = build_dataloader(w_dataset, imgs_per_gpu=16,
                   workers_per_gpu=2,
                   dist=False,
                   num_gpus=len(gpus))

t_dataset = CustomDataset(**t_data_cfg['train'])
t_dataset = build_dataloader(t_dataset, imgs_per_gpu=16,
                   workers_per_gpu=2,
                   dist=False,
                   num_gpus=len(gpus))

class detection(nn.Module):
    def __init__(self, cfg, train_cfg, test_cfg, speed_txt):
        super(detection, self).__init__()
        #self.resnet50 = ResNet(50, num_stages=3, out_indices=[2], strides=(1,2,2), dilations=(1,1,1))
        print(speed_txt)
        self.fbnet = FBNet(base, depth, space, speed_txt=speed_txt)#,weight_opt_dict, theta_opt_dict,w_cfg=w_sche_cfg)
        self.detect = TwoStageDetector(cfg, train_cfg, test_cfg)
        self.theta = self.fbnet.theta
        self.temp = self.fbnet.temp
        self.temp_decay = self.fbnet.temp_decay
        #self.init_weights = self.fbnet.init_weights()
        self.init_weights()
    def forward(self, **input):
        #input["x"] = self.resnet50(input.pop('img'))
        #input["x"] = self.fbnet(input.pop('img'),input.pop('temp'))
        input["img"], latloss = self.fbnet(input.pop('img'),input.pop('temp'))
        #print(input)
        loss = self.detect(**input)
        #loss, self.loss_vars = parse_losses(loss)
        return loss, latloss
    def init_weights(self):
        return self.fbnet.init_weights()


det = detection(mmcv_config(model_cfg['model_cfg']), 
                mmcv_config(model_cfg['train_cfg']), 
                mmcv_config(model_cfg['test_cfg']), 
                speed_txt=args.speed_txt)
print(det)
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
                                          'warmup_step' : 100,
                                          't_mul' : 1.5,
                                          'lr_mul' : 0.95,
                                        },alpha=0.1)
save_result_path = "./theta/"+args.fb_cfg.split('/')[-1][:-3]+'_'+args.model_cfg.split('/')[-1][:-3]
if not os.path.exists(save_result_path):
    os.makedirs(save_result_path)
print(save_result_path)
sys.exit(0)
searcher.search(
            train_w_ds = w_dataset,
            train_t_ds = t_dataset,
            epoch=50,
            start_w_epoch=20,
            log_frequence=10,
            save_result_path=save_result_path)


