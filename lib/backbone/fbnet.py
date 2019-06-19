import torch 
from torch import nn
import torch.optim as optim
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from .Unit import BASICUNIT
from .operator import MixedOp
import sys
sys.path.append('../../')
from utils import CosineDecayLR, AvgrageMeter
from itertools import accumulate
import time
import logging

class FBNet(nn.Module):
    def __init__(self, search_cfg,
                weight_opt_dict=None,
                theta_opt_dict=None,
                skip=True,
                speed_txt = './speed_cpu.txt', 
                init_temperature=5.0,
                temperature_decay=0.965,
                w_lr=CosineDecayLR,
                w_cfg={'T_max':400},
                t_lr=None,
                t_cfg=None
                ):
        """
        weight_opt_dict:dict
            weights param optim setting
        theta_opt_dict: dict
            theta param optim setting
        """
        super(FBNet, self).__init__()
        self.skip = skip 
        self.space = search_cfg['space']
        self.depth = search_cfg['depth']
        self.depth_num = sum(self.depth)
        self.num = len(self.space)
        self.theta = self.get_theta()
        self.temp = init_temperature
        self.temp_decay = temperature_decay
        self.weight_opt = weight_opt_dict
        self.theta_opt = theta_opt_dict
        self.w_lr = w_lr
        self.w_cfg = w_cfg
        self.t_lr = t_lr
        self.t_cfg = t_cfg
        self.out = [list(accumulate(self.depth))[z]-1 for z in search_cfg['out']]
        self.logger = logging
        self.base = nn.Sequential(*search_cfg['base'])
        self._ops = self.build()
        with open(speed_txt, "r") as f:
            self.speed = f.readlines()

    def get_theta(self):
        theta = []
        for i in range(len(self.depth)):
            theta.append(nn.Parameter(torch.ones((1, self.num)), 
                                                    requires_grad=True))
            for j in range(self.depth[i]-1):
                if self.skip:
                    theta.append(nn.Parameter(torch.ones((1, self.num+1)), 
                                                    requires_grad=True))
                else:
                    theta.append(nn.Parameter(torch.ones((1, self.num)), 
                                                    requires_grad=True))
        return theta 

    def build(self):
        _ops = nn.ModuleList()
        for i in range(len(self.depth)):
            for j in range(self.depth[i]):       
                blocks = nn.ModuleList()
                if j!=0 and self.skip:
                    blocks.append(BASICUNIT['Identity']())    
                for unit in self.space:
                    if j == 0:
                        unit['param']['stride'] = 2
                        unit['param']['_in'] = unit['param']['_in']
                        unit['param']['_out'] = unit['param']['_out']*2
                    else:
                        unit['param']['stride'] = 1
                        unit['param']['_in'] = unit['param']['_out']
                    blocks.append(BASICUNIT[unit['type']](**unit['param']))        
                _ops.append(MixedOp(blocks))
        assert len(_ops) == self.depth_num
        return _ops

    def forward(self, x, temperature):
        x = self.base(x)
        late_loss = 0
        outs = []
        for i in range(len(self._ops)):
            weight = nn.functional.gumbel_softmax(self.theta[i], temperature).cuda()
            late_loss += self.lat_loss(self.speed[i], weight)
            x = self._ops[i](x, weight)
            if i in self.out:
                outs.append(x)
        return outs , late_loss

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

            # if self.dcn is not None:
            #     for m in self.modules():
            #         if isinstance(m, Bottleneck) and hasattr(
            #                 m, 'conv2_offset'):
            #             constant_init(m.conv2_offset, 0)

            # if self.zero_init_residual:
            #     for m in self.modules():
            #         if isinstance(m, Bottleneck):
            #             constant_init(m.norm3, 0)
            #         elif isinstance(m, BasicBlock):
            #             constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None') 
    def lat_loss(self, speed, weight):
        speed = speed.strip().split()
        speed = [float(tmp) for tmp in speed]
        assert len(speed) == weight.size()[1]
        lat = weight*torch.Tensor(speed).cuda()
        return lat.sum()
