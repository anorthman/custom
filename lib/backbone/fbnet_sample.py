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
import copy
class FBNet_sample(nn.Module):
    def __init__(self, search_cfg, theta_txt,
                skip=True,
                ):
        super(FBNet_sample, self).__init__()

        self.skip = skip 
        self.space = search_cfg['space']
        self.depth = search_cfg['depth']
        self.depth_num = sum(self.depth)
        self.num = len(self.space)
        self.theta_txt = theta_txt
        self.out = [list(accumulate(self.depth))[z]-1 for z in search_cfg['out']]
        self.logger = logging
        self.base = nn.Sequential(*search_cfg['base'])
        self._ops = self.build()

    def build(self):
        res = []
        with open(self.theta_txt,"r") as f:
            thetas = f.readlines()
        for i in range(len(thetas)):
            _theta = torch.Tensor([float(x) for x in thetas[i].split(" ")])
            weight = nn.functional.softmax(_theta)
            # print(weight)
            _max = torch.argmax(weight)
            res.append(_max)
        # print("-------------------")
        # print(res)
        blocks = nn.ModuleList()
        k = 0
        for i in range(len(self.depth)):
            for j in range(self.depth[i]):       
                if j ==0 :
                    unit = copy.deepcopy(self.space[res[k]])
                elif j!=0 and self.skip and res[k] == 0:
                    blocks.append(BASICUNIT['Identity']())
                    k += 1
                    continue
                else :
                    unit = copy.deepcopy(self.space[res[k]-1])
                if j == 0:
                    unit['param']['stride'] = 2
                    unit['param']['_in'] = unit['param']['_in']*(2**i)
                    unit['param']['_out'] = unit['param']['_out']*(2**(i+1))
                else:
                    unit['param']['stride'] = 1
                    unit['param']['_in'] = unit['param']['_in']*(2**(i+1))
                    unit['param']['_out'] = unit['param']['_out']*(2**(i+1))
                blocks.append(BASICUNIT[unit['type']](**unit['param']))
                k += 1        
        return blocks

    def forward(self, x):
        x = self.base(x)
        outs = []
        for i in range(len(self._ops)):
            x = self._ops[i](x)
            if i in self.out:
                outs.append(x)
        return outs 

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



