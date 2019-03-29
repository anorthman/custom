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
    def __init__(self, depth, space, 
                weight_opt_dict=None,
                theta_opt_dict=None,
                skip=True, 
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
        self.space = space
        self.depth = depth
        self.depth_num = sum(depth)
        self.num = len(self.space)
        self._ops = self.build()
        self.theta = self.get_theta()
        self.temp = init_temperature
        self.temp_decay = temperature_decay
        self.weight_opt = weight_opt_dict
        self.theta_opt = theta_opt_dict
        self.w_lr = w_lr
        self.w_cfg = w_cfg
        self.t_lr = t_lr
        self.t_cfg = t_cfg
        self.ft_map = list(accumulate(self.depth))
        self.ft = [1,1,1,1]
        self.logger = logging
        #self.baseconv = nn.Conv2d()

        # self.weight = None

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
                if j!=0:
                    blocks.append(BASICUNIT['Identity']())    
                for unit in self.space:
                    if j == 0:
                        unit['param']['stride'] = 2
                        unit['param']['_in'] = unit['param']['_in']
                        unit['param']['_out'] = unit['param']['_out']*2
                    else:
                        unit['param']['stride'] = 1
                        unit['param']['_in'] = unit['param']['_out']
                    #     unit['param']['_in'] = unit['param']['_in']*(2**(i+1))
                    #     unit['param']['_out'] = unit['param']['_in']*(2**(i+1))
                    blocks.append(BASICUNIT[unit['type']](**unit['param']))        
                _ops.append(MixedOp(blocks))
        assert len(_ops) == self.depth_num
        return _ops

    # def get_weight(self, theta):
    #     weight = nn.functional.gumbel_softmax(theta, self.temp)
    #     return weight.cuda()

    def forward_train(self, temperature ,x):  
        for i in range(len(self._ops)):
            weight = nn.functional.gumbel_softmax(self.theta[i], temperature).cuda()
            x = self._ops[i](x, weight)
        return x

    def forward_traindet(self, x, temperature):
        outs = []
        j = 0
        for i in range(len(self._ops)):
            weight = nn.functional.gumbel_softmax(self.theta[i], temperature).cuda()
            x = self._ops[i](x, weight)
            if i == self.ft_map[j]-1:
                if self.ft[j] == 1:
                    outs.append(x)
                j += 1
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

    def lat_loss(self, speed):
        lat = []
        for i in range(len(sel._ops)):
            lat.append(weight[i]*speed[i].cuda())
        return lat.sum()

    def get_wopt(self, weights):
        self.weight_opt['params'] = weights
        opt_type = self.weight_opt.pop('type')
        #print(opt_type)
        w_opt = getattr(optim, opt_type)(**self.weight_opt)
        self.w_lr_scheduler = self.w_lr(w_opt, **self.w_cfg)
        return w_opt

    def step_w(self, _optim, **input):
        _optim.zero_grad()
        loss = self.forward(**input)
        loss.backward()
        _optim.step()
        self.w_lr_scheduler.step()

    def get_topt(self, theta):
        self.theta_opt['params'] = theta
        opt_type = self.theta_opt.pop('type')
        t_opt = getattr(optim, opt_type)(**self.theta_opt)
        self.t_lr_scheduler = None if self.t_lr is None  \
                            else self.t_lr(t_opt, **self.t_cfg)
        return t_opt

    def step_t(self, _optim, **input):
        _optim.zero_grad()
        loss = self.forward(**input)
        loss.backward()
        _optim.step()
        #self.t_lr_scheduler.step()       

    def search(self, theta, weights, train_w_ds, train_t_ds, **kwargs):
        num_epoch = kwargs.get('epoch', 100)
        start_w_epoch = kwargs.get('start_w_epoch', 5)
        self.log_frequence = kwargs.get('log_frequence', 50)
        w_optim = self.get_wopt(weights)
        t_optim = self.get_topt(theta) 

        assert start_w_epoch >= 1, "Start to train w first"

        for epoch in range(start_w_epoch):
            self.tic = time.time()
            self.logger.info("Start to train w for epoch %d" % epoch)
            for step, inputs in enumerate(train_w_ds):
                print("input", inputs)
                self.step_w(w_optim,**inputs)
                self.batch_end_callback(epoch, step)

        for epoch in range(num_epoch):
            self.tic = time.time()
            self.logger.info("Start to train arch for epoch %d" % (epoch+start_w_epoch))
            for step, inputs in enumerate(train_t_ds):
                self.step_t(t_optim, **inputs)
                self.batch_end_callback(epoch+start_w_epoch, step)
            
                self.tic = time.time()
                self.logger.info("Start to train w for epoch %d" % (epoch+start_w_epoch))
            for step, inputs in enumerate(train_w_ds):
                self.step_w(w_optim, **inputs)
                self.batch_end_callback(epoch+start_w_epoch, step)  



