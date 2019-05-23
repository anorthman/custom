import os
import logging
import time
from utils import CosineDecayLR
from mmcv.parallel import MMDataParallel
from mmdet.apis.train import parse_losses
from torch import nn
import torch.optim as optim
class fbnet_search(object):
    def __init__(self, model, gpus, imgs_per_gpu, 
                weight_opt_dict=None, 
                theta_opt_dict=None,
                weight_lr_sche=None,
                weight_lr_type=CosineDecayLR,
                mmcv_parallel=True, 
                save_result_path="./theta/",
                decay_temp_epoch=2,
                alpha=0.2, beta=0.6):
        assert isinstance(model, nn.Module), "model must be a module" 
        self.model = model.cuda().train()
        if mmcv_parallel:
            self.model = MMDataParallel(self.model, gpus) 
        self.weights = self.model.module.parameters()
        self.theta = self.model.module.theta
        self.w_optim = getattr(optim, weight_opt_dict.pop("type"))(self.weights,**weight_opt_dict)
        self.t_optim = getattr(optim, theta_opt_dict.pop("type"))(self.theta,**theta_opt_dict)
        self.logger = logging
        self.w_lr_scheduler = weight_lr_type(self.w_optim, **weight_lr_sche)
        self.temp = self.model.module.temp
        self.temp_decay = self.model.module.temp_decay
        self.decay_temp_epoch = decay_temp_epoch
        self._batch_end_cb_func = []
        self._batch_end_cb_func.append(lambda x, y: self._test_log(x, y))
        self._epoch_end_cb_func = []
        self._epoch_end_cb_func.append(lambda x: self.save_theta(x))
        self._epoch_end_cb_func.append(lambda x: self._update_temp(x))
        self.batch_size = imgs_per_gpu * len(gpus)
        self.save_result_path = save_result_path
        self.alpha = alpha
        self.beta = beta
        if not os.path.exists(self.save_result_path):
            os.mkdir(self.save_result_path) 

    def _test_log(self, epoch, batch):
        if (batch > 0) and (batch % self.log_frequence == 0):
            self.toc = time.time()
            speed = 1.0 * (self.batch_size * self.log_frequence) / (self.toc - self.tic)
            self.log_info(epoch, batch, speed=speed)
            self.tic = time.time()

    def log_info(self, epoch, batch, speed=None):
        msg = "Epoch[%d] Batch[%d]" % (epoch, batch)
        msg += " %s:%.3f"%("lat_loss", self.lateloss)
        for key in self.loss_vars.keys():
            msg += " %s:%.3f"%(key, self.loss_vars[key])
        if speed is not None:
            msg += ' Speed: %.6f samples/sec' % speed
        self.logger.info(msg)
        return msg
    def save_theta(self, epoch):
        save_path = "%s/epoch_%d_end_arch_params.txt" % \
                        (self.save_result_path, epoch)
        with open(save_path, 'w') as f:
          for t in self.theta:
            t_list = list(t.detach().cpu().numpy())
            s = ' '.join(["%.5f"%tmp for tmp in t_list[0]])
            f.write(s + '\n')
        self.logger.info("Save architecture paramterse to %s" % save_path)

    def _update_temp(self, epoch):
        if epoch % self.decay_temp_epoch == 0:
            self.temp *= self.temp_decay
            msg = 'Epoch[%d] ' % epoch
            msg += "Decay temperature to %.6f" % self.temp
            self.logger.info(msg)

    def step_w(self, _optim, **input):
        _optim.zero_grad()
        loss, lateloss = self.model(**input)
        loss, self.loss_vars = parse_losses(loss)
        self.lateloss = self.alpha*(lateloss.mean().log().pow(self.beta))
        #loss = loss + self.lateloss
        loss.backward()
        _optim.step()
        self.w_lr_scheduler.step()

    def step_t(self, _optim, **input):
        _optim.zero_grad()
        loss, lateloss = self.model(**input)
        loss, self.loss_vars = parse_losses(loss)
        self.lateloss = self.alpha*(lateloss.mean().log().pow(self.beta))
        #loss = loss + self.lateloss
        loss.backward()
        _optim.step()
        
    def batch_end_callback(self, epoch, batch):
        for func in self._batch_end_cb_func:
            func(epoch, batch)

    def epoch_end_callback(self, epoch):
        for func in self._epoch_end_cb_func:
            func(epoch)

    def search(self, train_w_ds, train_t_ds, **kwargs):
        num_epoch = kwargs.get('epoch', 100)
        start_w_epoch = kwargs.get('start_w_epoch', 5)
        self.log_frequence = kwargs.get('log_frequence', 50)

        assert start_w_epoch >= 1, "Start to train w first"

        for epoch in range(start_w_epoch):
            self.tic = time.time()
            self.logger.info("Start to train w for epoch %d" % epoch)
            for step, inputs in enumerate(train_w_ds):
                inputs['temp'] = self.temp 
                self.step_w(self.w_optim, **inputs)
                self.batch_end_callback(epoch, step)
            #self.epoch_end_callback(epoch)
        for epoch in range(num_epoch):
            self.tic = time.time()
            self.logger.info("Start to train arch for epoch %d" % (epoch+start_w_epoch))
            for step, inputs in enumerate(train_t_ds):
                inputs['temp'] = self.temp 
                self.step_t(self.t_optim, **inputs)
                self.batch_end_callback(epoch+start_w_epoch, step)
            self.tic = time.time()
            self.logger.info("Start to train w for epoch %d" % (epoch+start_w_epoch))
            for step, inputs in enumerate(train_w_ds):
                inputs['temp'] = self.temp 
                self.step_w(self.w_optim, **inputs)
                self.batch_end_callback(epoch+start_w_epoch, step)
            self.epoch_end_callback(epoch)
