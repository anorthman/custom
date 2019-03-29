import os
import logging
import time
from utils import CosineDecayLR
from mmcv.parallel import MMDataParallel
from mmdet.apis.train import parse_losses
from torch import nn

class fbnet_search(object):
    def __init__(self, model, gpus, imgs_per_gpu, w_cfg=None, mmcv_parallel=True, save_result_path="./theta/"):
        assert isinstance(model, nn.Module), "model must be a module" 
        self.model = model.cuda().train()
        if mmcv_parallel:
            self.model = MMDataParallel(self.model, gpus) 
        self.weights = self.model.module.parameters()
        self.theta = self.model.module.theta
        self.w_optim = self.model.module.get_wopt(self.weights)
        self.t_optim = self.model.module.get_topt(self.theta)
        self.logger = logging
        self.w_lr = CosineDecayLR 
        self.w_lr_scheduler = self.w_lr(self.w_optim, **w_cfg)
        self.temp = self.model.module.temp
        self.temp_decay = self.model.module.temp_decay
        self.decay_temperature_step = 100
        self._batch_end_cb_func = []
        self._batch_end_cb_func.append(lambda x, y: self._update_temp(x, y))
        self._batch_end_cb_func.append(lambda x, y: self._test_log(x, y))
        self._epoch_end_cb_func = []
        self._epoch_end_cb_func.append(lambda x: self.save_theta(x))
        self.batch_size = imgs_per_gpu * len(gpus)
        self.save_result_path = save_result_path
        if os.path.exists(self.save_result_path):
            os.mkdirs(self.save_result_path) 

    def _test_log(self, epoch, batch):
        if (batch > 0) and (batch % self.log_frequence == 0):
            self.toc = time.time()
            speed = 1.0 * (self.batch_size * self.log_frequence) / (self.toc - self.tic)
            self.log_info(epoch, batch, speed=speed)
            self.tic = time.time()

    def log_info(self, epoch, batch, speed=None):
        msg = "Epoch[%d] Batch[%d]" % (epoch, batch)
        for key in self.loss_vars.keys():
            msg += " %s:%.3f"%(key, self.loss_vars[key])
        if speed is not None:
            msg += ' Speed: %.6f samples/sec' % speed
        self.logger.info(msg)
        return msg
    def save_theta(epoch):
        save_path = "%s/epoch_%d_end_arch_params.txt" % \
                        (self.save_result_path, epoch)
        res = []
        with open(save_path, 'w') as f:
          for t in self.arch_params:
            t_list = list(t.detach().cpu().numpy())
            res.append(t_list)
            s = ' '.join([str(tmp) for tmp in t_list])
            f.write(s + '\n')
        self.logger.info("Save architecture paramterse to %s" % save_path)

    def _update_temp(self, epoch, batch=None):
        if (batch is None) or ((batch > 0) and (batch % self.decay_temperature_step == 0)):
            self.temp *= self.temp_decay
            msg = 'Epoch[%d] ' % epoch
            if not batch is None:
                msg += 'Batch[%d] ' % batch
            msg += "Decay temperature to %.6f" % self.temp
            self.logger.info(msg)

    def step_w(self, _optim, **input):
        _optim.zero_grad()
        loss = self.model(**input)
        loss, self.loss_vars = parse_losses(loss)
        loss.backward()
        _optim.step()
        self.w_lr_scheduler.step()

    def step_t(self, _optim, **input):
        _optim.zero_grad()
        loss = self.model(**input)
        loss, self.loss_vars = parse_losses(loss)
        loss.backward()
        _optim.step()
        loss.backward()
        _optim.step()

    def batch_end_callback(self, epoch, batch):
        for func in self._batch_end_cb_func:
            func(epoch, batch)

    def search(self, train_w_ds, train_t_ds, **kwargs):
        num_epoch = kwargs.get('epoch', 100)
        start_w_epoch = kwargs.get('start_w_epoch', 5)
        self.log_frequence = kwargs.get('log_frequence', 50)

        assert start_w_epoch >= 1, "Start to train w first"

        for epoch in range(start_w_epoch):
            self.tic = time.time()
            self.logger.info("Start to train w for epoch %d" % epoch)
            for step, inputs in enumerate(train_w_ds):
                inputs['temperature'] = self.temp 
                self.step_w(self.w_optim,**inputs)
                self.batch_end_callback(epoch, step)
            #self.epoch_end_callback(epoch)
        for epoch in range(num_epoch):
            self.tic = time.time()
            self.logger.info("Start to train arch for epoch %d" % (epoch+start_w_epoch))
            for step, inputs in enumerate(train_t_ds):
                inputs['temperature'] = self.temp 
                self.step_t(self.t_optim, **inputs)
                self.batch_end_callback(epoch+start_w_epoch, step)
            self.tic = time.time()
            self.logger.info("Start to train w for epoch %d" % (epoch+start_w_epoch))
            for step, inputs in enumerate(train_w_ds):
                self.step_w(self.w_optim, **inputs)
                self.batch_end_callback(epoch+start_w_epoch, step)
            self.epoch_end_callback(epoch)