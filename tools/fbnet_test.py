from __future__ import division
import _init_paths
import argparse

import torch
import mmcv
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel

from mmdet import datasets
from mmdet.core import results2json, coco_eval
from mmdet.datasets import build_dataloader
from mmdet.models import build_detector, detectors

from model.two_stage import TwoStageDetector
from backbone import FBNet_sample
from mmcv import Config as mmcv_config
from torch import nn
import time
from register import DETECTORS 
from model.single_stage import SingleStageDetector

class detection(nn.Module):
    def __init__(self, cfg, train_cfg, test_cfg, search_cfg, theta_txt):
        super(detection, self).__init__()
        self.fbnet = FBNet_sample(search_cfg, theta_txt)
        self.detect = DETECTORS[cfg['type']](cfg, train_cfg, test_cfg)
        # self.init_weights()
    def forward(self, **input):
        input["img"] = self.fbnet(input.pop('img')[0])
        # input['img'] = [input['img']]
        print(type(input['img']))
        loss = self.detect(**input)
        #loss, self.loss_vars = parse_losses(loss)
        return loss
    def init_weights(self):
        return self.fbnet.init_weights()

def single_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    torch.cuda.synchronize()
    start = time.time()
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)
        if show:
            model.module.detect.show_result(data, result, dataset.img_norm_cfg,
                                     dataset=dataset.CLASSES, score_thr=0.6)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    torch.cuda.synchronize()
    end = time.time()
    print("avg forward time:",(end-start)/i)
    return results


def _data_func(data, device_id):
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(return_loss=False, rescale=True, **data)


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    # parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--gpus', default=1, type=int, help='GPU number used for testing')
    parser.add_argument('--fb_cfg', help='fbnet backbone search space')
    parser.add_argument('--model_cfg', help='detector arch')
    parser.add_argument('--theta_txt', help='theta_txt build fbnet backbone arch')
    parser.add_argument(
        '--proc_per_gpu',
        default=1,
        type=int,
        help='Number of processes per GPU')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    search_cfg = mmcv_config.fromfile(args.fb_cfg)
    _space = search_cfg.search_space
    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    # cfg = mmcv.Config.fromfile(args.config)
    cfg = mmcv.Config.fromfile(args.model_cfg)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))
    if args.gpus == 1:
        model = detection(
            mmcv_config(cfg['model_cfg']), 
            mmcv_config(cfg['train_cfg']), 
            mmcv_config(cfg['test_cfg']),
            _sapce,
            args.theta_txt)
        # model = build_detector(
        #     cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        load_checkpoint(model, args.checkpoint)
        model = MMDataParallel(model, device_ids=[0])

        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            num_gpus=1,
            dist=False,
            shuffle=False)
        outputs = single_test(model, data_loader, args.show)
    else:
        model_args = cfg.model.copy()
        model_args.update(train_cfg=None, test_cfg=cfg.test_cfg)
        model_type = getattr(detectors, model_args.pop('type'))
        outputs = parallel_test(
            model_type,
            model_args,
            args.checkpoint,
            dataset,
            _data_func,
            range(args.gpus),
            workers_per_gpu=args.proc_per_gpu)

    if args.out:
        print('writing results to {}'.format(args.out))
        mmcv.dump(outputs, args.out)
        eval_types = args.eval
        if eval_types:
            print('Starting evaluate {}'.format(' and '.join(eval_types)))
            if eval_types == ['proposal_fast']:
                result_file = args.out
                coco_eval(result_file, eval_types, dataset.coco)
            else:
                if not isinstance(outputs[0], dict):
                    result_file = args.out + '.json'
                    results2json(dataset, outputs, result_file)
                    coco_eval(result_file, eval_types, dataset.coco)
                else:
                    for name in outputs[0]:
                        print('\nEvaluating {}'.format(name))
                        outputs_ = [out[name] for out in outputs]
                        result_file = args.out + '.{}.json'.format(name)
                        results2json(dataset, outputs_, result_file)
                        coco_eval(result_file, eval_types, dataset.coco)


if __name__ == '__main__':
    main()
