import torch.nn as nn

from .base_det import BaseDetector
from mmdet.models import builder
from register import DETECTORS
from mmdet.core import bbox2result


@DETECTORS.register_module
class SingleStageDetector(BaseDetector):

    def __init__(self,
                 # neck=None,
                 # bbox_head=None,
                 cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetector, self).__init__()
        # self.backbone = builder.build_backbone(backbone)
        if 'neck' in cfg:
            self.neck = builder.build_neck(cfg['neck'])
        self.bbox_head = builder.build_head(cfg['bbox_head'])
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SingleStageDetector, self).init_weights(pretrained)
        # self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        if self.with_neck:
            x = self.neck(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        if self.with_neck:
            x = self.neck(img)
        outs = self.bbox_head(x)
        #for out in outs:
        #    for t in out:
        #        print(t.size())
        #        print(t)
        #print("over")
        #exit()
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
