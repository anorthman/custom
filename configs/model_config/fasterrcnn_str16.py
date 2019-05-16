
model_cfg = dict(
    type='FasterRCNN',
    neck=dict(
        type='FPN',
        in_channels=[128],
        #in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=1,
        end_level=1),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[2,4,8,12],
        anchor_ratios=[0.5, 1.0],
        #anchor_strides=[4, 8, 16, 32, 64],
        anchor_strides=[16],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        use_sigmoid_cls=True),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        #featmap_strides=[4, 8, 16, 32],
        featmap_strides=[16]
        ),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=2,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        smoothl1_beta=1 / 9.0,
        debug=False),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100)
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)
# dataset settings
dataset_type = 'CustomDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=48,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        # ann_file="/home1/zhaoyu/dataset/car.pkl",
        #ann_file="/home1/zhaoyu/dataset/car_w.pkl",#'/home/zhaoyu/data/mmdet_data/testcar.pkl', #data_root + 'annotations/instances_train2017.json',
        ann_file="/home1/zhaoyu/dataset/traincar.pkl",
        #ann_file="/home1/zhaoyu/dataset/car_1img.pkl", #'/home/zhaoyu/data/mmdet_data/testcar.pkl', #data_root + 'annotations/instances_train2017.json',
        img_prefix='',
        #img_prefix=data_root + 'train2017/',
        img_scale=(1333, 320),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=False,
        with_label=True),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=True),
    test=dict(
        type=dataset_type,
        # ann_file="/home1/zhaoyu/dataset/car.pkl",
        ann_file="/home1/zhaoyu/dataset/testcar.pkl",
        # ann_file="/home1/zhaoyu/dataset/car_1img.pkl", #'/home/zhaoyu/data/mmdet_data/testcar.pkl', #data_root + 'annotations/instances_train2017.json',
        # ann_file='/home/zhaoyu/data/mmdet_data/testcar.pkl', #data_root + 'annotations/instances_val2017.json',
        #ann_file=data_root + 'annotations/instances_val2017.json',
        #img_prefix=data_root + 'val2017/',
        img_prefix='',
        img_scale=(1333, 320),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=False,
        test_mode=True))
# optimizer

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.00000)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[20, 28])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 30
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/car/search_fbnet2'
load_from = None
resume_from = None
workflow = [('train', 1)]
