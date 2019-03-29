import numpy as np

import h5py as h5
import os.path as osp
import pickle as pkl
import copy

def main():
    _roidb = []
    cache_path = _cache_path
    if osp.exists(cache_path):
        print ("cache_path exists ~ ")
    else:
        print ("begin set roidb")
        cnt = 0
        with open(_img_label_size, 'r') as f:
            try:
                while True:
                    img_path, label_path, height, width = next(f).strip().split()
                    h = int(height)
                    w = int(width)
                    bbs, labels, bbs_ignore, labels_ignore = _load_gt(label_path)
                    roidb = {
                        'filename': img_path,
                        'width': w,
                        'height': h,
                        'ann': {
                            'bboxes': bbs.astype(np.float32),
                            'labels': labels.astype(np.int64),
                            'bboxes_ignore': bbs_ignore.astype(np.float32),
                            'labels_ignore': labels_ignore.astype(np.int64)
                        }
                    }
                    if is_valid(roidb):
                        _roidb.append(roidb)
                        cnt += 1
                    if cnt % 100 == 0:
                        print ("processed {}".format(cnt))
            except StopIteration:
                pass
        num_images = cnt
        print ("set roidb done!")
        with open(cache_path, 'wb', pkl.HIGHEST_PROTOCOL) as cache_f:
            pkl.dump(_roidb, cache_f)

def is_valid(roidb):
    flag = True
    if roidb["ann"]["bboxes"].size == 0:
        flag = False
    return flag

def _load_gt(label_path):

    bbs_all = np.zeros((0, 4), dtype=np.float32)
    gt_classes = np.zeros((0), dtype=np.float32)
    ignore_bbs_all = np.zeros((0, 4), dtype=np.float32)
    ignore_gt_classes = np.zeros((0), dtype=np.float32)
    label_path = label_path.strip()
    if label_path.endswith(".h5"):
        with h5.File(label_path, 'r') as f:
            for cls in f.keys():
                if cls in _classes:                     
                    bbs = f[cls][...]
                    if bbs.size == 0:
                        continue
                    bbs[:, 2] = bbs[:, 0] + bbs[:, 2] - 1
                    bbs[:, 3] = bbs[:, 1] + bbs[:, 3] - 1
                    bbs_all = np.vstack((bbs_all, bbs))
                    gt_classes = np.hstack(
                        (gt_classes, np.ones(len(bbs)) * _class_to_ind[cls]))
                elif cls in ignore_clsses:
                    bbs = f[cls][...]
                    if bbs.size == 0:
                        continue
                    bbs[:, 2] = bbs[:, 0] + bbs[:, 2] - 1
                    bbs[:, 3] = bbs[:, 1] + bbs[:, 3] - 1
                    ignore_bbs_all = np.vstack((ignore_bbs_all, bbs))
                    ignore_gt_classes = np.hstack(
                        (ignore_gt_classes, np.ones(len(bbs)) * int(-1)))
                else:
                    continue
    else:
        raise TypeError('label must be h5file, but got others')                      

    assert len(bbs_all) == len(gt_classes), "bbs, gt_classes len differ!"
    assert len(ignore_bbs_all) == len(ignore_gt_classes),  "bbs, gt_classes len differ!"

    return bbs_all, gt_classes, ignore_bbs_all, ignore_gt_classes

if __name__ == '__main__':
    ignore_clsses = []
    _classes = ["background","person_upper"]
    num_classes = 2
    assert len(_classes) == int(num_classes)
    #_class_to_ind = dict(zip(_classes, [0,1,1]))
    _class_to_ind = dict(zip(_classes, range(num_classes)))
    _img_label_size = "/home/zhaoyu/workspace/pytorch1.0/mmdetection/tools/convert_datasets/img_label_size"
    _cache_path = "/home1/zhaoyu/dataset/wide.pkl"
    main()