import numpy as np
import h5py as h5
import os.path as osp
import pickle as pkl
import copy
import xmltodict

def split_data(imglist, data_name, classes, min_scale, data_root='./data/'):
    with open(imglist, "r") as f:
        lines = f.readlines()
    length = len(lines)
    cnt = 0
    # train w dataset
    cache_w = data_root+data_name+"_w"+".pkl"
    if osp.exists(cache_w):
        print('cache_w exist')
    else:
        _roidb = []
        for i in range(0, int(length*0.8)):
            img_path, label_path, height, width = lines[i].strip().split()
            h = int(height)
            w = int(width)
            label_path = data_root+label_path
            bbs, labels, bbs_ignore, labels_ignore = _load_gt(label_path, h, w, min_scale, classes)
            if bbs.size == 0:
                continue
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
        with open(cache_w, 'wb', pkl.HIGHEST_PROTOCOL) as cache_f:
            pkl.dump(_roidb, cache_f)
        print ("set w roidb done!","img_nums:",cnt)
    ## train theta dataset
    cache_t = data_root+data_name+"_t"+".pkl"
    if osp.exists(cache_t):
        print('cache_t exist')
    else:
        _roidb = [] 
        cnt = 0
        for i in range(int(length*0.8), length):
            img_path, label_path, height, width = lines[i].strip().split()
            h = int(height)
            w = int(width)
            label_path = data_root+label_path
            bbs, labels, bbs_ignore, labels_ignore = _load_gt(label_path, h, w, min_scale, classes)
            if bbs.size == 0:
                continue
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
        cache_t = data_root+data_name+"_t"+".pkl"
        with open(cache_t, 'wb', pkl.HIGHEST_PROTOCOL) as cache_f:
            pkl.dump(_roidb, cache_f)
        print ("set t roidb done! img nums:", cnt)
    return cache_w, cache_t

def is_valid(roidb):
    flag = True
    if roidb["ann"]["bboxes"].size == 0:
        flag = False
    return flag

def _load_gt(label_path, height, width, min_scale, classes, ignore_clsses=None, h=1080, w=1920):

    bbs_all = np.zeros((0, 4), dtype=np.float32)
    gt_classes = np.zeros((0), dtype=np.float32)
    ignore_bbs_all = np.zeros((0, 4), dtype=np.float32)
    ignore_gt_classes = np.zeros((0), dtype=np.float32)
    label_path = label_path.strip()
    area = height * width
    _class_to_ind = dict(zip(classes, range(len(classes))))
    if label_path.endswith(".h5"):
        with h5.File(label_path, 'r') as f:
            for cls in f.keys():
                if cls in classes:                     
                    bbs = f[cls][...]
                    if bbs.size == 0:
                        continue
                    _bbs = []
                    _bbs_min = []
                    for i in range(bbs.shape[0]):
                        w = float(bbs[i,2])
                        h = float(bbs[i,3])
                        bbs[i, 2] = bbs[i, 0] + w - 1 
                        bbs[i, 3] = bbs[i, 1] + h - 1 
                        if w *h < height*width*min_scale/1080/1920:
                            _bbs_min.append(bbs[i])
                        else: 
                            _bbs.append(bbs[i])
                    if len(_bbs_min) != 0:
                        ignore_bbs_all = np.vstack((ignore_bbs_all, np.array(_bbs_min)))
                        ignore_gt_classes = np.hstack(
                            (ignore_gt_classes, np.ones(len(_bbs_min)) * int(-1)))
                    if len(_bbs) != 0:
                        bbs_all = np.vstack((bbs_all, np.array(_bbs)))
                        gt_classes = np.hstack(
                            (gt_classes, np.ones(len(_bbs)) * _class_to_ind[cls]))
                elif ignore_clsses != None:
                    if cls in ignore_clsses:
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
    elif label_path.endswith(".xml"):
        with open(label_path, 'r') as f:
            print(label_path)
            d = xmltodict.parse(f.read())
            anno = d['annotation']
            objs = anno['object']
            m = {}
            if not isinstance(objs, list):
                objs = [objs]
            for obj in objs:
                label = obj['name']
                box = obj['bndbox']
                x1 = box['xmin']
                y1 = box['ymin']
                x2 = box['xmax']
                y2 = box['ymax']
                bb = [x1, y1, x2, y2]
                bb = [int(x) for x in bb]
                if bb[0] >= bb[2] or bb[1] >= bb[3]:
                    print(bb)
                    continue
                if label in m.keys():
                    m[label].append(bb)
                else:
                    m[label] = [bb]
            for cls in m.keys():
                if cls in classes:
                    bbs = np.array(m[cls])
                    if bbs.size == 0:
                        continue
                    _bbs = []
                    _bbs_min = []
                    for i in range(bbs.shape[0]):
                        w = float(bbs[i,2])
                        h = float(bbs[i,3])
                        if w *h < height*width*min_scale/1080/1920:
                            _bbs_min.append(bbs[i])
                        else: 
                            _bbs.append(bbs[i])
                    if len(_bbs_min) != 0:
                        ignore_bbs_all = np.vstack((ignore_bbs_all, np.array(_bbs_min)))
                        ignore_gt_classes = np.hstack(
                            (ignore_gt_classes, np.ones(len(_bbs_min)) * int(-1)))
                    if len(_bbs) != 0:
                        bbs_all = np.vstack((bbs_all, np.array(_bbs)))
                        gt_classes = np.hstack(
                            (gt_classes, np.ones(len(_bbs)) * _class_to_ind[cls]))
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

