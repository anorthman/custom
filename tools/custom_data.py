import numpy as np

import h5py as h5
import os.path as osp
import pickle as pkl
import copy
import xmltodict

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
                    print(img_path)
                    label_path = "./data/"+label_path	
                    h = int(height)
                    w = int(width)
                    bbs, labels, bbs_ignore, labels_ignore = _load_gt(label_path, h, w, min_scale_percen)
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

def _load_gt(label_path, height, width, min_scale_percen):

    bbs_all = np.zeros((0, 4), dtype=np.float32)
    gt_classes = np.zeros((0), dtype=np.float32)
    ignore_bbs_all = np.zeros((0, 4), dtype=np.float32)
    ignore_gt_classes = np.zeros((0), dtype=np.float32)
    label_path = label_path.strip()
    area = height * width
    if label_path.endswith(".h5"):
        with h5.File(label_path, 'r') as f:
            for cls in f.keys():
                if cls in _classes:                     
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
                        if w *h < float(area)/float(min_scale_percen):
                            print("w",w,"h",h)  
                            _bbs_min.append(bbs[i])
                        else: 
                            _bbs.append(bbs[i])
                    print("min", len(_bbs_min), "left", len(_bbs))
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
    elif label_path.endswith(".xml"):
        with open(label_path, 'r') as f:
            print(label_path)
            d = xmltodict.parse(f.read())
            anno = d['annotation']
            if "object" not in anno.keys():
                return bbs_all,bbs_all,bbs_all,bbs_all
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
                if cls in _classes:
                    bbs = np.array(m[cls])
                    if bbs.size == 0:
                        continue
                    _bbs = []
                    _bbs_min = []
                    for i in range(bbs.shape[0]):
                        w = float(bbs[i,2])
                        h = float(bbs[i,3])
                        #bbs[i, 2] = bbs[i, 0] + w - 1 
                        #bbs[i, 3] = bbs[i, 1] + h - 1 
                        if w *h < float(area)/float(min_scale_percen):
                            _bbs_min.append(bbs[i])
                        else: 
                            _bbs.append(bbs[i])
                    print("min", len(_bbs_min), "left", len(_bbs))
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

if __name__ == '__main__':
    ignore_clsses = []
    # _classes = ["background","person_upper"]
    # img_label_size = "/home/zhaoyu/workspace/pytorch1.0/mmdetection/tools/convert_datasets/img_label_size"
    # _cache_path = "/home1/zhaoyu/dataset/wide.pkl"
    _classes = ["background", "face"]
    #_img_label_size = "/home/zhaoyu/workspace/pytorch1.0/mmdetection_/car_1img.txt"
    #_img_label_size = "/home1/zhaoyu/dataset/pbtc/raw/info/train_img_label_size"# "/home/zhaoyu/workspace/pytorch1.0/mmdetection_/car_1img.txt"
    #_img_label_size = "/home1/zhaoyu/dataset/pbtc/raw/info/train_img_label_size_fbnet_t"# "/home/zhaoyu/workspace/pytorch1.0/mmdetection_/car_1img.txt"
    #_cache_path = "/home1/zhaoyu/dataset/car_t.pkl"
    _img_label_size = "data/newlibraf_info/train_fov_tang_imglist"#/home1/zhaoyu//dataset/pbtc/raw/info/test_img_label_size_2000"
    _cache_path = "./data/newlibraf_info/train_fov_tang.pkl"
    num_classes = 2
    assert len(_classes) == int(num_classes)
    #_class_to_ind = dict(zip(_classes, [0,1,1]))
    _class_to_ind = dict(zip(_classes, range(num_classes)))
    min_scale_percen = 12960  
    main()
