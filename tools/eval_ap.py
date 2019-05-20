import numpy as np
import json
from xml.dom.minidom import parse
import xmltodict
from terminaltables import AsciiTable
import os
import h5py
import json
import cv2

def get_res(num_cls,txt_file,root_path,_ind_to_cate):
    all_image_res = []

    f = open(txt_file,'r')
    info = f.readlines()

    for i in range(len(info)):
        per_image_res = []
        for k in range(num_cls):
            cls_bbox = np.empty(shape=[0,4],dtype = float)
            per_image_res.append(cls_bbox)

        line = info[i].strip('\n')
        line = line.split()
        for j in range(len(line)):
            if line[j][-4:] == '.txt':
                res = line[j]
        res_file = root_path + res
        _res = open(res_file,'r')
        res = _res.readlines()
        bboxes = np.empty(shape=[0,4],dtype=float)
        label = []
        for k in range(len(res)):
            res_ = res[k].strip('\n').split()
            x1 = float(res_[0])
            y1 = float(res_[1])
            x2 = float(res_[2])
            y2 = float(res_[3])
            bboxes = np.append(bboxes,[[x1,y1,x2,y2]],axis=0)
            label.append(int(res_[4]))

        assert len(label)==len(bboxes)
        for m in range(len(bboxes)):
            cate = label[m]
            list_ind_cate = list(_ind_to_cate.keys())
            index = list_ind_cate.index(cate)
            box = bboxes[m]
            per_image_res[index] = np.append(per_image_res[index],[box],axis=0)
        all_image_res.append(per_image_res)
    return all_image_res

def get_gt(txt_file,root_path,_cate_to_ind,mini_size=30*30):
    all_gt_bbox = []
    all_gt_label = []
    all_gt_ignore = []

    f = open(txt_file,'r')
    gt_info = f.readlines()

    for i in range(len(gt_info)):
        per_image_gt_bbox = np.empty(shape=[0,4],dtype=float)
        per_image_gt_label = []
        per_image_gt_ignore = []

        line = gt_info[i].strip('\n').split()
        for j in range(1, 2):
            if line[j][-4:] == '.xml':
                gt = line[j]
                gt_file = root_path+gt

                import io
                with io.open(gt_file, 'r', encoding='ascii') as f:
                    d = xmltodict.parse(f.read())
                    anno = d['annotation']
                    objs = anno['object']
                    info = anno['size']
                    img_info = np.array([info['width'],info['height']],dtype=np.int32)
                    m = {}
                    if not isinstance(objs, list):
                        objs = [objs]
                    for obj in objs:
                        label = obj['name']
                        label = _cate_to_ind[label]
                        box = obj['bndbox']
                        x1 = float(box['xmin'])
                        y1 = float(box['ymin'])
                        x2 = float(box['xmax'])
                        y2 = float(box['ymax'])
                        bb = [x1, y1, x2, y2]
                        bb = [int(x) for x in bb]
                        bb = [bb]
                        if label in list(m.keys()):
                            m[label] = np.vstack((m[label],np.array(bb)))
                        else:
                            m[label] = np.array(bb)
                    image_size = img_info
                    image_width = image_size[0]
                    image_height = image_size[1]
            elif line[j].endswith('.h5'):
                m = dict()
                h5data = h5py.File(line[j],'r')
                for key in h5data.keys():
                    if key == "img_info":
                        image_width = h5data[key][0]
                        image_height = h5data[key][1]
                    else:
                        label = _cate_to_ind[key] 
                        m[label] = h5data[key][:]
                        m[label][:,3] += m[label][:,1]
                        m[label][:,2] += m[label][:,0]
                h5data.close()
            else:
                # pass
                print('only support xml/h5 file gtfiles')
        m_key = list(m.keys())
        for k in range(len(m_key)):
            label = m_key[k]
            box = m[label]

            per_image_gt_bbox = np.append(per_image_gt_bbox,box,axis=0)
            per_image_gt_label = per_image_gt_label+[label]*len(box)

        per_image_gt_label = np.array(per_image_gt_label)
        for b in range(len(per_image_gt_bbox)):
            bbox = list(per_image_gt_bbox[b])
            b_x1 = bbox[0]
            b_y1 = bbox[1]
            b_x2 = bbox[2]
            b_y2 = bbox[3]
            area = (b_x2-b_x1+1)*(b_y2-b_y1+1)
            if mini_size ==0:
                per_image_gt_ignore.append(0)
                
            else:
                #if area<(image_width*image_height*100)/(1920*1080):
                if area<(image_width*image_height)/(100):
                    per_image_gt_ignore.append(1)
                else:
                    per_image_gt_ignore.append(0)

        per_image_gt_ignore = np.array(per_image_gt_ignore)
        all_gt_bbox.append(per_image_gt_bbox)
        all_gt_label.append(per_image_gt_label)
        all_gt_ignore.append(per_image_gt_ignore)

    return all_gt_bbox, all_gt_label, all_gt_ignore


def compute_iou(box, boxes, maxDets = 500,iou = 0.5):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    #print(box,boxes[0,:])
    #exit()
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    #print(iou)
    return iou

def bbox_overlaps(bboxes1, bboxes2, mode='iou'):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)

    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
        bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
        bboxes2[:, 3] - bboxes2[:, 1] + 1)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(
            y_end - y_start + 1, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious

def tpfp_imagenet(det_bboxes,
                gt_bboxes,
                gt_ignore,
                default_iou_thr,
                area_ranges=None):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bbox (ndarray): the detected bbox -->(x1,y1,x2,y2)
        gt_bboxes (ndarray): ground truth bboxes of this image
        gt_ignore (ndarray): indicate if gts are ignored for evaluation or not
        default_iou_thr (float): the iou thresholds for medium and large bboxes
        area_ranges (list or None): gt bbox area ranges

    Returns:
        tuple: two arrays (tp, fp) whose elements are 0 and 1
    """
    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp
    # of a certain scale.
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0] + 1) * (
                det_bboxes[:, 3] - det_bboxes[:, 1] + 1)
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
        return tp, fp
    ious = bbox_overlaps(det_bboxes, gt_bboxes )
    gt_w = gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1
    gt_h = gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1
    iou_thrs = np.minimum((gt_w * gt_h) / ((gt_w + 10.0) * (gt_h + 10.0)),
                                                default_iou_thr)
    # sort all detections by scores in descending order
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore, dtype=bool)
        else:
            gt_areas = gt_w * gt_h
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
        for i in sort_inds:
            max_iou = -1
            matched_gt = -1
            # find best overlapped available gt
            for j in range(num_gts):
                # different from PASCAL VOC: allow finding other gts if the
                # best overlaped ones are already matched by other det bboxes
                if gt_covered[j]:
                    continue
                elif ious[i, j] >= iou_thrs[j] and ious[i, j] > max_iou:
                    max_iou = ious[i, j]
                    matched_gt = j
            # there are 4 cases for a det bbox:
            # 1. it matches a gt, tp = 1, fp = 0
            # 2. it matches an ignored gt, tp = 0, fp = 0
            # 3. it matches no gt and within area range, tp = 0, fp = 1
            # 4. it matches no gt but is beyond area range, tp = 0, fp = 0
            if matched_gt >= 0:
                gt_covered[matched_gt] = 1
                if not (gt_ignore[matched_gt] or gt_area_ignore[matched_gt]):
                    tp[k, i] = 1
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :4]
                area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return tp, fp

def get_cls_results(det_results, gt_bboxes, gt_labels, gt_ignore, class_id):
    """Get det results and gt information of a certain class."""
    cls_dets = [det[class_id]
                for det in det_results]  # det bboxes of this class
    cls_gts = []  # gt bboxes of this class
    cls_gt_ignore = []
    for j in range(len(gt_bboxes)):
        gt_bbox = gt_bboxes[j]
        cls_inds = (gt_labels[j] == class_id )
        cls_gt = gt_bbox[cls_inds, :] if gt_bbox.shape[0] > 0 else gt_bbox
        cls_gts.append(cls_gt)
        if gt_ignore is None:
            cls_gt_ignore.append(np.zeros(cls_gt.shape[0], dtype=np.int32))
        else:
            cls_gt_ignore.append(gt_ignore[j][cls_inds])
    return cls_dets, cls_gts, cls_gt_ignore

def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (ndarray): shape (num_scales, num_dets) or (num_dets, )
        precisions (ndarray): shape (num_scales, num_dets) or (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or ndarray: calculated average precision
    """
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    if no_scale:
        ap = ap[0]
    return ap

def eval_map(det_results,
            gt_bboxes,
            gt_labels,
            gt_ignore=None,
            scale_ranges=None,
            iou_thr=0.5,
            dataset=None,
            print_summary=True):
    """Evaluate mAP of a dataset.

    Args:
        det_results (list): a list of list, [[cls1_det, cls2_det, ...], ...]
        gt_bboxes (list): ground truth bboxes of each image, a list of K*4 array.
        gt_labels (list): ground truth labels of each image, a list of K array
        gt_ignore (list): gt ignore indicators of each image, a list of K array
        scale_ranges (list, optional): [(min1, max1), (min2, max2), ...]
        iou_thr (float): IoU threshold
        dataset (None or str or list): dataset name or dataset classes, there
            are minor differences in metrics for different datsets, e.g.
            "voc07", "imagenet_det", etc.
        print_summary (bool): whether to print the mAP summary

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(gt_bboxes) == len(gt_labels)
    if gt_ignore is not None:
        assert len(gt_ignore) == len(gt_labels)
        for i in range(len(gt_ignore)):
            assert len(gt_labels[i]) == len(gt_ignore[i])
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                                if scale_ranges is not None else None)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    eval_results = []
    num_classes = len(det_results[0])  # positive class num
    #print('num_classes is : ' , num_classes)
    gt_labels = [ label if label.ndim == 1 else label[:, 0] for label in gt_labels]

    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gt_ignore = get_cls_results(
            det_results, gt_bboxes, gt_labels, gt_ignore, i)

        # calculate tp and fp for each image
        tpfp_func = tpfp_imagenet
        tpfp = [
            tpfp_func(cls_dets[j], cls_gts[j], cls_gt_ignore[j], iou_thr,
                area_ranges) for j in range(len(cls_dets))
        ]

        tp, fp = tuple(zip(*tpfp))
        # calculate gt number of each scale, gts ignored or beyond scale
        # are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for j, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += np.sum(np.logical_not(cls_gt_ignore[j]))
            else:
                gt_areas = (bbox[:, 2] - bbox[:, 0] + 1) * (
                    bbox[:, 3] - bbox[:, 1] + 1)
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum(
                        np.logical_not(cls_gt_ignore[j]) &
                        (gt_areas >= min_area) & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' 
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = [
            all_ap[all_num_gts[:, i] > 0, i].mean()
            if np.any(all_num_gts[:, i] > 0) else 0.0
            for i in range(num_scales)
        ]
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0
    if print_summary:
        print_map_summary(mean_ap, eval_results, dataset)

    return mean_ap, eval_results

def print_map_summary(mean_ap, results, dataset=None):
    """Print mAP and results of each class.

    Args:
        mean_ap(float): calculated from `eval_map`
        results(list): calculated from `eval_map`
        dataset(None or str or list): dataset name or dataset classes.
    """
    num_scales = len(results[0]['ap']) if isinstance(results[0]['ap'],np.ndarray) else 1
    num_classes = len(results)

    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    precisions = np.zeros((num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    for i, cls_result in enumerate(results):
        if cls_result['recall'].size > 0:
            recalls[:, i] = np.array(cls_result['recall'], ndmin=2)[:, -1]
            precisions[:, i] = np.array(
                cls_result['precision'], ndmin=2)[:, -1]
        aps[:, i] = cls_result['ap']
        num_gts[:, i] = cls_result['num_gts']

    if dataset is None:
        label_names = [str(i) for i in range(1, num_classes + 1)]
    elif mmcv.is_str(dataset):
        label_names = get_classes(dataset)
    else:
        label_names = dataset

    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]
    header = ['class', 'gts', 'dets', 'recall', 'precision', 'ap']
    for i in range(num_scales):
        table_data = [header]
        for j in range(num_classes):
            row_data = [
                label_names[j], num_gts[i, j], results[j]['num_dets'],
                '{:.3f}'.format(recalls[i, j]), '{:.3f}'.format(
                    precisions[i, j]), '{:.3f}'.format(aps[i, j])
            ]
            table_data.append(row_data)
        table_data.append(['mAP', '', '', '', '', '{:.3f}'.format(mean_ap[i])])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True

        # print(table.table)


def get_image_name_list(gt_txt,res_root):
    image_list_1 = []
    f = open(gt_txt,'r')
    js = f.readlines()
    f.close()
    for i in range(len(js)):
        info = json.loads(js[i])
        image_list_1.append(info['image'])


    image_list_2 = []
    txt_files = os.listdir(res_root)
    for i in range(len(txt_files)):
        image = txt_files[i].strip('.txt')+('.jpg')
        image_list_2.append(image)

    if len(image_list_1) >= len(image_list_2):
        image_list = image_list_2
    else:
        image_list = image_list_1

    return image_list


def summary_all_results(cat,results,thr_iou):
    header = ['thr_iou']
    for i in range(len(cat)):
        #header.append('class_' + str(cat[i]))
        header.append(str(cat[i]))

    table_data = [header]
    for j in range(len(thr_iou)):
        row_data = [thr_iou[j]]
        for k in  range(len(cat)):
            row_data.append(round(results[j,k],2))

        table_data.append(row_data)


    mAP = ['mAP']
    for k in range(np.shape(results)[1]):
        mAP.append(round(np.mean(results[:,k]),3))
    table_data.append(mAP)
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True

    print(table.table)
