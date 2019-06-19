import re
import sys
import collections
import subprocess as sp
import matplotlib.pyplot as plt
import os
import numpy as np
import yaml
from easydict import EasyDict as edict
import time
import scipy

field_list = ["iter", "loss", "accuracy", "loss_bbox", "loss_cls", "rpn_loss_bbox", "rpn_loss_cls"]
field_tobe_list = ["accuracy", "loss_bbox", "loss_cls", "rpn_loss_bbox", "rpn_loss_cls"]
show_field_list = ["loss", "accuracy", "loss_bbox", "loss_cls", "rpn_loss_bbox", "rpn_loss_cls"]


def smooth(x, smooth_len=256):
    box = np.ones(smooth_len) / smooth_len
    x_smooth = np.convolve(x, box, mode='same')
    for i in range(smooth_len / 2 + 1):
        start = 0
        end = i + smooth_len / 2
        x_smooth[i] = np.sum(x[start:end]) * 1.0 / (end - start)
    for i in range(-1, -smooth_len / 2, -1):
        start = i - smooth_len / 2
        x_smooth[i] = np.sum(x[start:]) * 1.0 / (-start)
    return x_smooth

def cut_log(log):
    indexlist = []
    # testlist = []
    with open(log, 'r') as f:
        index = 0
        indextest = 0
        for line in f:
            if "Epoch" in line and "loss" in line:
                indexlist.append(index)
            index += 1
    # tmp_list = indexlist[1:]
    # tmp_list.append(index)
    # cut_indexlist = zip(indexlist, tmp_list)
    # return cut_indexlist
    return indexlist


def dowith_rpn_partlog(partlog):
    info = np.zeros((len(field_tobe_list), 30), dtype=np.float)

    field_cnt = np.zeros(len(field_tobe_list), dtype=np.int32)
    test_loss = []
    test_iter = []
    ii = 0
    c = 0
    print(partlog)
    for line in partlog:
        if "Epoch" in line and "loss" in line:
            lst = line.split()
            loss = float(lst[-1])
            print(loss)
        for index, field in enumerate(field_tobe_list):
            if field + " =" in line:
                info[index, field_cnt[index]] = float(line.split(field+ ' =')[1].split('(')[0])
                field_cnt[index] += 1

    info_sum = np.sum(info, 1)
    info_mean = info_sum * 1.0 / field_cnt
    info_list = [iter, loss]
    info_list.extend(info_mean)
    return info_list,test_iter,test_loss


def is_filter_partlog(partlog):
    flag = False

    tmpstr = "".join(partlog)
    if "Testing net" in tmpstr:
        flag = False
    return flag


def parse_log(log):
    lines = open(log, 'r').readlines()
    part_log_indexlist = cut_log(log)
    #all_info = np.zeros((len(part_log_indexlist), len(field_list)))
    cnt = 0
    loss_test = []
    for index in part_log_indexlist:
        partlog = lines[index].strip()
        loss = partlog.split('loss:')[1]
        loss_test.append(loss)
    return loss_test


def show_log(names, losses, is_show):
    plt.figure()
    for i in range(len(losses)):
        loss = [float(x) for x in losses[i]]
        plt.plot(loss,label=str(names[i]))
    legend = plt.legend(loc='upper right', shadow=True, fontsize='x-large')
    plt.ylim([0,1.0])
    plt.xlabel("iter x *")
    plt.show()


is_show= True
logs = sys.argv[1:]
losses = []
names = []
for log in logs:
    name = log.split("/")[-2]
    loss = parse_log(log)
    losses.append(loss)
    names.append(name)
show_log(names,losses, is_show)