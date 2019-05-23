import numpy as np
from eval_ap import *

import pickle 
import os
import argparse

def pkl2txt(pkl,gt_pickle):
	print(pkl, gt_pickle)
	with open(gt_pickle,"rb") as f:
		gt = pickle.load(f)
	with open(pkl,"rb") as f:
		data = pickle.load(f)
	assert len(gt) == len(data), "gt and result must have same image_nums"
	filelist = open('filelist',"w")
	root_txt = pkl.split('.')[0] + "_result/"
	if not os.path.exists(root_txt):
		os.makedirs(root_txt)
	for i in range(len(data)):
		with open(root_txt+str(i)+'.txt',"w") as g:
			filelist.write(root_txt+str(i)+'.txt'+" ")
			for j in range(data[i][0].shape[0]):
				a = [g.write(str(data[i][0][j][k])+" ") for k in range(4)]
				g.write(str(0)+" ")
				g.write(str(data[i][0][j][-1])+"\n")
			filelist.write(gt[i]['filename'].replace('jpg','xml')+"\n")
	filelist.close()

def main():
	# txt_file = './data/500res_gt.txt'
	# root_path = '/media/clx/DATA/chelixuan/chelixuan_2019/0516eval_universial/'
	# num_cls = 6
	# _ind_to_cate = {0: 'person_normal', 1: 'visperson_normal', 2: 'person_cycling', 3: 'person_others', 4: 'visperson_cycling', 5: 'ignore'}
	
	txt_file = 'filelist'
	root_path = './'
	num_cls = 1
	#num_cls = 4
	# _ind_to_cate = {0: 'person'}
	_ind_to_cate = {0: 'face'} #,1: 'car', 2: 'bicycle',3:'tricycle'}
	#_ind_to_cate = {0: 'person',1: 'car', 2: 'bicycle',3:'tricycle'}

	_cate_to_ind = {value:key for key,value in _ind_to_cate.items()}

	all_iou_thr = [0.4,0.45]

	all_image_res = get_res(num_cls,txt_file,root_path,_ind_to_cate)
	root_path = './data/'
	all_gt_bbox, all_gt_label, all_gt_ignore = get_gt(txt_file,root_path,_cate_to_ind,mini_size=30*30)

	map_results = np.empty(shape=[len(all_iou_thr),num_cls])

	for i in range(len(all_iou_thr)):
		mean_ap, eval_results = eval_map(all_image_res,all_gt_bbox,all_gt_label,iou_thr=all_iou_thr[i],gt_ignore=None)
		# mean_ap, eval_results = eval_map(all_image_res,all_gt_bbox,all_gt_label,iou_thr=all_iou_thr[i],gt_ignore=all_gt_ignore)
		for j in range(num_cls):
			map_results[i,j] = eval_results[j]['ap']
	
	cat = list(_cate_to_ind.keys())
	summary_all_results(cat,map_results,thr_iou= all_iou_thr)

def parse_args():

    parser = argparse.ArgumentParser(description="Test model return ap")
    parser.add_argument('det_pkl', type=str, default=None,
                        help='det_reslut data from mmdet')
    parser.add_argument('gt_pkl', type=str, default=None,
                        help='custom data for mmdet')


    args = parser.parse_args()
    return args
if __name__ == '__main__':
	args = parse_args()
	#pkl2txt('theta/epoch_30_end_arch_params.txt.pkl','/home1/zhaoyu/dataset/testcar.pkl')
	pkl2txt(args.det_pkl,args.gt_pkl)
	main()


