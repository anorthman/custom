import _init_paths
import time
import argparse
from mmcv import Config
import torch

from backbone.Unit import BASICUNIT
from torch import nn

class timenet(nn.Module):
	def __init__(self, scale, unit, iters):
		super(timenet, self).__init__()
		self.scale = scale
		self.unit = unit
		self.iters = iters
	def forward(self, x):
		x = self.unit(x)
		return x 
	def test_time(self,is_cuda):
		print(self.scale)
		if is_cuda:
			inputs = torch.randn((self.scale)).cuda()
		else:
			inputs = torch.randn((self.scale))
		torch.cuda.synchronize()
		start = time.time()
		for i in range(self.iters):
			y = self.forward(inputs)
		torch.cuda.synchronize()
		end = time.time()
		print((end-start)/self.iters)
		return ((end-start)/self.iters)

parser = argparse.ArgumentParser(description="Test a block time .")
parser.add_argument('--fb_cfg', type=str, default=None,
					help='config contains of all blocks.')
parser.add_argument("--gpu", action='store_true',
					help="test time with gpu or cpu")
parser.add_argument("--skip", type=bool, default=True,
					help="add skip or not")
parser.add_argument("--scale", type=list, default=[1,3,108,192],
					help="input of the model")

args = parser.parse_args()
search_cfg = Config.fromfile(args.fb_cfg)
_space = search_cfg.search_space
#base = _space['base']
depth = _space['depth']
space = _space['space']
skip = args.skip
scale = args.scale
is_cuda = args.gpu
print(is_cuda)

writefile = "./speed/"+args.fb_cfg.split("/")[-1]+"_speed_gpu.txt" if is_cuda else "./speed/"+args.fb_cfg.split("/")[-1]+"_speed_cpu.txt"
with open(writefile, "w") as f:
	for i in range(len(depth)):
		scale[2],scale[3] = int(scale[2]/2), int(scale[3]/2)
		for j in range(depth[i]):	   
			#blocks = nn.ModuleList()
			if j!=0 and skip:
				net = timenet(scale, BASICUNIT['Identity'](), 1000)
				#net = net.cuda()
				times = net.test_time(is_cuda)
				f.write(str(times*1000)+" ")  
			for unit in space:
				print("^^^^^^^^")
				if j == 0:				
					unit['param']['stride'] = 2
					unit['param']['_in'] = unit['param']['_in']
					unit['param']['_out'] = unit['param']['_out']*2
				else:
					unit['param']['stride'] = 1
					unit['param']['_in'] = unit['param']['_out']
				scale[1] = unit['param']['_in']
				print(unit)
				if is_cuda:
					net = timenet(scale, BASICUNIT[unit['type']](**unit['param']), 1000).cuda()
				else:
					net = timenet(scale, BASICUNIT[unit['type']](**unit['param']), 1000)
				times = net.test_time(is_cuda)
				f.write(str(times*1000)+" ")
			f.write('\n')





