# fbnet detection search

## Introduction
FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search(https://arxiv.org/abs/1812.03443)
The master branch works with (https://github.com/open-mmlab/mmdetection)edb03937964b583a59dd1bddf76eaba82df9e8c0

- **test_block_time**

python  tools/test_time.py configs/search_config/bottlenetck_kernel.py 

- **fbnet_search**

./search.sh configs/model_config/retinanet_fpn.py configs/search_config/bottlenetck_kernel.py speed/bottlenetck_kernel.py_speed_gpu.txt

- **fbnet_train** (according to search_result)

./train.sh configs/model_config/retinanet_fpn.py configs/search_config/bottlenetck_kernel.py theta/bottlenetck_kernel_retinanet_fpn/base.txt --work_dir ./your_path_tosave

- **fbnet_test** (according to train_result)

./test.sh configs/model_config/retinanet_fpn.py configs/search_config/bottlenetck_kernel.py theta/bottlenetck_kernel_retinanet_fpn/base.txt your_path_tosave/lastest.pth --show
