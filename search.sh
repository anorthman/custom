#! /bin/sh 

model_cfg=$1
fb_cfg=$2
speed_txt=$3
python tools/fbnet_search.py \
    --model_cfg $model_cfg \
    --fb_cfg $fb_cfg \
    --speed_txt $speed_txt \
    2>&1 | tee search.log
    #--fb_cfg configs/search_config/bottlenetck_kernel.py \
    #--speed_txt speed/bottlenetck_kernel.py_speed_gpu.txt
