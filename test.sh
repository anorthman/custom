#! /bin/sh
 
theta_txt=$1
checkpoint=$2
python tools/fbnet_test.py \
    $checkpoint \
    --fb_cfg configs/search_config/bottlenetck_kernel.py \
    --model_cfg configs/model_config/fasterrcnn_str16.py \
    --theta_txt $theta_txt \
    --show 
    #--out ${theta_txt}.pkl
    #--theta_txt theta/bottlenetck_kernel_fasterrcnn_str16/fbent2_3x3.txt \
