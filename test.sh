#! /bin/sh
 
fb_cfg=$1
theta_txt=$2
checkpoint=$3
shift 3
out=$@
python tools/fbnet_test.py \
    $checkpoint \
    --fb_cfg $fb_cfg \
    --model_cfg configs/model_config/fasterrcnn_str16.py \
    --theta_txt $theta_txt \
    $out #--out ./work_dirs/face/search_fbnet_epoch50/result.pkl \
#    --show 
    #--out ${theta_txt}.pkl
    #--theta_txt theta/bottlenetck_kernel_fasterrcnn_str16/fbent2_3x3.txt \
