#! /bin/sh
 
model_cfg=$1
fb_cfg=$2
theta_txt=$3
checkpoint=$4
shift 4
out=$@
python tools/fbnet_test.py \
    $checkpoint \
    --fb_cfg $fb_cfg \
    --model_cfg $model_cfg \
    --theta_txt $theta_txt \
    $out #--out ./work_dirs/face/search_fbnet_epoch50/result.pkl \
