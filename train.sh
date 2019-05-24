#! /bin/sh 

model_cfg=$1
fb_cfg=$2
theta_txt=$3
shift 3
work_dirs=$@
export CUDA_VISIBLE_DEVICES=2,3 
python tools/fbnet_train.py \
    --fb_cfg $fb_cfg \
    --model_cfg $model_cfg \
    --gpus 2 \
    --theta_txt $theta_txt \
    $work_dirs    
    #--theta_txt theta/epoch_30_end_arch_params.txt \
    #--work_dir ./work_dirs/car/search_fbnet2 \
