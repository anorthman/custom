#! /bin/sh 

fb_cfg=$1
theta_txt=$2
export CUDA_VISIBLE_DEVICES=2,3 
python tools/fbnet_train.py \
    --fb_cfg $fb_cfg \
    --model_cfg configs/model_config/fasterrcnn_str16.py \
    --gpus 2 \
    --theta_txt $theta_txt \
    --work_dir ./work_dirs/face/search_fbnet_bottleneckbase3x3 \
    #--theta_txt theta/epoch_30_end_arch_params.txt \
    #--work_dir ./work_dirs/car/search_fbnet2 \
