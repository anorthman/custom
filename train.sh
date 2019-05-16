#! /bin/sh 


export CUDA_VISIBLE_DEVICES=0 
python tools/fbnet_train.py \
    --fb_cfg configs/search_config/bottlenetck_kernel.py \
    --model_cfg configs/model_config/fasterrcnn_str16.py \
    --gpus 1 \
    --theta_txt theta/bottlenetck_kernel_fasterrcnn_str16/fbent2_3x3.txt \
    --work_dir ./work_dirs/car/search_fbnet2_3x3base \
    #--theta_txt theta/epoch_30_end_arch_params.txt \
    #--work_dir ./work_dirs/car/search_fbnet2 \
