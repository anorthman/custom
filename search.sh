#! /bin/sh 

fb_cfg=$1
speed_txt=$2
python tools/fbnet_search.py \
    --model_cfg configs/model_config/fasterrcnn_str16.py \
    --fb_cfg $fb_cfg \
    --speed_txt $speed_txt \
    2>&1 | tee search.log
    #--fb_cfg configs/search_config/bottlenetck_kernel.py \
    #--speed_txt speed/bottlenetck_kernel.py_speed_gpu.txt
