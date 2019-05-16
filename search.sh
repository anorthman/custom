#! /bin/sh 

python tools/fbnet_search.py \
    --fb_cfg configs/search_config/bottlenetck_kernel.py \
    --model_cfg configs/model_config/fasterrcnn_str16.py \
    --speed_txt speed/bottlenetck_kernel.py_speed_gpu.txt
    2>&1 | tee search.log
