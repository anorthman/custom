# fbnet detection search

## test_block_time (lat loss not bp in current code)
python  tools/test_time.py configs/search_config/bottlenetck_kernel.py 
## fbnet_search
./search.sh configs/model_config/retinanet_fpn.py configs/search_config/bottlenetck_kernel.py speed/bottlenetck_kernel.py_speed_gpu.txt
## fbnet_train (according to search_result)

