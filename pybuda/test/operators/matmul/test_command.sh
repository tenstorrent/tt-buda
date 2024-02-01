# 
# Commands for running matmul tests
# 

# Run single test
# 
# To run using default parameters
# model,     --mm_model     --> model_1, Note: for matmul ops we have 7 models, model_[1-3, 6-8, 10]
# training,  --mm_train     --> True
# recompute, --mm_recompute --> True
# shape,     --mm_shape     --> [1, 16, 32, 64]
pytest -svv test_matmul_single.py

# Few examples with passed arguments
pytest -svv test_matmul_single.py --mm_model model_3 --mm_train True --mm_recompute True --mm_shape '[1, 32, 96, 128]'
pytest -svv test_matmul_single.py --mm_model model_1 --mm_train False --mm_recompute True --mm_shape '[1, 32, 256, 128]'
pytest -svv test_matmul_single.py --mm_model model_2 --mm_train True --mm_recompute False
pytest -svv test_matmul_single.py --mm_model model_7 --mm_train False
pytest -svv test_matmul_single.py --mm_model model_6 --mm_shape '[1, 32, 256, 2048]'

# Issues
pytest -svv test_matmul_single.py --mm_model model_2 --mm_train False --mm_recompute True --mm_shape '[38, 236, 141, 73]'
pytest -svv test_matmul_single.py --mm_model model_2 --mm_train False --mm_recompute True --mm_shape '[138, 204, 134, 134]'

pytest -svv test_matmul_single.py --mm_model model_2 --mm_train False --mm_recompute True --mm_shape '[1, 1, 8192, 8192]'
