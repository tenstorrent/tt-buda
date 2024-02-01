# 
# Commands for running reduce tests
# 

# Run single test
# 
# To run using default parameters
# model,           --red_model     --> model_1, Note: for matmul ops we have 5 models, model_[1-5]
# training,        --red_train     --> True
# recompute,       --red_recompute --> True
# shape,           --red_shape     --> [1, 16, 32, 64]
# operation,       --red_op        --> 'ReduceSum'
# dimension,       --red_dim       --> 2
# keep dimensions, --red_keepdim   --> True
pytest -svv test_reduce_nd_single.py

# Few examples with passed arguments
pytest -svv test_reduce_nd_single.py --red_model model_3 --red_train True --red_recompute True --red_shape '[1, 32, 96, 128]'
pytest -svv test_reduce_nd_single.py --red_model model_1 --red_train False --red_recompute True --red_shape '[1, 32, 256, 128]'
pytest -svv test_reduce_nd_single.py --red_model model_2 --red_train True --red_recompute False --red_op 'ReduceSum' --red_dim 1
pytest -svv test_reduce_nd_single.py --red_model model_5 --red_train False --red_keepdim True
pytest -svv test_reduce_nd_single.py --red_model model_4 --red_shape '[1, 32, 256, 2048]' --red_keepdim False
