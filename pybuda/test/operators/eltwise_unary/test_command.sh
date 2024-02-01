# 
# Commands for running element-wise unary tests
# 

# Run single test
# 
# To run using default parameters
# model,     --un_model     --> model_1, Note: for binary ops we have 10 models, model_[1-10]
# training,  --un_train     --> True
# recompute, --un_recompute --> True
# shape,     --un_shape     --> [1, 16, 32, 64]
# operation, --un_op        --> Exp
pytest -svv test_eltwise_unary_single.py

# Few examples with passed arguments
pytest -svv test_eltwise_unary_single.py --un_model model_3 --un_train True --un_recompute True --un_shape '[1, 32, 96, 128]' --un_op 'Log'
pytest -svv test_eltwise_unary_single.py --un_model model_1 --un_train False --un_recompute True --un_shape '[1, 32, 256, 128]'
pytest -svv test_eltwise_unary_single.py --un_model model_2 --un_train True --un_recompute False
pytest -svv test_eltwise_unary_single.py --un_model model_5 --un_train False --un_op 'Gelu'
pytest -svv test_eltwise_unary_single.py --un_model model_4 --un_shape '[1, 32, 256, 2048]'

# Issues
pytest -svv test_eltwise_unary_single.py --un_model model_4 --un_train True --un_recompute False --un_op 'Exp' --un_shape '[21, 127, 102, 19]'


# pytest -svv pybuda/test/operators/eltwise_unary/test_eltwise_unary_single.py --un_model model_6 --un_train True --un_recompute False --un_op 'Relu' --un_shape '[1, 12, 13]'
# pytest -svv pybuda/test/operators/eltwise_unary/test_eltwise_unary_single.py --un_model model_7 --un_train True --un_recompute True --un_op 'Exp' --un_shape '[1, 12, 13]'