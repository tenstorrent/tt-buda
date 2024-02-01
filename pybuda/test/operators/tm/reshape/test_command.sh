# 
# Commands for running reshape tests
# 

# Run single test
# 
# To run using default parameters
# model,     --resh_model     --> model_1, Note: for reshape ops we have 5 models, model_[1-5]
# training,  --resh_train     --> True
# recompute, --resh_recompute --> True
# old shape, --resh_oshape    --> [1, 16, 32, 64]
# new shape, --resh_nshape    --> [1, 32, 32, 32]
pytest -svv test_reshape_single.py

# Few examples with passed arguments
pytest -svv test_reshape_single.py --resh_model model_3 --resh_train True --resh_recompute True --resh_oshape '[1, 32, 96, 128]' --resh_nshape '[1, 32, 192, 64]'
pytest -svv test_reshape_single.py --resh_model model_1 --resh_train False --resh_recompute True --resh_nshape '[1, 32, 256, 128]' --resh_oshape '[2, 32, 128, 128]'
pytest -svv test_reshape_single.py --resh_model model_2 --resh_train True --resh_recompute False
pytest -svv test_reshape_single.py --resh_model model_5 --resh_train False
pytest -svv test_reshape_single.py --resh_model model_4 --resh_oshape '[1, 32, 256, 2048]' --resh_nshape '[1, 32, 512, 1024]'