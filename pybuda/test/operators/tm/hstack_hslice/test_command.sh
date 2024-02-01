# 
# Commands for running hstack and hslice tests
# 

# Run single test
# 
# To run using default parameters
# model,     --tmh_model     --> model_1, Note: for tm horizontal ops we have 5 models, model_[1-5]
# training,  --tmh_train     --> True
# recompute, --tmh_recompute --> True
# shape,     --tmh_shape     --> [1, 16, 32, 64]
# slice,     --tmh_slice     --> 4
pytest -svv test_hstack_hslice_single.py

# Few examples with passed arguments
pytest -svv test_hstack_hslice_single.py --tmh_model model_3 --tmh_train True --tmh_recompute True --tmh_shape '[1, 32, 96, 128]' --tmh_slice 8
pytest -svv test_hstack_hslice_single.py --tmh_model model_1 --tmh_train False --tmh_recompute True --tmh_shape '[1, 32, 256, 128]'
pytest -svv test_hstack_hslice_single.py --tmh_model model_2 --tmh_train True --tmh_recompute False
pytest -svv test_hstack_hslice_single.py --tmh_model model_5 --tmh_train False
pytest -svv test_hstack_hslice_single.py --tmh_model model_4 --tmh_shape '[1, 32, 256, 2048]'