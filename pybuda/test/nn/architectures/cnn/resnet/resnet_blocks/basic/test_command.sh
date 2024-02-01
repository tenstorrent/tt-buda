# 
# Commands for running resnet basic block
# 

# Run single test
# 
# To run using default parameters
# channels,               --basic_in_out_ch  --> 64
# kernel size,            --basic_kernel     --> 3
# original shape,         --basic_orig_shape --> (32, 32)
# stride,                 --basic_stride     --> 1
# dilation,               --basic_dilation   --> 1
# depthwise,              --basic_depthwise  --> False
# bias,                   --basic_bias       --> False
# pytest -svv test_basic_single.py

# # Issues
# pytest -svv test_basic_single.py --basic_in_out_ch 32 --basic_kernel 7 --basic_orig_shape '(32, 32)'
# pytest -svv test_basic_single.py --basic_in_out_ch 32 --basic_kernel 9 --basic_orig_shape '(512, 64)'
# pytest -svv test_basic_single.py --basic_in_out_ch 64 --basic_kernel 9 --basic_orig_shape '(80, 120)'

# pytest -svv test_basic_single.py --basic_in_out_ch 32 --basic_kernel 2 --basic_orig_shape '(512, 64)'
# pytest -svv test_basic_single.py --basic_in_out_ch 64 --basic_kernel 4 --basic_orig_shape '(512, 64)'
# pytest -svv test_basic_single.py --basic_in_out_ch 64 --basic_kernel 6 --basic_orig_shape '(512, 64)'

# pytest -svv test_basic_single.py --basic_in_out_ch 64 --basic_kernel 7 --basic_orig_shape '(512, 64)' --basic_bias True