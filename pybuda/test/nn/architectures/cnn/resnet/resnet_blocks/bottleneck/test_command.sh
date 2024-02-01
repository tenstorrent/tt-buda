# 
# Commands for running resnet bottleneck block
# 

# Run single test
# 
# To run using default parameters
# training,               --bneck_train      --> True
# recompute,              --bneck_recompute  --> False 
# channels                --bneck_in_out_ch  --> 64
# in between channels,    --bneck_inbtw_ch   --> 32
# original shape,         --bneck_orig_shape --> (32, 32)
# stride,                 --bneck_stride     --> 1
# dilation,               --bneck_dilation   --> 1
# depthwise,              --bneck_depthwise  --> False
# bias,                   --bneck_bias       --> False
# pytest -svv test_bottleneck_single.py

# # Examples for runnning bottleneck single test
# pytest -svv test_bottleneck_single.py --bneck_train True --bneck_recompute False --bneck_in_out_ch 256 --bneck_inbtw_ch 128
# pytest -svv test_bottleneck_single.py --bneck_train False --bneck_recompute False --bneck_orig_shape '(64, 128)' --bneck_bias False
# pytest -svv test_bottleneck_single.py --bneck_inbtw_ch 256 --bneck_orig_shape '(128, 128)' --bneck_depthwise False --bneck_stride 1
# pytest -svv test_bottleneck_single.py --bneck_in_out_ch 64 --bneck_inbtw_ch 128 --bneck_orig_shape '(320, 480)' --bneck_bias False
# pytest -svv test_bottleneck_single.py --bneck_recompute False --bneck_in_out_ch 90 --bneck_orig_shape '190, 230' --bneck_dilation 1
# pytest -svv test_bottleneck_single.py --bneck_inbtw_ch 80 --bneck_orig_shape '(170, 90)' --bneck_stride 1 --bneck_depthwise False --bneck_bias False

# # Issues
# pytest -svv test_bottleneck_single.py --bneck_train True