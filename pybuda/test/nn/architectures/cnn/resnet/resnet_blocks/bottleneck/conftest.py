# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Conftest for parameters setup for resnet bottleneck block
#

def pytest_addoption(parser):
    # training
    parser.addoption(
        "--bneck_train",
        action="store",
        default=True,
        help="Run the training or not."
    )
    # recompute
    parser.addoption(
        "--bneck_recompute",
        action="store",
        default=False,
        help="Recompute or not."
    )
    # input and output channels
    parser.addoption(
        "--bneck_in_out_ch", 
        action="store", 
        default=64, 
        help="Input and output channel size in convolutional resnet basic block."
    )
    # in between channels
    parser.addoption(
        "--bneck_inbtw_ch",
        action="store",
        default=32,
        help="Input and output channels in convolutional layers inside the block."
    )
    # original shape
    parser.addoption(
        "--bneck_orig_shape", 
        action="store", 
        default='(32, 32)', 
        help="Shape of the activations matrix."
    )
    # stride
    parser.addoption(
        "--bneck_stride", 
        action="store", 
        default=1, 
        help="Stride in convolutional layers."
    )
    # dilation
    parser.addoption(
        "--bneck_dilation",
        action="store",
        default=1,
        help="Dilation in convolutional layers."
    )
    # depthwise
    parser.addoption(
        "--bneck_depthwise",
        action="store",
        default=False,
        help="Apply depthwise convolution or not."
    )
    # bias
    parser.addoption(
        "--bneck_bias",
        action="store",
        default=False,
        help="Use bias in convolution or not."
    )

def pytest_generate_tests(metafunc):

	option_train = metafunc.config.option.bneck_train
	if 'bneck_train' in metafunc.fixturenames and option_train is not None:
		metafunc.parametrize("bneck_train", [option_train])

	option_recompute = metafunc.config.option.bneck_recompute
	if 'bneck_recompute' in metafunc.fixturenames and option_recompute is not None:
		metafunc.parametrize("bneck_recompute", [option_recompute])

	option_bneck_inbtw_ch = metafunc.config.option.bneck_inbtw_ch
	if 'bneck_inbtw_ch' in metafunc.fixturenames and option_bneck_inbtw_ch is not None:
		metafunc.parametrize("bneck_inbtw_ch", [option_bneck_inbtw_ch])

	option_in_out_ch = metafunc.config.option.bneck_in_out_ch
	if 'bneck_in_out_ch' in metafunc.fixturenames and option_in_out_ch is not None:
		metafunc.parametrize("bneck_in_out_ch", [option_in_out_ch])

	option_orig_shape = metafunc.config.option.bneck_orig_shape
	if 'bneck_orig_shape' in metafunc.fixturenames and option_orig_shape is not None:
		metafunc.parametrize("bneck_orig_shape", [option_orig_shape])

	option_stride = metafunc.config.option.bneck_stride
	if 'bneck_stride' in metafunc.fixturenames and option_stride is not None:
		metafunc.parametrize("bneck_stride", [option_stride])

	option_dilation = metafunc.config.option.bneck_dilation
	if 'bneck_dilation' in metafunc.fixturenames and option_dilation is not None:
		metafunc.parametrize("bneck_dilation", [option_dilation])

	option_depthwise = metafunc.config.option.bneck_depthwise
	if 'bneck_depthwise' in metafunc.fixturenames and option_depthwise is not None:
		metafunc.parametrize("bneck_depthwise", [option_depthwise])

	option_bias = metafunc.config.option.bneck_bias
	if 'bneck_bias' in metafunc.fixturenames and option_bias is not None:
		metafunc.parametrize("bneck_bias", [option_bias])