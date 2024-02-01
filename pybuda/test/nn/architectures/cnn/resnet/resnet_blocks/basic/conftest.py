# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Conftest for parameters setup for resnet basic block
#


def pytest_addoption(parser):
    # input and output channels
    parser.addoption(
        "--basic_in_out_ch", 
        action="store", 
        default=64, 
        help="Input and output channel size in convolutional resnet basic block."
    )
    # kernel size
    parser.addoption(
        "--basic_kernel", 
        action="store", 
        default=3, 
        help="Kernel size in convolutional layers."
    )
    # original shape
    parser.addoption(
        "--basic_orig_shape", 
        action="store", 
        default='(32, 32)', 
        help="Shape of the activations matrix."
    )
    # stride
    parser.addoption(
        "--basic_stride", 
        action="store", 
        default=1, 
        help="Stride in convolutional layers."
    )
    # dilation
    parser.addoption(
        "--basic_dilation",
        action="store",
        default=1,
        help="Dilation in convolutional layers."
    )
    # depthwise
    parser.addoption(
        "--basic_depthwise",
        action="store",
        default=False,
        help="Apply depthwise convolution or not."
    )
    # bias
    parser.addoption(
        "--basic_bias",
        action="store",
        default=False,
        help="Use bias in convolution or not."
    )

def pytest_generate_tests(metafunc):

	option_in_out_ch = metafunc.config.option.basic_in_out_ch
	if 'basic_in_out_ch' in metafunc.fixturenames and option_in_out_ch is not None:
		metafunc.parametrize("basic_in_out_ch", [option_in_out_ch])

	option_kernel = metafunc.config.option.basic_kernel
	if 'basic_kernel' in metafunc.fixturenames and option_kernel is not None:
		metafunc.parametrize("basic_kernel", [option_kernel])

	option_orig_shape = metafunc.config.option.basic_orig_shape
	if 'basic_orig_shape' in metafunc.fixturenames and option_orig_shape is not None:
		metafunc.parametrize("basic_orig_shape", [option_orig_shape])

	option_stride = metafunc.config.option.basic_stride
	if 'basic_stride' in metafunc.fixturenames and option_stride is not None:
		metafunc.parametrize("basic_stride", [option_stride])

	option_dilation = metafunc.config.option.basic_dilation
	if 'basic_dilation' in metafunc.fixturenames and option_dilation is not None:
		metafunc.parametrize("basic_dilation", [option_dilation])

	option_depthwise = metafunc.config.option.basic_depthwise
	if 'basic_depthwise' in metafunc.fixturenames and option_depthwise is not None:
		metafunc.parametrize("basic_depthwise", [option_depthwise])

	option_bias = metafunc.config.option.basic_bias
	if 'basic_bias' in metafunc.fixturenames and option_bias is not None:
		metafunc.parametrize("basic_bias", [option_bias])
