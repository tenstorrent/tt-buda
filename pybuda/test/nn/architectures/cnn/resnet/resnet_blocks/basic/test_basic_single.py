# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#

# import pytest

# import torch

# import pybuda
# import pybuda.op

# from pybuda import (
#     PyBudaModule,
#     TTDevice,
#     TTDeviceType,
#     Tensor,
#     pybuda_compile,
#     CompilerConfig,
#     VerifyConfig,
# )

# from pybuda.op.nn import Conv2dModule
# from pybuda.op.convolution import Conv2d
# from pybuda.utils import align_up_tile
# from pybuda.op.eval import compare_tensor_to_golden

# from . import TestBasicBlock


# @pytest.mark.xfail
# def test_basic_block(
#     basic_in_out_ch,
#     basic_kernel,
#     basic_orig_shape,
#     basic_stride,
#     basic_dilation,
#     basic_depthwise,
#     basic_bias
# ):

#     in_out_channels = int(basic_in_out_ch)
#     kernel_size = int(basic_kernel)
#     original_shape = eval(basic_orig_shape)
#     stride = int(basic_stride)
#     dilation = int(basic_dilation)
#     depthwise = bool(basic_depthwise)
#     bias = bool(basic_bias)

#     model = TestBasicBlock(
#         in_channels=in_out_channels,
#         out_channels=in_out_channels,
#         kernel_size=kernel_size,
#         stride=stride,
#         dilation=dilation,
#         depthwise=depthwise,
#         bias=bias
#     )

#     tt0 = TTDevice("tt0", devtype=TTDeviceType.Golden)
#     tt0.place_module(model)

#     activations = Tensor.create_from_torch(
#         torch.rand(
#             1, 
#             in_out_channels,
#             original_shape[0], 
#             original_shape[1], 
#             requires_grad=True
#         )
#     )
#     _, _, _, outputs, _ = pybuda_compile(
#         tt0,
#         "conv2d",
#         activations,
#         compiler_cfg=CompilerConfig(
#             enable_training=False,
#             enable_recompute=False
#         ),
#         verify_cfg=VerifyConfig(),
#     )