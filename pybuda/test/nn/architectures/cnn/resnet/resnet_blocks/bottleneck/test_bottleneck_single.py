# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#

# import pytest

# import torch

# import pybuda
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
# import pybuda.op

# from . import TestBottleneckBlock



# @pytest.mark.fail
# def test_bottleneck_block(
#     bneck_train,
#     bneck_recompute,
#     bneck_in_out_ch,
#     bneck_inbtw_ch,
#     bneck_orig_shape,
#     bneck_stride,
#     bneck_dilation,
#     bneck_depthwise,
#     bneck_bias,
#     # padding_mode,
# ):

#     training = bool(bneck_train)
#     recompute = bool(bneck_recompute)
#     in_out_channels = int(bneck_in_out_ch)
#     inbetween_channels = int(bneck_inbtw_ch)
#     original_shape = eval(bneck_orig_shape)
#     stride = int(bneck_stride)
#     dilation = int(bneck_dilation)
#     depthwise = bool(bneck_depthwise)
#     bias = bool(bneck_bias)

#     model = TestBottleneckBlock(
#         in_channels=in_out_channels,
#         out_channels=in_out_channels,
#         inbetween_channels=inbetween_channels,
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
#             enable_training=training,
#             enable_recompute=recompute
#         ),
#         verify_cfg=VerifyConfig(),
#     )