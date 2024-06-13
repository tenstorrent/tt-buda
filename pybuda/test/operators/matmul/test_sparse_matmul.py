
# From sanity

import pytest

from pybuda.config import _get_global_compiler_config
from pybuda import Tensor
import torch

from pybuda import pybuda

from pybuda.verify.config import TestKind, VerifyConfig

from test.common import run

from pybuda.module import PyBudaModule




# Tests from sanity moved here:

@pytest.mark.parametrize("config", ["3x3conv", "data_mismatch", "c_stream", "in_out_stream"])
def test_sparse_matmul(test_device, config):
    from pybuda.op.eval.sparse_utils import create_conv2d_sparse_picker_matrix

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "MaximizeTMinimizeGrid"

    if config == "3x3conv":
        iH, iW = (64, 64)
        inC = 32
        kH, kW = (3, 3)
        stride = (2, 2)
        padding = (kW // 2, kW // 2, kH // 2, kH // 2)
        dilation = 1

        t = torch.arange(iH*iW*inC, dtype=torch.float32).reshape((1, 1, iH * iW, inC))
        act = Tensor.create_from_torch(t)

        pickers = []
        for kY in range(kH):
            for kX in range(kW):
                y_shift = ((kH - 1) // 2) - kY
                x_shift = ((kW - 1) // 2) - kX
                picker = create_conv2d_sparse_picker_matrix(iH, iW, y_shift, x_shift, kH, kW, stride, padding, dilation, tile_align=True)
                pickers.append(picker)
        sparse = Tensor.create_from_torch(torch.stack(pickers).unsqueeze(0), constant=True)
    elif config == "data_mismatch":
        minimal_tiles = 2
        act = torch.randn(32*minimal_tiles,32).unsqueeze(0).unsqueeze(0)
        out_tiles = minimal_tiles // 2
        eye = torch.eye(32*minimal_tiles, 32*minimal_tiles)
        pickers = [
            eye[:(out_tiles*32), :].to_sparse(),
            eye[(out_tiles*32-16):-16, :].to_sparse(),
        ]
        sparse = Tensor.create_from_torch(torch.stack(pickers).unsqueeze(0), constant=True)
    elif config == "c_stream":

        pytest.skip() # tenstorrent/budabackend#1543
        pybuda.config.override_t_stream_dir("sparse0.lc2", "C")
        pybuda.config.override_t_stream_shape("sparse0.lc2", (1, 32))
        iH, iW = (64, 64)
        inC = 1024
        kH, kW = (1, 1)
        stride = (2, 2)
        padding = (kW // 2, kW // 2, kH // 2, kH // 2)
        dilation = 1

        t = torch.arange(iH*iW*inC, dtype=torch.float32).reshape((1, 1, iH * iW, inC))
        act = Tensor.create_from_torch(t)

        pickers = []
        for kY in range(kH):
            for kX in range(kW):
                y_shift = ((kH - 1) // 2) - kY
                x_shift = ((kW - 1) // 2) - kX
                picker = create_conv2d_sparse_picker_matrix(iH, iW, y_shift, x_shift, kH, kW, stride, padding, dilation, tile_align=True)
                pickers.append(picker)
        sparse = Tensor.create_from_torch(torch.stack(pickers).unsqueeze(0), constant=True)
    elif config == "in_out_stream":
        pybuda.config.override_t_stream_dir("buf0", "R")
        pybuda.config.override_t_stream_shape("buf0", (2, 1))
        pybuda.config.override_t_stream_dir("sparse0.lc2", "R")
        pybuda.config.override_t_stream_shape("sparse0.lc2", (3, 1))

        iH, iW = (32, 32)
        inC = 32
        kH, kW = (3, 3)
        stride = (2, 2)
        padding = (kW // 2, kW // 2, kH // 2, kH // 2)
        dilation = 1

        t = torch.arange(iH*iW*inC, dtype=torch.float32).reshape((1, 1, iH * iW, inC))
        act = Tensor.create_from_torch(t)

        pickers = []
        for kY in range(kH):
            for kX in range(kW):
                y_shift = ((kH - 1) // 2) - kY
                x_shift = ((kW - 1) // 2) - kX
                picker = create_conv2d_sparse_picker_matrix(iH, iW, y_shift, x_shift, kH, kW, stride, padding, dilation, tile_align=True)
                pickers.append(picker)
        sparse = Tensor.create_from_torch(torch.stack(pickers).unsqueeze(0), constant=True)
    else:
        raise RuntimeError("Unknown config")

    @run(
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch),
    )
    def simple_sparse_matmul(act, sparse=None):
        if config == "in_out_stream":
            act = pybuda.op.Buffer("buf0", act)
        return pybuda.op.SparseMatmul("sparse0", sparse, act)

    simple_sparse_matmul(act, sparse=sparse)

    
def test_z_sparse_matmul(test_device):
    input_shape = (1, 64, 128, 128)

    class Model(PyBudaModule):
        def __init__(self):
            super().__init__(name="sparsematmul_test")
            rows = torch.arange(0, 128).tolist()
            cols = rows
            sparse = torch.sparse_coo_tensor([rows, cols],torch.ones(len(cols)), (128, 128), dtype=torch.float32)
            sparse = torch.stack([sparse]*64, -3)
            sparse = torch.unsqueeze(sparse, 0) 
            self.add_constant("sparse")
            self.set_constant("sparse", pybuda.Tensor.create_from_torch(sparse, constant=True))

        def forward(self, x):
            out = pybuda.op.SparseMatmul("", self.get_constant("sparse"), x)
            return out

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "MaximizeTMinimizeGrid"

    pybuda.verify.verify_module(
        Model(),
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=TestKind.INFERENCE,
        ),
    )