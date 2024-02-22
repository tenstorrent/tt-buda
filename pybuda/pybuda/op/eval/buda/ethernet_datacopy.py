import os

from ..interface import BudaEltwiseUnaryOp

import torch
import pybuda
from pybuda._C.backend_api import get_op_model_execution_cycles
from pybuda._C.balancer import FactorizedInt
from pybuda._C import UnsupportedHWOpsError
from ..common import op_model_to_desc
from pybuda.utils import align_up_tile, round_up_div
from .tm import eval as tm_eval
from ..common import to_torch_operands, op_model_to_desc, get_compiler_cached_cycles
from pybuda.tensor import pad_pytorch_tensor_to_buda
from pybuda._C.backend_api import get_op_model_execution_cycles
from pybuda._C.backend_api import get_op_model_param
from pybuda.pybudaglobal import TILE_DIM
from pybuda._C.graph import UBlockOrder, Shape


class EthernetDatacopy(BudaEltwiseUnaryOp):
    @classmethod
    def create(cls):
        self = cls("ethernet_datacopy")
        return self

    def eval(self, tensors):
        assert len(tensors) == 1, "Ethernet Datacopy should have one input"
        shape = tensors[0].shape
        original_types = [o.dtype for o in tensors]
        ret = tensors[0]
        if ret.dtype != original_types[0]:
            ret = ret.type(original_types[0])

        return ret

    def shape(self, tensor_shapes, tile_height, tile_width):
        shape = tensor_shapes[0]
        assert len(tensor_shapes) == 1, "Ethernet Datacopy should have one input"
        if tile_height == TILE_DIM:
            shape[-2] = align_up_tile(shape[-2])
        elif tile_height < TILE_DIM:
            shape[-2] = tile_height
        else:
            raise RuntimeError(
                f"Tile height {tile_height} is larger than max allowed TILE_DIM {TILE_DIM}"
            )

        return shape, []

    def parallelization(self, op_shape, fracture_factor):
        return (op_shape.outputs[0].rt, op_shape.outputs[0].ct)

    def input_ublock_order(self, num_operands):
        return None

    def execution_cycles(self, arch_name, op_model) -> int:
        op_model_desc = op_model_to_desc("ethernet_datacopy", arch_name, op_model)

        compiler_cache_cycles = get_compiler_cached_cycles(op_model_desc)
        if compiler_cache_cycles is not None:
            return compiler_cache_cycles

        tile_weight = get_op_model_param(op_model_desc, "tile_weight")
        output_shape = op_model.op_shape.outputs[0]
        num_tiles = (output_shape.z * output_shape.rt * output_shape.ct) / (
            op_model.grid_shape.r * op_model.grid_shape.c
        )
        cycle_count = tile_weight * num_tiles
        return min(int(cycle_count), 1 << 30)
