# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from ..interface import BudaEltwiseNaryOp

import torch
import pybuda
from pybuda._C.backend_api import get_op_model_execution_cycles
from pybuda._C.balancer import FactorizedInt
from pybuda._C import UnsupportedHWOpsError
from ....pybudaglobal import TILE_DIM
from ..common import op_model_to_desc
from pybuda.utils import align_up_tile, round_up_div
from .tm import eval as tm_eval


def slice_tms(tensor, slice_factor_r, slice_factor_c, ublock_order_r=True):
    if ublock_order_r:
        if slice_factor_r > 1:
            tensor = tm_eval("vslice", [slice_factor_r], [tensor])
        if slice_factor_c > 1:
            tensor = tm_eval("hslice", [slice_factor_c], [tensor])
    else:
        if slice_factor_c > 1:
            tensor = tm_eval("hslice", [slice_factor_c], [tensor])
        if slice_factor_r > 1:
            tensor = tm_eval("vslice", [slice_factor_r], [tensor])
    return tensor


def stack_tms(tensor, stack_factor_r, stack_factor_c, ublock_order_r=True):
    if ublock_order_r:
        if stack_factor_c > 1:
            tensor = tm_eval("hstack", [stack_factor_c], [tensor])
        if stack_factor_r > 1:
            tensor = tm_eval("vstack", [stack_factor_r], [tensor])
    else:
        if stack_factor_r > 1:
            tensor = tm_eval("vstack", [stack_factor_r], [tensor])
        if stack_factor_c > 1:
            tensor = tm_eval("hstack", [stack_factor_c], [tensor])
    return tensor


def as_physical_offset(
    logical_offset,
    dim,
    shape,
    ublock_order_r=True,
    ublock_r=1,
    ublock_c=1,
    grid_r=1,
    grid_c=1,
    t_stream_factor_r=1,
    t_stream_factor_c=1,
):
    assert dim >= 1 and dim <= 3
    if dim == 1:
        physical_offset = logical_offset * t_stream_factor_r * t_stream_factor_c
    else:
        logical_offset //= ublock_r if (dim == 2) else ublock_c
        ublock_order_cuts_dim = dim == 3 if ublock_order_r else dim == 2
        num_ublocks_r, num_ublocks_c = (
            shape[-2] // (TILE_DIM * ublock_r * grid_r * t_stream_factor_r),
            shape[-1] // (TILE_DIM * ublock_c * grid_c * t_stream_factor_c),
        )
        num_ublocks_dim = num_ublocks_r if dim == 3 else num_ublocks_c
        physical_offset = (
            logical_offset
            if ublock_order_cuts_dim
            else logical_offset * num_ublocks_dim
        )
    return physical_offset


def snap_to_buda(shape):
    shape = list(shape)
    while len(shape) < 4:
        shape.insert(0, 1)
    assert len(shape) == 4
    shape[-2] = align_up_tile(shape[-2])
    shape[-1] = align_up_tile(shape[-1])
    return shape


class Splice(BudaEltwiseNaryOp):
    supported_splice_types = ["concatenate", "select", "interleave"]

    @classmethod
    def create(
        cls,
        splice_type,
        dim,
        ranges,
        input_shapes,
    ):
        assert dim >= 1 and dim <= 3, f"Illegal dim {dim}"
        assert (
            splice_type in Splice.supported_splice_types
        ), f"Illegal splice_type {splice_type}"
        assert len(ranges) == len(input_shapes)

        self = cls("splice")
        self.splice_type = splice_type
        self.dim = dim
        self.canonical_ranges = ranges
        self.input_shapes = input_shapes

        self.update_ranges()

        if dim == 1:
            self.set_buda_attr("granularity", "t")

        return self

    @classmethod
    def create_concatenate(
        cls,
        dim,
        input_shapes,
    ):
        if dim < 0:
            dim += 4
        if not (dim >= 1 and dim <= 3):
            raise UnsupportedHWOpsError("Splice op can only operate on dims 1, 2, or 3")
        input_shapes = [snap_to_buda(shape) for shape in input_shapes]
        ranges = []
        for input_shape in input_shapes:
            length = input_shape[dim] // (TILE_DIM if dim > 1 else 1)
            ranges.append((0, length, length))

        return cls.create("concatenate", dim, ranges, input_shapes)

    @classmethod
    def create_select(
        cls,
        dim,
        index,
        length,
        stride,
        input_shape,
    ):
        if dim < 0:
            dim += 4
        if not (dim >= 1 and dim <= 3):
            raise UnsupportedHWOpsError("Splice op can only operate on dims 1, 2, or 3")
        tilize = TILE_DIM if dim > 1 else 1
        index, length, stride = index // tilize, length // tilize, stride // tilize
        ranges = [(index, length, stride - index)]
        input_shapes = [snap_to_buda(input_shape)]
        return cls.create("select", dim, ranges, input_shapes)

    @classmethod
    def create_interleave(
        cls,
        dim,
        stride,
        input_shapes,
    ):
        if dim < 0:
            dim += 4
        if not (dim >= 1 and dim <= 3):
            raise UnsupportedHWOpsError("Splice op can only operate on dims 1, 2, or 3")
        input_shapes = [snap_to_buda(shape) for shape in input_shapes]
        ranges = [(0, stride, stride)] * len(input_shapes)
        return cls.create("interleave", dim, ranges, input_shapes)

    def eval(self, tensors):
        tensors = list(tensors)  # create a new list
        expected_shape, _ = self.shape(
            [tensor.shape for tensor in tensors], TILE_DIM, TILE_DIM
        )
        is_splice_mode_ublock = self.dim != 1

        if is_splice_mode_ublock:
            # Slice each tensor down to just 1 ublock face
            for input_index, tensor in enumerate(tensors):
                slice_factor_r = tensor.shape[2] // (
                    TILE_DIM * self.ublock_r * self.grid_r
                )
                slice_factor_c = tensor.shape[3] // (
                    TILE_DIM * self.ublock_c * self.grid_c
                )

                # Slice into grid chunks
                tensor = slice_tms(tensor, self.grid_r, self.grid_c)

                # Slice into ublock chunks
                tensor = slice_tms(
                    tensor, slice_factor_r, slice_factor_c, self.ublock_order_r
                )

                face_shape_tiles = (
                    tensor.shape[2] // TILE_DIM,
                    tensor.shape[3] // TILE_DIM,
                )
                ublock = (self.ublock_r, self.ublock_c)
                assert (
                    face_shape_tiles == ublock
                ), f"We should be left with faces of ublocks {face_shape_tiles} != ublock[{ublock}]"

                tensors[input_index] = tensor

        # Round-robin and pick ublocks
        current_starts = [0] * len(tensors)
        results = []

        splices_done = [False] * len(tensors)
        while not all(splices_done):
            for input_index, tensor in enumerate(tensors):
                index, length, stride = self.ranges[input_index]
                current_start = current_starts[input_index]
                if current_start < tensor.shape[1]:
                    # Take ublock slices
                    segment = tensor[
                        :,
                        (current_start + index) : (current_start + index + length),
                        :,
                        :,
                    ]
                    results.append(segment)
                    current_starts[input_index] = current_start + index + stride
                else:
                    splices_done[input_index] = True

        # Cat all results along z-dim since thats how we spliced!
        #  .... unless splice asked to occur in w-dim = 0 ... Then cat along w-dim
        result = torch.cat(results, dim=1)

        # Re-stack ublocks into the output shape
        if is_splice_mode_ublock:
            stack_factor_r = expected_shape[2] // (
                TILE_DIM * self.ublock_r * self.grid_r
            )
            stack_factor_c = expected_shape[3] // (
                TILE_DIM * self.ublock_c * self.grid_c
            )

            # Stack back the ublock chunks into per-grid chunks
            result = stack_tms(
                result, stack_factor_r, stack_factor_c, self.ublock_order_r
            )

            # Stack back the grid
            result = stack_tms(result, self.grid_r, self.grid_c)

        return result

    def shape(self, tensor_shapes, tile_height, tile_width):
        if self.splice_type == "concatenate":
            output_shape = list(tensor_shapes[0])
            output_shape[self.dim] = 0
            for input_shape in tensor_shapes:
                output_shape[self.dim] += input_shape[self.dim]
            if self.dim > 1:
                output_shape[self.dim] = align_up_tile(output_shape[self.dim])
            return output_shape, []

        elif self.splice_type == "select":
            assert (
                len(tensor_shapes) == 1
            ), f"Select should have only 1 tensor_shape: len(tensor_shapes) = {len(tensor_shapes)}"
            assert len(self.canonical_ranges) == 1
            shape = list(tensor_shapes[0])
            index, length, stride = self.ranges[0] if self.dim == 1 else self.canonical_ranges[0]
            shape[self.dim] = length * round_up_div(shape[self.dim] - index, index + stride)
            if self.dim >= 2:
                shape[self.dim] = align_up_tile(shape[self.dim])
            return tuple(shape), []

        elif self.splice_type == "interleave":
            assert self.dim == 1
            output_shape = list(tensor_shapes[0])
            for tensor_shape in tensor_shapes[1:]:
                output_shape[self.dim] += tensor_shape[self.dim]
            return output_shape, []

        else:
            assert False, f"Unhandled splice type: {self.splice_type}"

    def _get_input_tile_dims(self, op_shape=None):
        input_shapes = op_shape.inputs if op_shape is not None else self.input_shapes
        tile_dim = 1 if op_shape is not None else TILE_DIM
        if self.dim == 1:
            return [shape[-3] for shape in input_shapes]
        return [shape[self.dim] // tile_dim for shape in input_shapes]

    def _get_output_tile_shape(self, op_shape=None):
        if op_shape is not None:
            return op_shape.outputs[0].rt, op_shape.outputs[0].ct
        output_shape = self.shape(self.input_shapes, TILE_DIM, TILE_DIM)[0]
        return (output_shape[2] // TILE_DIM, output_shape[3] // TILE_DIM)

    def parallelization(self, op_shape, fracture_factor):
        output_rt, output_ct = self._get_output_tile_shape(op_shape=op_shape)
        input_tile_dims = self._get_input_tile_dims(op_shape=op_shape)

        def mode_ublock_par(ranges, input_dims, output_dim):
            input_dim = input_dims[0]
            index, length, stride = ranges[0]

            par = 1
            if (
                len(ranges) == 1
                and input_dim % (index + stride) == 0
                and output_dim % (index + stride) == 0
            ):
                par = input_dim // (index + stride)
            return par

        if self.dim == 1:
            return output_rt, output_ct
        elif self.dim == 2:
            par_r = mode_ublock_par(self.canonical_ranges, input_tile_dims, output_rt)
            return (par_r, output_ct)
        elif self.dim == 3:
            par_c = mode_ublock_par(self.canonical_ranges, input_tile_dims, output_ct)
            return (output_rt, par_c)
        assert False

    def input_ublock_order(self, num_operands):
        return None

    def execution_cycles(self, arch_name, op_model) -> int:
        op_model_desc = op_model_to_desc("splice", arch_name, op_model)
        return get_op_model_execution_cycles(op_model_desc)

    def convert_mode_t(self):
        assert (
            self.dim == 2 or self.dim == 3
        ), f"Illegal dim for convert_mode_t: {self.dim}"
        output_dim = self._get_output_tile_shape()[self.dim - 2]
        orig_z = self.shape(self.input_shapes, TILE_DIM, TILE_DIM)[0][1]
        orig_dim = self.dim
        input_dims = self._get_input_tile_dims()
        par = FactorizedInt(output_dim)
        for i, (index, length, stride) in enumerate(self.canonical_ranges):
            par = par & FactorizedInt(input_dims[i])
            if index > 0:
                par = par & FactorizedInt(index)
            par = par & FactorizedInt(length)
            if stride - length > 0:
                par = par & FactorizedInt(stride - length)

        input_slices = []
        canonical_ranges = []
        input_shapes = []
        for i, (index, length, stride) in enumerate(self.canonical_ranges):
            assert input_dims[i] % par.max_factor == 0
            assert index % par.max_factor == 0
            assert length % par.max_factor == 0
            assert stride % par.max_factor == 0
            slice_amount = input_dims[i] // par.max_factor
            input_slices.append(slice_amount)
            input_shape = list(self.input_shapes[i])
            input_shape[self.dim] //= slice_amount
            input_shape[1] *= slice_amount
            input_shapes.append(tuple(input_shape))
            canonical_ranges.append(
                (
                    index // par.max_factor,
                    length // par.max_factor,
                    stride // par.max_factor,
                )
            )

        self.dim = 1
        self.canonical_ranges = canonical_ranges
        self.input_shapes = input_shapes
        self.set_buda_attr("granularity", "t")

        new_z = self.shape(self.input_shapes, TILE_DIM, TILE_DIM)[0][1]
        assert new_z % orig_z == 0
        output_stack = new_z // orig_z

        return orig_dim, input_slices, output_stack

    def update_ranges(
        self,
        ublock_order_r=True,
        ublock_r=1,
        ublock_c=1,
        grid_r=1,
        grid_c=1,
        t_stream_factor_r=1,
        t_stream_factor_c=1,
    ):
        self.ublock_order_r = ublock_order_r
        self.ublock_r = ublock_r
        self.ublock_c = ublock_c
        self.grid_r = grid_r
        self.grid_c = grid_c
        self.t_stream_factor_r = t_stream_factor_r
        self.t_stream_factor_c = t_stream_factor_c

        as_phys = lambda offset, shape: as_physical_offset(
            offset,
            self.dim,
            shape,
            ublock_order_r=ublock_order_r,
            ublock_r=ublock_r,
            ublock_c=ublock_c,
            grid_r=grid_r,
            grid_c=grid_c,
            t_stream_factor_r=t_stream_factor_r,
            t_stream_factor_c=t_stream_factor_c,
        )
        ranges = []
        for rng, shape in zip(self.canonical_ranges, self.input_shapes):
            index, length, stride = rng
            assert length > 0
            ranges.append(
                (as_phys(index, shape), as_phys(length, shape), as_phys(stride, shape))
            )
        self.ranges = ranges

        for i, rng in enumerate(self.ranges):
            self.set_buda_attr(f"input{i}", rng)
