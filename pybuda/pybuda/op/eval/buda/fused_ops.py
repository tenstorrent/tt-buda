# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from ..common import op_model_to_desc, get_compiler_cached_cycles
from pybuda._C.graph import UBlockOrder
from pybuda._C.backend_api import get_op_model_execution_cycles

DEST_INPUT_OR_OUTPUT_COEFF = 0.5
DEST_INPUT_AND_OUTPUT_COEFF = 0.2

def eval(type, attr, ops):
    raise RuntimeError("Fused op should never be directly evaluated.")

def shape(type, attr, ops, tile_height, tile_width):
    raise RuntimeError("Fused op's shape should never be directly evaluated.")

def execution_cycles(type, arch_name, op_model, sub_op_models) -> int:
    if not sub_op_models:
        raise RuntimeError("execution_cycles has to be called with a list of FusedSubOpModel objects for fused ops.")

    # Fused op execution cycles are calculated as a sum of the execution cycles of all the sub ops, with some speedup for 
    # binary fpu ops using dest on input and/or output since they are not math bound and skipped unpack/pack is noticeable
    total_cycle_count = 0
    for sub_op_model in sub_op_models:
        op_model_desc = op_model_to_desc(type, arch_name, op_model, sub_op_model)
        cycle_coeff = 1

        if op_model_desc.type == "add" or op_model_desc.type == "subtract" or op_model_desc.type == "multiply":
            if sub_op_model.has_dest_input and sub_op_model.has_dest_output:
                cycle_coeff = DEST_INPUT_AND_OUTPUT_COEFF
            elif sub_op_model.has_dest_input or sub_op_model.has_dest_output:
                cycle_coeff = DEST_INPUT_OR_OUTPUT_COEFF

        total_cycle_count += int(cycle_coeff * get_op_model_execution_cycles(op_model_desc))

    return total_cycle_count

def input_ublock_order(type, attr, num_operands):
    if len(attr) == 0:
        return None

    if len(attr) >= 2 and attr[1]: # has_broadcast_c
        # If fused op has broacast C, then u_block_order has to be R due to BE limitations.
        return [UBlockOrder.R] * num_operands

    # If there'a reduce in the fusion, then we need to set appropriate order, on the appropriate input
    # TODO: we don't know what input needs reducing here :( 
    return None

    reduce_dim = attr[0]
    if reduce_dim == 1:
        return None
    if reduce_dim == 2:
        return [UBlockOrder.C] * num_operands
    if reduce_dim == 3:
        return [UBlockOrder.R] * num_operands

    return None


def parallelization(type, attr, op_shape):
    # If there's a reduce, then we have to force the 
    reduce_dim = attr[0] if len(attr) >= 1 else 0
    if reduce_dim == 1:
        return None # TODO: remove once backend supports
        return (op_shape.outputs[0].rt, op_shape.outputs[0].ct)
    if reduce_dim == 2:
        return (1, op_shape.outputs[0].ct)
    if reduce_dim == 3:
        return (op_shape.outputs[0].rt, 1)

    return (op_shape.outputs[0].rt, op_shape.outputs[0].ct)

