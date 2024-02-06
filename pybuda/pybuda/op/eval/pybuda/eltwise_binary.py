# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from typing import List, Tuple
from pybuda.pybudaglobal import TILE_DIM
from pybuda.tensor import Tensor
import numpy as np
import torch
from .transpose import TransposeTM
from ..buda.exp import Exp as BudaExp
from .reciprocal import Reciprocal
from .log import Log
from ..buda.log import Log as BudaLog
from .nop import Nop
from ..buda.nop import Nop as BudaNop

from ..common import to_torch_operands
from pybuda.utils import align_up_tile
from pybuda.op.eval.common import calculate_tile_size


def eval(type, attr, ops):
    assert len(ops) == 2, "Eltwise binary should have two inputs"
    assert type == "binary_stack" or len(attr) == 0, "Eltwise binary should have no attributes"
    
    t_ops = to_torch_operands(*ops)

    if t_ops[0].dtype != t_ops[1].dtype:
        if t_ops[0].dtype == torch.bool:
            t_ops = (t_ops[0].type(t_ops[1].dtype), t_ops[1])
        else:
            t_ops = (t_ops[0], t_ops[1].type(t_ops[0].dtype))
    
    f = {
        "add": lambda i: torch.add(t_ops[0], t_ops[1]),
        "divide": lambda i: torch.divide(t_ops[0], t_ops[1]),
        "subtract": lambda i: torch.subtract(t_ops[0], t_ops[1]),
        "multiply": lambda i: torch.multiply(t_ops[0], t_ops[1]),
        "maximum": lambda i: torch.maximum(t_ops[0], t_ops[1]),
        "minimum": lambda i: torch.minimum(t_ops[0], t_ops[1]),
        "heaviside": lambda i: torch.heaviside(t_ops[0], t_ops[1]),
        "binary_stack": lambda i: torch.stack((t_ops[0], t_ops[1]), axis=attr[0]).flatten(attr[0]-1, attr[0]),
        "power": lambda i: torch.pow(t_ops[0], t_ops[1]),
        "greater": lambda i: torch.gt(t_ops[0], t_ops[1]).to(t_ops[0].dtype),
        "greater_equal": lambda i: torch.ge(t_ops[0], t_ops[1]).to(t_ops[0].dtype),
        "less": lambda i: torch.lt(t_ops[0], t_ops[1]).to(t_ops[0].dtype),
        "less_equal": lambda i: torch.le(t_ops[0], t_ops[1]).to(t_ops[0].dtype),
        "equal": lambda i: torch.eq(t_ops[0], t_ops[1]).to(t_ops[0].dtype),
        "not_equal": lambda i: torch.ne(t_ops[0], t_ops[1]).to(t_ops[0].dtype),
        "logical_and": lambda i: torch.logical_and(t_ops[0], t_ops[1]).to(t_ops[0].dtype),
    }
    assert type in f, f"{type} not defined in eval map for eltwise binary ops."

    return f[type](t_ops)

# Return shape, and list of dimensions that were broadcast on operands
def shape(type, attr, ops) -> Tuple[Tuple, List]:
    assert len(ops) == 2, "Eltwise binary should have two inputs"

    if type == "binary_stack":
        dim = attr[0]
        assert ops[0] == ops[1]
        output_shape = list(ops[0])
        output_shape[dim] *= 2
        return tuple(output_shape), []

    assert len(attr) == 0, "Eltwise binary should have no attributes"

    broadcast = []
    output_shape = []

    ops[0] = list(ops[0])
    while len(ops[0]) < len(ops[1]):
        ops[0] = [1] + ops[0]

    ops[1] = list(ops[1])
    while len(ops[1]) < len(ops[0]):
        ops[1] = [1] + ops[1]

    for dim in range(len(ops[0])):
        if ops[0][dim] != ops[1][dim]:
            if ops[1][dim] == 1:
                broadcast.append((1, dim - len(ops[1]), ops[0][dim])) # Convert to negative indexing
                output_shape.append(ops[0][dim])
            else:
                assert ops[0][dim] == 1, f"Eltwise binary ops must have the same shape in both inputs, or one operand must be 1 wide to broadcast: {ops[0]} vs {ops[1]}"
                broadcast.append((0, dim - len(ops[0]), ops[1][dim])) # Convert to negative indexing
                output_shape.append(ops[1][dim])
        else:
            output_shape.append(ops[0][dim])

    return tuple(output_shape), broadcast

def lower(type, attr, lc, ops, outputs):
    assert len(ops) == 2, "Eltwise binary should have two inputs"

    if type == "binary_stack":
        dim = attr[0]
        if dim < 0:
            dim += len(lc.shape(ops[0]))

        if dim == len(lc.shape(ops[0])) - 1:
            lc.op("binary_vstack", ops, [])
        elif dim == len(lc.shape(ops[0])) - 2:
            lc.op("binary_hstack", ops, [])

    elif type in ["greater", "less", "greater_equal", "less_equal", "equal", "not_equal"]:

        A = ops[0]
        B = ops[1]
        in_shape = A.shape.as_list()
        amplification = 1e4

        if len(in_shape) > 4:
            raise RuntimeError("Shape size is out of range.")
        if len(in_shape) < 4:
            in_shape = (4 - len(in_shape)) * [1] + in_shape

        in_shape[-1] = ((in_shape[-1] - 1) // TILE_DIM + 1) * TILE_DIM
        in_shape[-2] = ((in_shape[-2] - 1) // TILE_DIM + 1) * TILE_DIM
        
        one = lc.tensor(torch.ones(in_shape))
        amplifier = lc.tensor(torch.zeros(in_shape) + amplification)

        def ge(A, B):
            diff = lc.op("subtract", (A, B))
                    # diff = A - B
            diff = lc.op("multiply", (diff, amplifier))
                    # diff = (A - B) * amplifier
            diff_one = lc.op("add", (diff, one))
                    # diff + 1.0
            res = lc.op(BudaNop.create(relu_en=True, relu_threshold=1.0, relu_mode="min" ), (diff_one, ))
                    # res = ReLU(diff + 1.0, 1.0)
            res = lc.op(BudaNop.create(relu_en=True, relu_threshold=1.0, relu_mode="max"), (res, ))
                    # res = Inv_ReLU(res, 1.0)
            return res

        def le(A, B):
            return ge(B, A)

        def gt(A, B):
            return lc.op("subtract", (one, le(A, B)))

        def lt(A, B):
            return lc.op("subtract", (one, ge(A, B)))

        def ne(A, B):
            return lc.op("add", (gt(A, B), lt(A, B)))

        def eq(A, B):
            return lc.op("subtract", (one, ne(A, B)))

        if type == "greater":
            gt(A, B)
        elif type == "greater_equal":
            ge(A, B)
        elif type == "less":
            lt(A, B)
        elif type == "less_equal":
            le(A, B)
        elif type == "equal":
            eq(A, B)
        else:
            ne(A, B)
    elif type == "power":
        #lc.op("power_binary", ops, attr)  # 'power' backend op is unary
        ln_x = lc.op(BudaLog.create(), [ops[0]])
        y_ln_x = lc.op("multiply", (ops[1], ln_x)) 
        approximate_mode = "true" if "PYBUDA_EXP_APPROX" in os.environ else "false"
        lc.op(BudaExp.create(approximate_mode=approximate_mode), [y_ln_x])            
    else:
        # Find proper tile sizes
        if bool(int(os.environ.get("PYBUDA_ENABLE_TINY_TILE", "0"))):
            node_shape = lc.pybuda_shape()
            tile_height = calculate_tile_size(node_shape[-2])
            tile_width = calculate_tile_size(node_shape[-1])
        else:
            tile_height, tile_width = TILE_DIM, TILE_DIM

        ops0_dims = len(ops[0].shape)
        ops1_dims = len(ops[1].shape)
        if ops0_dims == 5 and ops1_dims < 5:
            while ops1_dims < 5:
                ops[1] = lc.op(BudaNop.create(unsqueeze = "unsqueeze", unsqueeze_dim=ops1_dims), [ops[1]], tag="dont_remove")
                ops1_dims += 1
        elif ops1_dims == 5 and ops0_dims < 5:
            while ops0_dims < 5:
                ops[0] = lc.op(BudaNop.create(unsqueeze = "unsqueeze", unsqueeze_dim=ops0_dims), [ops[0]], tag="dont_remove")
                ops0_dims += 1
        lc.op(type, ops, attr, {}, "", tile_height, TILE_DIM) # straight 1-1 for all other binaries

    assert type != "take", "Take should be constevaled"

def backward(op_type, attr, ac, operand, inputs, output, grad):

    assert len(inputs) == 2, "Eltwise binary should have two inputs"
    assert len(attr) == 0, "Eltwise binary should have no attributes"
    assert operand < 2, "Invalid operand index"

    # Some operands are implicitly broadcasted, so their shapes in backward() need to be unbroadcasted for grad accumulation

    shapes = [
        inputs[0].shape.as_list(),
        inputs[1].shape.as_list()
    ]

    # Pad to longer dims
    longer_dims = max(len(s) for s in shapes)
    shapes = [[1] * (longer_dims - len(s)) + s for s in shapes]
    # Pad gradient shape to longer dims
    grad_shape = [1] * (longer_dims - len(grad.shape.as_list())) + grad.shape.as_list()
    grad_shape_len = len(grad_shape)

    if op_type == "add":
        if inputs[operand].shape != grad.shape:
            for i in range(len(shapes[operand])):
                if shapes[operand][i] < grad_shape[i]:
                    # Negative indexing for reduce axis
                    grad = ac.op("reduce_sum", (grad,), (i - grad_shape_len,))
        return ac.op(Nop.create(), (grad,))  # pass gradient through

    elif op_type == "subtract":
        if inputs[operand].shape != grad.shape:
            for i in range(len(shapes[operand])):
                if shapes[operand][i] < grad.shape[i]:
                    grad = ac.op("reduce_sum", (grad,), (i,))
        if operand == 0:
            return ac.op(Nop.create(), (grad,))
        else:
            return ac.op("multiply", (grad, ac.constant(-1)))

    elif op_type == "multiply":
        op_grad = ac.op("multiply", (grad, inputs[1-operand]))
        if inputs[operand].shape != grad.shape:
            for i in range(len(shapes[operand])):
                if shapes[operand][i] < grad_shape[i]:
                    op_grad = ac.op("reduce_sum", (op_grad,), (i - grad_shape_len,))
        return op_grad

    elif op_type == "maximum":
        # TODO
        return ac.op(Nop.create(), (grad,)) # pass gradient through

    elif op_type == "power": 
        if operand == 0: # dx = y * (x^y) * recp(x)
            recip = ac.op(Reciprocal.create(), (inputs[0],))
            partial_grad = ac.op("multiply", (output, recip))  
            pow_grad = ac.op("multiply", (inputs[1], partial_grad))
        if operand == 1: # dy = (x^y) * ln(x)
            ln_x = ac.op(Log.create(), [inputs[0]])
            pow_grad = ac.op("multiply", (output, ln_x)) 
        return ac.op("multiply", (pow_grad, grad))

    assert False, f"{op_type} not defined in eltwise binary backward."


def decompose(op_type, attr, dc, inputs):
    if op_type == "binary_stack":
        operand0 = inputs[0]
        operand1 = inputs[1]

        axis = attr[0]
        # import pdb; pdb.set_trace()
        if operand0.shape[axis] % TILE_DIM != 0 or operand1.shape[axis] % TILE_DIM != 0:
            # Currently only support TILE aligned binary stack
            return
        assert operand0.shape == operand1.shape, "Inputs to BinaryStack must have the same shape"

        total_size = operand0.shape[axis] + operand1.shape[axis]
        if axis == -1:
            # Operand 0
            vstack0 = dc.op("vstack", [operand0], (operand0.shape[-3],))
            transpose0 = dc.op(TransposeTM.create(-2, -1), [vstack0])
            # Picker matmul to expand and interleave input size
            cols = torch.arange(operand0.shape[axis])
            rows = cols * 2
            operand0_picker = torch.sparse_coo_tensor(
                [rows.tolist(), cols.tolist()],
                torch.ones(cols.shape[0]),
                (total_size, operand0.shape[axis]),
                dtype=torch.float32,
            )
            lhs0 = dc.tensor(operand0_picker)
            operand0_mm = dc.op("sparse_matmul", [lhs0, transpose0])
            # Convert shape back
            transpose_back0 = dc.op(TransposeTM.create(-2, -1), [operand0_mm])
            vslice0 = dc.op("vslice", [transpose_back0], (operand0.shape[-3],))


            # Operand 1
            vstack1 = dc.op("vstack", [operand1], (operand1.shape[-3],))
            transpose1 = dc.op(TransposeTM.create(-2, -1), [vstack1])
            # Picker matmul to expand and interleave input size
            cols = torch.arange(operand1.shape[axis])
            rows = cols * 2 + 1
            operand1_picker = torch.sparse_coo_tensor(
                [rows.tolist(), cols.tolist()],
                torch.ones(cols.shape[0]),
                (total_size, operand1.shape[axis]),
                dtype=torch.float32,
            )
            lhs1 = dc.tensor(operand1_picker)
            operand1_mm = dc.op("sparse_matmul", [lhs1, transpose1])
            # Convert shape back
            transpose_back1 = dc.op(TransposeTM.create(-2, -1), [operand1_mm])
            vslice1 = dc.op("vslice", [transpose_back1], (operand1.shape[-3],))

            # Add 2 sides together
            result = dc.op("add", [vslice0, vslice1],)
            dc.fuse(result)
            return
        elif axis == -2:
            # Operand 0
            hstack0 = dc.op("hstack", [operand0], (operand0.shape[-3],))
            # Picker matmul to expand and interleave input size
            cols = torch.arange(operand0.shape[axis])
            rows = cols * 2
            operand0_picker = torch.sparse_coo_tensor(
                [rows.tolist(), cols.tolist()],
                torch.ones(cols.shape[0]),
                (total_size, operand0.shape[axis]),
                dtype=torch.float32,
            )
            lhs0 = dc.tensor(operand0_picker)
            operand0_mm = dc.op("sparse_matmul", [lhs0, hstack0])
            # Convert shape back
            hslice0 = dc.op("hslice", [operand0_mm], (operand0.shape[-3],))


            # Operand 1
            hstack1 = dc.op("hstack", [operand1], (operand1.shape[-3]))
            # Picker matmul to expand and interleave input size
            cols = torch.arange(operand1.shape[axis])
            rows = cols * 2 + 1
            operand1_picker = torch.sparse_coo_tensor(
                [rows.tolist(), cols.tolist()],
                torch.ones(cols.shape[0]),
                (total_size, operand1.shape[axis]),
                dtype=torch.float32,
            )
            lhs1 = dc.tensor(operand1_picker)
            operand1_mm = dc.op("sparse_matmul", [lhs1, hstack1])
            # Convert shape back
            hslice1 = dc.op("hslice", [operand1_mm], (operand1.shape[-3],))

            # Add 2 sides together
            result = dc.op("add", [hslice0, hslice1],)
            dc.fuse(result)
            return
        else:
            raise RuntimeError(f"Found BinaryStack op with axis {axis}")

    elif op_type == "divide":
        recip = dc.op(Reciprocal.create(), [inputs[1]])
        result = dc.op("multiply", [inputs[0], recip])
        dc.fuse(result)
        return
    # Can be used if backend don't implement maximum op in the future. 
    #
    # assert len(inputs) == 2, "Eltwise binary should have two inputs"
    # if op_type == "maximum":
    #     x = inputs[0]
    #     y = inputs[1]

    #     a_ge = dc.op("greater_equal", (x, y))
    #     b_lt = dc.op("less", (x, y))
    #     a_ge_val = dc.op("multiply", (x, a_ge))
    #     b_lt_val = dc.op("multiply", (y, b_lt))
    #     res = dc.op("add", (a_ge_val, b_lt_val))

    #     dc.fuse(res)
    #     return

    ops0_dims = len(inputs[0].shape)
    ops1_dims = len(inputs[1].shape)
    if ops0_dims > ops1_dims and ops0_dims == 5:
        ops1 = dc.op("reshape", [inputs[1]], list(inputs[0].shape))
        result = dc.op(op_type, [inputs[0], ops1])
        dc.fuse(result)
    elif ops1_dims > ops0_dims and ops1_dims == 5:
        ops0 = dc.op("reshape", [inputs[0]], list(inputs[1].shape))
        result = dc.op(op_type, [ops0, inputs[1]])
        dc.fuse(result)

def decompose_post_autograd(op_type, attr, dc, inputs):
    assert len(inputs) == 2, "Eltwise binary should have two inputs"
    if op_type == "heaviside":
        x = inputs[0]
        y = inputs[1]
        shape = x.shape.as_list()
        zero = dc.tensor(torch.zeros(shape))
        x_gt = dc.op("greater", (x, zero))
        x_eq = dc.op("equal", (x, zero))
        res = dc.op("multiply", (x_eq, y))
        res = dc.op("add", (res, x_gt))
        dc.fuse(res)
        return
    elif op_type == "maximum":
        operand0, operand1 = inputs[0], inputs[1]
        orig_op0_shape = operand0.shape.as_list()
        orig_op1_shape = operand1.shape.as_list()
        vslice_op0, vslice_op1 = False, False
        slice_factor = None
        
        if len(orig_op1_shape) > 2 and orig_op1_shape[-3] != 1:
            slice_factor = orig_op1_shape[-3]
            vslice_op1 = True

        if len(orig_op0_shape) > 2 and orig_op0_shape[-3] != 1:
            slice_factor = orig_op0_shape[-3]
            vslice_op0 = True

        if vslice_op0 and vslice_op1:
            assert orig_op0_shape[-3] == orig_op1_shape[-3]

        op0_shape = operand0.shape.as_list()
        op1_shape = operand1.shape.as_list()

        max_operand_nd = max(len(op0_shape), len(op1_shape), 3)
        while len(operand0.shape) < max_operand_nd:
            operand0 = dc.op("unsqueeze", [operand0], (0, len(operand0.shape)))
        while len(operand1.shape) < max_operand_nd:
            operand1 = dc.op("unsqueeze", [operand1], (0, len(operand1.shape)))

        if (slice_factor != None):
            concat_z = dc.op("interleave", [operand0, operand1], (-3, 1))
            result = dc.op("reduce_max", [concat_z], (-3, 2))
        else:
            concat_z = dc.op("concatenate", [operand0, operand1], (-3,))
            result = dc.op("reduce_max", [concat_z], (-3,))
        
        while len(result.shape) > max_operand_nd:
            result = dc.op("squeeze", [result], (0,))

        dc.fuse(result)
        return
    else:
        ops0_dims = len(inputs[0].shape)
        ops1_dims = len(inputs[1].shape)
        if ops0_dims > ops1_dims and ops0_dims == 5:
            ops1 = dc.op("reshape", [inputs[1]], list(inputs[0].shape))
            result = dc.op(op_type, [inputs[0], ops1])
            dc.fuse(result)
        elif ops1_dims > ops0_dims and ops1_dims == 5:
            ops0 = dc.op("reshape", [inputs[0]], list(inputs[1].shape))
            result = dc.op(op_type, [ops0, inputs[1]])
            dc.fuse(result)

def decompose_post_optimize(op_type, attr, dc, inputs):
    operand0, operand1 = inputs[0], inputs[1]
    orig_op0_shape = operand0.shape.as_list()
    orig_op1_shape = operand1.shape.as_list()
    if op_type == "minimum":
        negative_one = torch.ones([1,]) * -1
        negative_one_tensor = dc.tensor(negative_one)

        neg_op0 = dc.op("multiply", [operand0, negative_one_tensor])
        neg_op1 = dc.op("multiply", [operand1, negative_one_tensor])

        binary_max = dc.op("maximum", [neg_op0, neg_op1])

        result = dc.op("multiply", [binary_max, negative_one_tensor])
        dc.fuse(result)
        return
    
    if op_type == "binary_stack":
        axis = attr[0]
        
        operand0 = dc.op("pad_tile", [operand0], (-2, orig_op0_shape[-2]))
        operand0 = dc.op("pad_tile", [operand0], (-1, orig_op0_shape[-1]))
        padded_op0_shape = operand0.shape
        operand1 = dc.op("pad_tile", [operand1], (-2, orig_op1_shape[-2]))
        operand1 = dc.op("pad_tile", [operand1], (-1, orig_op1_shape[-1]))
        padded_op1_shape = operand1.shape
        
        total_size = operand0.shape[axis] + operand1.shape[axis]
        # import pdb; pdb.set_trace()
        if axis == -1:
            # Operand 0
            vstack0 = dc.op("vstack", [operand0], (operand0.shape[-3],))
            transpose0 = dc.op(TransposeTM.create(-2, -1), [vstack0])
            # Picker matmul to expand and interleave input size
            cols = torch.arange(operand0.shape[axis])
            rows = cols * 2
            operand0_picker = torch.sparse_coo_tensor(
                [rows.tolist(), cols.tolist()],
                torch.ones(cols.shape[0]),
                (total_size, operand0.shape[axis]),
                dtype=torch.float32,
            )
            lhs0 = dc.tensor(operand0_picker)
            operand0_mm = dc.op("sparse_matmul", [lhs0, transpose0])
            # Convert shape back
            transpose_back0 = dc.op(TransposeTM.create(-2, -1), [operand0_mm])
            vslice0 = dc.op("vslice", [transpose_back0], (operand0.shape[-3],))


            # Operand 1
            vstack1 = dc.op("vstack", [operand1], (operand1.shape[-3],))
            transpose1 = dc.op(TransposeTM.create(-2, -1), [vstack1])
            # Picker matmul to expand and interleave input size
            cols = torch.arange(operand1.shape[axis])
            rows = cols * 2 + 1
            operand1_picker = torch.sparse_coo_tensor(
                [rows.tolist(), cols.tolist()],
                torch.ones(cols.shape[0]),
                (total_size, operand1.shape[axis]),
                dtype=torch.float32,
            )
            lhs1 = dc.tensor(operand1_picker)
            operand1_mm = dc.op("sparse_matmul", [lhs1, transpose1])
            # Convert shape back
            transpose_back1 = dc.op(TransposeTM.create(-2, -1), [operand1_mm])
            vslice1 = dc.op("vslice", [transpose_back1], (operand1.shape[-3],))

            # Add 2 sides together
            result = dc.op("add", [vslice0, vslice1],)
            
            # Narrow back down to original size
            if result.shape[-1] - (orig_op0_shape[-1] + orig_op1_shape[-1]) >= TILE_DIM:
                result = dc.op("vstack", [result], (orig_op0_shape[-3],))
                result = dc.op(TransposeTM.create(-2, -1), [result])
                
                cols = torch.arange(orig_op0_shape[-1] + orig_op1_shape[-1])
                rows = cols
                
                size = align_up_tile(orig_op0_shape[-1] + orig_op1_shape[-1])
                
                picker = torch.sparse_coo_tensor(
                    [rows.tolist(), cols.tolist()],
                    torch.ones(cols.shape[0]),
                    (size, result.shape[-2]),
                    dtype=torch.float32,
                )
                # import pdb; pdb.set_trace()
                lhs = dc.tensor(picker)
                result = dc.op("sparse_matmul", [lhs, result])
                
                result = dc.op(TransposeTM.create(-2, -1), [result])
                result = dc.op("vslice", [result], (orig_op0_shape[-3],))
                
            result = dc.op("narrow", [result], (-1, 0, orig_op1_shape[-1] + orig_op0_shape[-1], result.shape[-1]))
            result = dc.op("narrow", [result], (-2, 0, orig_op1_shape[-2], result.shape[-2]))
            
            dc.fuse(result)
            return
        else:
            raise RuntimeError(f"Found BinaryStack op with axis {axis}")


def initial_flops_estimate(type, attr, ops):
    flops = 0
    output_shape = shape(type, attr, ops)[0]
    if type in ["add", "subtract", "power", "maximum", "minumum", "multiply"]:
        flops = np.prod(output_shape)
    
    return flops
    
