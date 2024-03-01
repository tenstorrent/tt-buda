# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

import torch
import torch.nn.functional
from loguru import logger
from ..common import to_torch_operands
from ....pybudaglobal import TILE_DIM
from ....tensor import buda_dataformat_to_pytorch_dtype
import numpy as np
from pybuda.op.eval.common import calculate_tile_size
from .tanh import Tanh
from ..buda.log import Log as BudaLog
from .nop import Nop
from ..buda.nop import Nop as BudaNop
from .buffer import Buffer

from ..buda.exp import Exp as BudaExp
from .exp import Exp
from .reciprocal import Reciprocal

M_2_SQRTPI  = 1.12837916709551257390	# 2/sqrt(pi) 
M_SQRT2     = 1.41421356237309504880	# sqrt(2) 
M_SQRT1_2   = 0.7071067811865476

# Reference implementation is at pytorch/aten/src/ATen/native/cpu/Activation.cpp
# https://github.com/pytorch/pytorch/blob/4f8b986e28736b59bc46cd0873a0f36fdaa6f5b8/aten/src/ATen/native/cpu/Activation.cpp
def gelu_derivative(x, approximate):
    if approximate == "none":
        cdf = 0.5 * (1 + torch.erf(x * M_SQRT1_2))
        pdf = 0.5 * M_SQRT1_2 * M_2_SQRTPI * torch.exp(x * x * -0.5)
        return cdf + x * pdf
    elif approximate == "tanh":
        intermediate_0 = 0.5 * (1 + torch.tanh((M_2_SQRTPI / M_SQRT2) * (x + 0.044715 * torch.pow(x, 3))))
        intermediate_1 = x * torch.exp(-0.5 * x * x) * (0.5 * M_2_SQRTPI / M_SQRT2)
        return intermediate_0 + intermediate_1
    else:
        raise RuntimeError(f"Gelu does not support {approximate} approximation mode.")

def gelu_forward(x, approximate):
    if approximate == "none":
        return torch.nn.functional.gelu(x)
    elif approximate == "tanh":
        import math
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    else:
         raise RuntimeError(f"Gelu does not support {approximate} approximation mode.")


def tile_broadcast(attr, i):
    dim, size = attr
    while len(i.shape) <= ((-dim - 1) if dim < 0 else dim):
        i = i.unsqueeze(0)
    shape = list(i.shape)
    shape[dim] = size
    return torch.broadcast_to(i, shape)


def eval(type, attr, ops):
    assert len(ops) == 1, "Eltwise unary should have one input"
    assert (
            len(attr) == 0 or
            (type == "clip" and len(attr) == 2) or
            (type == "argmax" and len(attr) == 1) or
            (type == "leaky_relu" and len(attr) == 1) or
            (type == "relu" and len(attr) <= 2) or
            (type == "cumsum" and len(attr) == 2) or
            (type == "dropout" and len(attr) == 3) or
            (type == "tile_broadcast" and len(attr) == 2) or
            (type == "gelu" and len(attr) == 1) or
            (type == "gelu_derivative" and len(attr) == 1) or 
            (type == "pow" and len(attr) == 1)
        ), "Eltwise unary should have no attributes, execpt for clip, leaky_relu, and cumsum"
    
    t_ops = to_torch_operands(*ops)

    # Some ops don't support non-fp32 in pytorch
    original_types = [o.dtype for o in t_ops]
    if original_types[0] != torch.float32:
        if type in ["gelu", "gelu_derivative"]:
            t_ops = tuple(t.type(torch.float32) for t in t_ops)

    if type == "dropout":
        p, training, seed = attr
        rng_state = torch.get_rng_state()
        torch.manual_seed(seed)
        ret = torch.nn.functional.dropout(t_ops[0], p=p, training=bool(training))
        torch.set_rng_state(rng_state)
        return ret

    if type == "relu":

        def relu(x, threshold):
            return x * (x >= threshold).to(x.dtype)

        def inv_relu(x, threshold):
            ir = threshold * (x >= threshold).to(x.dtype) + x * (~(x >= threshold)).to(x.dtype)
            return relu(ir, 0.0)

        x = t_ops[0]
        if len(attr) > 0:
            threshold = attr[0]
        else:
            threshold = 0.0
        if len(attr) > 1:
            mode = attr[1]
        else:
            mode = "min"

        if mode == "min":
            return relu(x, threshold)
        else:
            return inv_relu(x, threshold)

    # relu_threshold = attr[0] if len(attr) > 0 else 0.0
    f = {
        "exp": lambda i: torch.exp(i[0]),
        "sqrt": lambda i: torch.sqrt(i[0]),
        # "relu": lambda i: i[0] * (i[0] >= relu_threshold).to(i[0].dtype),
        "leaky_relu": lambda i: torch.nn.functional.leaky_relu(i[0], attr[0]),
        "gelu": lambda i : gelu_forward(i[0], approximate=attr[0]),
        "gelu_derivative": lambda i : gelu_derivative(i[0], approximate=attr[0]),
        "nop": lambda i: i[0],
        "tilizer": lambda i: i[0],
        "ethernet_datacopy": lambda i: i[0],
        "buffer": lambda i: i[0],
        "reciprocal": lambda i: torch.reciprocal(i[0] + 1e-10), # add epsilon to avoid infinity
        "log": lambda i: torch.log(i[0] + 1e-10), # add epsilon to avoid nan
        "sigmoid": lambda i: torch.sigmoid(i[0]),
        "clip": lambda i: torch.clip(i[0], min=attr[0], max=attr[1]),
        "abs": lambda i: torch.abs(i[0]),
        "cosine": lambda i: torch.cos(i[0]),
        "sine": lambda i: torch.sin(i[0]),
        "tile_broadcast": lambda i: tile_broadcast(attr, i[0]),
        "argmax": lambda i: torch.argmax(i[0], dim=attr[0] if len(attr) > 0 else None, keepdims=True),
        "tanh": lambda i: torch.tanh(i[0]),
        "cumsum": lambda i: torch.cumsum(i[0], dim=attr[0]),
        "logical_not": lambda i: torch.logical_not(i[0]),
        "pow": lambda i: torch.pow(i[0], attr[0])
    }

    assert type in f, f"{type} not defined in eval map for eltwise unary ops."

    ret = f[type](t_ops)
    if ret.dtype != original_types[0]:
        ret = ret.type(original_types[0])

    return ret

def shape(type, attr, ops):
    assert len(ops) == 1, "Eltwise unary should have one input"
    assert (
            len(attr) == 0 or
            (type == "ethernet_datacopy" and (len(attr) == 1 or len(attr) == 2)) or
            (type == "clip" and len(attr) == 2) or
            (type == "argmax" and len(attr) == 1) or
            (type == "leaky_relu" and len(attr) == 1) or
            (type == "relu" and len(attr) <= 2) or
            (type == "cumsum" and len(attr) == 2) or
            (type == "dropout" and len(attr) == 3) or
            (type == "tile_broadcast" and len(attr) == 2) or
            (type == "gelu" and len(attr) == 1) or
            (type == "gelu_derivative" and len(attr) == 1) or
            (type == "pow" and len(attr) == 1)
        ), "Eltwise unary should have no attributes, execpt for clip, leaky_relu and cumsum"

    if type == "argmax":
        dim = attr[0] if len(attr) > 0 else None
        if dim is not None:
            shape = list(ops[0])
            shape[dim] = 1
        else:
            shape = [1] * len(ops[0])
        return tuple(shape), []

    if type == "tile_broadcast":
        assert len(attr) == 2, "Tile broadcast should have two attributes - dim and size"
        dim = attr[0]
        size = attr[1]
        shape = len(ops[0].shape)
        shape[dim] = size
        return shape, []

    return ops[0], []

def lower(type, attr, lc, ops, outputs):
    assert len(ops) == 1 or type == "tile_broadcast", "Eltwise unary should one input"
    
    if type == "relu":
        threshold = 0.0
        mode = "min"
        if len(attr) > 0:
            f32_epsilon = 1.19209289551e-07
            threshold = attr[0] - f32_epsilon
        if len(attr) > 1:
            mode = attr[1]
        lc.op(BudaNop.create(relu_en=True, relu_threshold=threshold, relu_mode=mode), ops)
        
    elif type == "leaky_relu":
        lc.op("lrelu", ops, attr, {"slope": attr[0]})

    elif type == "tile_broadcast":
        assert len(attr) == 2, "Tile broadcast should have two attributes - dim and size"
        dim = attr[0]
        size = attr[1]
        shape_size = len(ops[0].shape)
        output_shape = outputs[0].shape
        output_dim = output_shape[dim]

        if dim < 0:
            dim += 4
        else:
            dim += 4 - len(ops[0].shape)

        assert dim in [2, 3], f"Tile broadcast is only valid on the last two dims (R/C): {shape_size}, {dim}"
        assert size <= TILE_DIM and size > 1, f"Tile broadcast can only broadcast within one tile"

        if bool(int(os.environ.get("PYBUDA_ENABLE_TINY_TILE", "0"))) and dim == 2:
            node_shape = lc.pybuda_shape()
            tile_height = calculate_tile_size(node_shape[-2])
            if node_shape[-2] % tile_height == 0:
                lc.op(BudaNop.create(), ops, tile_height=tile_height,tile_width=TILE_DIM)
                return # Don't need to tile bcast to full tile


        broadcast_dim = TILE_DIM
        if output_dim % TILE_DIM != 0:
            broadcast_dim = ((output_dim // TILE_DIM) + 1) * TILE_DIM

        # use matmul to perform broadcast
        
        if dim == 2:
            assert len(ops[0].shape) == 1 or ops[0].shape[-2] == 1, "Tile broadcast must be on dim that is 1"
            if shape_size == 4:
                # Preseve W dim
                const_shape = (ops[0].shape[0], 1, broadcast_dim, TILE_DIM)
            else:
                const_shape = (1, 1, broadcast_dim, TILE_DIM)
            
            tensor = torch.zeros(const_shape, dtype=buda_dataformat_to_pytorch_dtype(ops[0].output_df))
            tensor[:, :, 0:output_dim, 0] = 1.0 # row broadcast
            const = lc.tensor(tensor)
            if output_dim % TILE_DIM != 0:
                # this isn't just a tile broadcast any more, we're broadcasting to a special dim
                lc.op("matmul", (const, ops[0]))
            else:
                lc.op("matmul", (const, ops[0]), tag="tile_broadcast_r")
        else: 
            assert ops[0].shape[-1] == 1, "Tile broadcast must be on dim that is 1"
            if shape_size == 4:
                # Preseve W dim
                const_shape = (ops[0].shape[0], 1, TILE_DIM, broadcast_dim, )
            else:
                const_shape = (1, 1, TILE_DIM, broadcast_dim, )
            tensor = torch.zeros(const_shape, dtype=buda_dataformat_to_pytorch_dtype(ops[0].output_df))
            tensor[:, :, 0, 0:output_dim] = 1.0 # column broadcast
            const = lc.tensor(tensor)
            if output_dim % TILE_DIM != 0:
                # this isn't just a tile broadcast any more, we're broadcasting to a special dim
                lc.op("matmul", (ops[0], const))
            else:
                lc.op("matmul", (ops[0], const), tag="tile_broadcast_c")

    elif type == "exp":
        lc.op("exp", ops, [], {"approximate_mode": "true" if "PYBUDA_EXP_APPROX" in os.environ else "false"})

    elif type == "reciprocal":
        lc.op("reciprocal", ops, [], {"approximate_mode": "true" if "PYBUDA_EXP_APPROX" in os.environ else "false"})

    elif type == "dropout":
        p, training, seed = attr
        if bool(training):
            r = ops[0].shape[-2] if len(ops[0].shape) > 1 else 1
            c = ops[0].shape[-1]
            buda_attr = {"p": p, "seed": seed}
            lc.op(type, ops, attr + [r, c, 1, 1, True, False], buda_attr) # straigh 1-1 for all other unaries
        else:
            lc.op(BudaNop.create(), ops)
    elif type == "gelu":
        lc.op("gelu", ops, attr, {"approximate_mode": "true" if attr[0] == "tanh" else "false"})
    elif type == "gelu_derivative":
        lc.op("gelu_derivative", ops, attr, {"approximate_mode": "true" if "PYBUDA_EXP_APPROX" in os.environ else "false"})

    elif type == "clip":
        
        min_value = attr[0]
        max_value = attr[1]

        # Inf protection
        if max_value > 65504.0:
            max_value = 65504.0

        if (min_value == 0) and (max_value >= 0):
            lc.op(BudaNop.create(relu_en=True, relu_threshold=max_value, relu_mode="max"), (ops[0], ))
            return

        shape = list(ops[0].shape.as_list())
        # Align up to tile 
        shape[-2] = ((shape[-2] - 1) // TILE_DIM + 1) * TILE_DIM
        shape[-1] = ((shape[-1] - 1) // TILE_DIM + 1) * TILE_DIM
        # Align up to 4 dimensions
        if len(shape) > 4:
            raise RuntimeError("Operator clip, operand must have number of dimensions less or equal to 4. ")
        if len(shape) < 4:
            shape = (4 - len(shape)) * [1] + shape

        min_value_tensor = lc.tensor(torch.zeros(shape) + min_value)
        max_value_tensor = lc.tensor(torch.zeros(shape) + max_value)
        diff_tensor = lc.tensor(torch.zeros(shape) + max_value - min_value)

        # General Formula/Algorithm
        # y = ReLU(x - min_value) + min_value
        # y = ReLU(0.0 - y + max_value) - max_value
        # y = 0.0 - y

        res = lc.op("subtract", (ops[0], min_value_tensor))
                # x - min_value
        res = lc.op(BudaNop.create(relu_en=True, relu_threshold=0.0, relu_mode="min"), (res, ))
                # ReLU(x - min_value)
        res = lc.op("subtract", (diff_tensor, res))
                # diff_value - ReLU(x - min_value), diff = max - min
        res = lc.op(BudaNop.create(relu_en=True, relu_threshold=0.0, relu_mode="min"), (res, ))
                # ReLU(diff_value - ReLU(x - min_value))
        lc.op("subtract", (max_value_tensor, res))
                # max_value - ReLU(diff_value - ReLU(x - min_value))

    elif type == "pow":
        if isinstance(attr[0], int):
            buda_attr = {"exp": attr[0]}
            lc.op("power", ops, attr, buda_attr)
        else:
            exponent_value = attr[0]
            shape = list(ops[0].shape.as_list()) 
            ln_x = lc.op(BudaLog.create(), ops)
            y_ln_x = lc.op("multiply", (lc.tensor(torch.zeros(shape) + exponent_value), ln_x)) 
            approximate_mode = "true" if "PYBUDA_EXP_APPROX" in os.environ else "false"
            lc.op(BudaExp.create(approximate_mode=approximate_mode), [y_ln_x])          

    else:
        # Find proper tile sizes
        if bool(int(os.environ.get("PYBUDA_ENABLE_TINY_TILE", "0"))):
            node_shape = list(ops[0].shape)
            tile_height = calculate_tile_size(node_shape[-2])
            tile_width = calculate_tile_size(node_shape[-1])
            buda_attr = {} if tile_height == TILE_DIM else {"vector": "r"}
        else:
            tile_height, tile_width = TILE_DIM, TILE_DIM
            buda_attr = {}
        lc.op(type, ops, attr, buda_attr, "", tile_height, TILE_DIM) # straigh 1-1 for all other unaries

def backward(type, attr, ac, operand, inputs, output, grad):

    assert len(inputs) == 1, "Eltwise unary should have one input"
    assert operand == 0, "Invalid operand index"
    assert ( 
            len(attr) == 0 or
            (type == "clip" and len(attr) == 2) or
            (type == "argmax" and len(attr) == 1) or
            (type == "leaky_relu" and len(attr) == 1) or
            (type == "relu" and len(attr) <= 2) or
            (type == "cumsum" and len(attr) == 2) or
            (type == "dropout" and len(attr) == 3) or
            (type == "tile_broadcast" and len(attr) == 2) or
            (type == "gelu" and len(attr) == 1) or 
            (type == "pow" and len(attr) == 1)
        ), "Eltwise unary should have no attributes, execpt for clip, leaky_relu and cumsum"

    if type == "nop":
        return ac.op(Nop.create(), (grad, ))

    if type == "tilizer":
        return ac.op(Nop.create(), (grad, ))

    if type == "tile_broadcast": # the full TM broadcast will generate a reduce
        return ac.op(Nop.create(), (grad, ))

    if type == "buffer":
        return ac.op(Buffer.create(), (grad, ))

    if type == "exp":
        return ac.op("multiply", (output, grad))

    if type == "reciprocal": # -1/x^2
        sq = ac.op("multiply", (output, output))
        neg = ac.op("multiply", (sq, ac.constant(-1)))
        return ac.op("multiply", (neg, grad))

    if type == "sqrt": # 0.5 / f(x)
        rec = ac.op(Reciprocal.create(), (output,))
        mult = ac.op("multiply", (rec, ac.constant(0.5)))
        return ac.op("multiply", (mult, grad))

    if type == "relu":
        # set theashold
        threshold = 0.0
        shape = inputs[0].shape.as_list()
        if len(attr) > 0:
            f32_epsilon = 1.19209289551e-07
            threshold = attr[0] - f32_epsilon 
        threshold_tensor = ac.tensor(torch.zeros(shape) + threshold)  
       
        # handle different modes 
        mode = "min"
        if len(attr) > 1:
            mode = attr[1]

        if mode == "min":
            relud = ac.op("greater_equal", (inputs[0], threshold_tensor))
        else:
            l_relud = ac.op("less", (inputs[0], threshold_tensor))
            g_relud = ac.op("greater_equal", (inputs[0], ac.tensor(torch.zeros(shape))))
            relud = ac.op("multiply", (l_relud, g_relud))

        return ac.op("multiply", (relud, grad))

    if type == "leaky_relu":
        alpha = ac.constant(attr[0])

        relu_dx = ac.op("heaviside", (output, ac.constant(0.0)))

        l_relu_dx = ac.op("multiply", (output, ac.constant(-1.0)))
        l_relu_dx = ac.op("heaviside", (l_relu_dx, ac.constant(0.0)))
        l_relu_dx = ac.op("multiply", (l_relu_dx, alpha))
        l_relu_dx = ac.op("add", (relu_dx, l_relu_dx))

        res = ac.op("multiply", (l_relu_dx, grad))

        return res

    if type == "gelu":
        gelud = ac.op("gelu_derivative", (inputs[0],), attr)
        return ac.op("multiply", (gelud, grad))

    if type == "log":
        recip = ac.op(Reciprocal.create(), (inputs[0],))
        return ac.op("multiply", (recip, grad))

    if type == "sigmoid":
        sigm_ = ac.op("subtract", (ac.constant(1), output))
        dsigm = ac.op("multiply", (output, sigm_))
        res = ac.op("multiply", (dsigm, grad))
        return res

    if type == "tanh":
        tanh_square = ac.op("multiply", (output, output))
        subtract = ac.op("subtract", (ac.constant(1), tanh_square))
        res = ac.op("multiply", (subtract, grad))
        return res

    if type == "argmax":
        raise RuntimeError("Argmax does not require grad and does not have a backwards function")
    
    if type == "cumsum":
        dim = attr[0]
        
        assert dim == 0, "Unsupported dim different then 0 for cumulative sum backward pass"
        
        if dim == 0:
            return ac.op(Nop.create(), (grad, ))
        
        return res

    if type == "dropout":
        return ac.op("dropout", (grad, ), attr)

    if type == "clip":
        x = inputs[0]
        shape = x.shape.as_list()
        min_value = attr[0]
        max_value = attr[1]
        min_value_tensor = ac.tensor(torch.zeros(shape) + min_value)
        max_value_tensor = ac.tensor(torch.zeros(shape) + max_value)

        ge_x = ac.op("greater_equal", (x, min_value_tensor))
        le_x = ac.op("less_equal", (x, max_value_tensor))
        mask = ac.op("multiply", (ge_x, le_x))
        res = ac.op("multiply", (mask, grad))
        return res

    elif type == "pow":
        exponent_value = attr[0]
        shape = list(inputs[0].shape.as_list())
        recip = ac.op(Reciprocal.create(), (inputs[0],))
        partial_grad = ac.op("multiply", (output, recip))  
        pow_grad = ac.op("multiply", (ac.tensor(torch.zeros(shape) + exponent_value), partial_grad))
        return ac.op("multiply", (pow_grad, grad))
    
    elif type == "abs":
        
        heaviside = ac.op("heaviside", (inputs[0], ac.constant(0.5)))
        subtract = ac.op("subtract", (heaviside, ac.constant(0.5)))
        stretched = ac.op("multiply", (subtract, ac.constant(2.0)))
        return ac.op("multiply", (stretched, grad))

    assert False, f"{type} not defined in eltwise unary backward."


def decompose(type, attr, dc, inputs):
    if type == "argmax":
        inp_node = inputs[0]
        axis = attr[0] if len(attr) > 0 else None

        if axis is None:
            import math
            inp_node = dc.op("reshape", [inp_node], (1, math.prod(inp_node.shape.as_list())))
            axis = -1

        input_shape = inp_node.shape.as_list()
        if axis >= 0:
            axis -= len(input_shape)
            assert axis < 0, "valid axis should be < 0 after subtracting len(input_shape)" 

        # First we want to get array of zeros and ones, with ones standing on the indices of maximums. 
        # For example, starting array is [1, 3, 5, 2, 0, 5]. We want to get [0, 0, 1, 0, 0, 1]. 
        # We do that by multiplying array with some large number (10^10), subtracting maximum of the array from array, 
        # then add 1 to each element to make sure that only maximums are now above 0 (equal to 1). 
        # Then we threshold the array with ReLu to get [0, 0, 1, 0, 0, 1]. 
        # Then we multiply that array with array of indices [0,1,2,3,4,5] to get [0,0,2,0,0,5]. 
        # The rest is manipulation how to extract first maximum index. 
        # We do that by taking complement of [0, 0, 1, 0, 0, 1] => [1, 1, 0, 1, 1, 0] and multiplying it 
        # with size(6) and add it to [0,0,2,0,0,5] => [6,6,2,6,6,5] and just find argmin of this array which is 2.

        data_type = buda_dataformat_to_pytorch_dtype(inp_node.output_df)
        indices_shape = [dim if i == axis + len(input_shape) else 1 for i, dim in enumerate(input_shape)]

        indices = torch.arange(input_shape[axis], dtype=data_type).reshape(indices_shape)
        indices_tensor = dc.tensor(indices)

        factor = torch.ones((input_shape), dtype=data_type) * 1e10
        factor_tensor = dc.tensor(factor)

        ones = torch.ones((input_shape), dtype=data_type) 
        ones_tensor = dc.tensor(ones)
        negative_ones = dc.tensor(ones * (-1))

        # this it the tensor that has all elements equal to input shape on axis on which we do argmax.
        offset_tensor = dc.tensor(ones * input_shape[axis])

        scaled_input = dc.op("multiply", (inp_node, factor_tensor),)
        max_1 = dc.op("reduce_max", [scaled_input], [axis])
        scaled_input = dc.op("subtract", (scaled_input, max_1))
        scaled_input = dc.op("add", [scaled_input, ones_tensor],)

        relu_1 = dc.op("relu", (scaled_input,))
        relu_1_complement = dc.op("subtract", (ones_tensor, relu_1))

        mul_1 = dc.op("multiply", [relu_1, indices_tensor],)
        mul_2 = dc.op("multiply", [relu_1_complement, offset_tensor],)
        add_1 = dc.op("add", [mul_1, mul_2],)
        negative_add_1 = dc.op("multiply", [add_1, negative_ones])
        negative_argmax = dc.op("reduce_max", [negative_add_1], [axis])

        output_neg_ones = torch.ones((negative_argmax.shape.as_list()), dtype=data_type) * (-1)
        output_neg_ones_tensor = dc.tensor(output_neg_ones)
        argmax = dc.op("multiply", [negative_argmax, output_neg_ones_tensor])

        dc.fuse(argmax)

    elif type == "sigmoid" and bool(int(os.environ.get("PYBUDA_DECOMPOSE_SIGMOID", "0"))):
        inp = inputs[0]
        minus_one = dc.tensor(torch.ones([1,1]) * -1)
        plus_one = dc.tensor(torch.ones([1,1]))
        neg_ = dc.op("multiply", [inp, minus_one])
        exp_ = dc.op(Exp.create(), [neg_])
        result = dc.op("add", [plus_one, exp_])
        result = dc.op(Reciprocal.create(), [result])
        dc.fuse(result)

    elif type == "gelu" and bool(int(os.environ.get("PYBUDA_DECOMPOSE_GELU", "0"))):
        inp_node = inputs[0]
        data_type = buda_dataformat_to_pytorch_dtype(inp_node.output_df)
        one_half = dc.tensor(torch.ones((1), dtype=data_type) * 0.5)
        sqrt_2pi = dc.tensor(torch.ones((1), dtype=data_type) * 0.79788)
        one = dc.tensor(torch.ones((1), dtype=data_type))
        const = dc.tensor(torch.ones((1), dtype=data_type) * 0.044715)
        x_squared = dc.op("multiply", [inp_node, inp_node])
        x_cubed = dc.op("multiply", [inp_node, x_squared])
        x_cuber_times_const = dc.op("multiply", [x_cubed, const])
        plus_x = dc.op("add", [x_cuber_times_const, inp_node])
        times_sqrt_2pi = dc.op("multiply", [plus_x, sqrt_2pi])
        tanh = dc.op(Tanh.create(), [times_sqrt_2pi])
        plus_one = dc.op("add", [tanh, one])
        times_x = dc.op("multiply", [plus_one, inp_node])
        result = dc.op("multiply", [times_x, one_half])
        dc.fuse(result)


def initial_flops_estimate(type, attr, ops):
    flops = 0
    sfpu_unary_ops = ["exp", "sqrt", "relu", "leaky_relu", "gelu", "gelu_derivative", "reciprocal", "log", "sigmoid", "abs", "cosine", "sine", "argmax", "tanh", "cumsum", "pow",]
    output_shape = shape(type, attr, ops)[0]
    
    if type in sfpu_unary_ops:
        flops = np.prod(output_shape)
    
    return flops
    
