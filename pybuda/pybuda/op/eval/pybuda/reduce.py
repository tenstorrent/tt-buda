# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from ..common import to_torch_operands
from ....pybudaglobal import TILE_DIM, align_up_tile
from ....tensor import buda_dataformat_to_pytorch_dtype
from .transpose import TransposeTM
import torch
import numpy as np
import math

def eval(type, attr, ops):
    assert len(ops) == 1, "Reduce should have one input"
    assert len(attr) == 1 or len(attr) == 2 and type == "reduce_max" or len(attr) == 3 and type == "grouped_reduce_avg", "Reduce should have one dim parameter, and optional stride attr OR mandatory groups attr for grouped reduce."

    t_ops = to_torch_operands(*ops)
    f = {
        "reduce_sum": lambda i: torch.sum(t_ops[0], attr[0], keepdim=True),
        "reduce_avg": lambda i: torch.mean(t_ops[0], attr[0], keepdim=True),
        "reduce_max": lambda i: torch.max(t_ops[0], dim=attr[0], keepdim=True)[0],
    }

    if len(attr) == 2 and type == "reduce_max" and attr[0] == -3:
        z = attr[1]
        assert t_ops[0].shape[-3] % z == 0
        ret = t_ops[0].squeeze(0)
        squeeze_failed = False
        if len(ret.shape) == len(t_ops[0].shape):
            squeeze_failed = True
        ret = t_ops[0].squeeze(0).split(z)
        ret = [torch.max(s, dim=0)[0] for s in ret]
        result = torch.stack(ret).unsqueeze(0) if not squeeze_failed else torch.stack(ret)
    elif type == "grouped_reduce_avg":
        keep_dims = attr[2]
        groups = attr[1]
        dim = attr[0]
        assert t_ops[0].shape[dim] % groups == 0, "Groups must be a factor of the dimension size."
        newshape = list(t_ops[0].shape[:dim]) + [groups] + list(t_ops[0].shape[dim+1:]) if not keep_dims else t_ops[0].shape
        group_size = t_ops[0].shape[dim] // groups
        ret = t_ops[0].reshape(list(t_ops[0].shape[:dim]) + [groups, group_size] + list(t_ops[0].shape[dim+1:]))
        if dim >= 0:
            dim +=1
        ret = torch.mean(ret, dim=dim, keepdim=True)
        if keep_dims:
            ret = torch.cat([ret] * group_size, dim=dim)
        result = ret.reshape(newshape)
    else:
        assert type in f, f"{type} not defined in eval map for reduce ops."
        result = f[type](t_ops)

    return result

def shape(type, attr, ops):
    assert len(ops) == 1, "Reduce should have one input"
    assert len(attr) == 1 or len(attr) == 2 and type == "reduce_max" or len(attr) == 3 and type == "grouped_reduce_avg", "Reduce should have one dim parameter, and optional stride attr OR mandatory groups attr for grouped reduce."

    ret = list(ops[0])
    if len(attr) == 2 and type == "reduce_max":
        ret[attr[0]] = ret[attr[0]] // attr[1]
    elif type == "grouped_reduce_avg":
        if not attr[2]:
            ret[attr[0]] = attr[1]
    else:
        if isinstance(attr[0], list):
            for dim in attr[0]:
                ret[dim] = 1
        else:
            ret[attr[0]] = 1

    return tuple(ret), []

def lower(type, attr, lc, ops, outputs):
    assert len(ops) == 1, "Reduce should have one input"
    assert len(attr) in [1, 2, 3], "Reduce should have one dim parameter, and an optional 'tile broadcast' one"

    inp_shape = ops[0].shape
    reduce_dim = attr[0]
    if len(attr) == 2 and type != "reduce_max":
        tile_broadcast = attr[1] == 1
    else:
        tile_broadcast = False

    input_shape = lc.shape(ops[0])
    if reduce_dim < 0:
        reduce_dim += len(input_shape)

    if type == "reduce_max":
        reduce_dim += 4 - len(input_shape)
        if reduce_dim in [2, 3] and input_shape[attr[0]] % TILE_DIM != 0:
            # Reduce dim is not a multiple of tile dim, so we need to pad it with negative inf 
            # to make sure we don't get a wrong max value
            const_shape = input_shape
            zero_tensor = torch.zeros(const_shape)
            if reduce_dim == 2:
                neg_inf_tensor = torch.nn.functional.pad(
                    zero_tensor, 
                    (0,0,0,TILE_DIM - const_shape[-2] % TILE_DIM,), 
                    mode='constant', 
                    value=-1e6
                )
                if const_shape[-1] % TILE_DIM != 0:
                    neg_inf_tensor = torch.nn.functional.pad(
                        neg_inf_tensor, 
                        (0,TILE_DIM - const_shape[-1] % TILE_DIM,), 
                        mode='constant', 
                        value=0
                    )
            elif reduce_dim == 3:
                neg_inf_tensor = torch.nn.functional.pad(
                    zero_tensor, 
                    (0, TILE_DIM - const_shape[-1] % TILE_DIM,), 
                    mode='constant', 
                    value=-1e6
                )
                if const_shape[-2] % TILE_DIM != 0:
                    neg_inf_tensor = torch.nn.functional.pad(
                        neg_inf_tensor, 
                        (0,0,0,TILE_DIM - const_shape[-2] % TILE_DIM,), 
                        mode='constant', 
                        value=0,
                    )

            while len(neg_inf_tensor.shape) < 4:
                neg_inf_tensor = torch.unsqueeze(neg_inf_tensor, 0)

            const = lc.tensor(neg_inf_tensor)
            padded_input = lc.op("add", [ops[0], const], (), {})
            ops = [padded_input]

        buda_attr = {"dim": ["w", "z", "r", "c"][reduce_dim], "type": "max"}
        z = input_shape[1]
        if reduce_dim == 1:
            z = input_shape[-3] if len(attr) == 1 else attr[1]
            buda_attr["z"] = z
        lc.op("reduce", ops, (reduce_dim, "max", z), buda_attr)
        return

    dtype = buda_dataformat_to_pytorch_dtype(ops[0].output_df)
    is_z = (len(input_shape) - reduce_dim) == 3
    if is_z:
        if type == "grouped_reduce_avg":
            raise NotImplementedError("Grouped reduce avg not implemented for Z-dim")
        else:
            tensor = torch.eye(align_up_tile(ops[0].shape[-1])).unsqueeze(0).unsqueeze(0).to(dtype)
        if type == "reduce_avg":
            tensor /= inp_shape[-3]
        const = lc.tensor(tensor)
        accumulate = True
        lc.op("matmul", [ops[0], const], (accumulate,), {"accumulate": True, "z": input_shape[-3]}, tag="reduce_z")
        return

    const_value = 1
    if type == "reduce_avg":
        const_value = 1 / inp_shape[reduce_dim]
    elif type == "grouped_reduce_avg":
        const_value = attr[1] / inp_shape[reduce_dim]
        
    def pad_to_tile_dim(n):
        if n % TILE_DIM == 0:
            return n
        return n + TILE_DIM - (n % TILE_DIM)


    reduce_len = input_shape[reduce_dim]

    if reduce_len == 1 and not tile_broadcast:
        # Nothing to reduce
        lc.op("nop", ops)
        return

    if reduce_len % TILE_DIM == 0 and type != "grouped_reduce_avg":
        broadcast_const = True
        if tile_broadcast:
            tensor = torch.full(size=(1, 1, TILE_DIM, TILE_DIM), fill_value=const_value, dtype=dtype)
        else:
            tensor = torch.zeros(1, 1, TILE_DIM, TILE_DIM, dtype=dtype)
            if reduce_dim == len(input_shape) - 1:
                tensor[:, :, :, 0] = const_value
            else:
                tensor[:, :, 0, :] = const_value
    else:
        torch.set_printoptions(threshold=10000, linewidth=10000)
        padded_reduce_len = pad_to_tile_dim(reduce_len)
        broadcast_const = False
        if reduce_dim == len(input_shape) - 1:
            if type == "grouped_reduce_avg":
                groups = attr[1]
                keep_dims = attr[2]
                group_size = reduce_len // groups
                tensor = torch.zeros(1, 1, padded_reduce_len, align_up_tile(groups if not keep_dims else reduce_len), dtype=dtype)
                for g in range(groups):
                    if not keep_dims:
                        tensor[:, :, (g*group_size):((g+1)*group_size), g] = const_value
                    else:
                        tensor[:, :, (g*group_size):((g+1)*group_size), (g*group_size):((g+1)*group_size)] = const_value
            else:
                tensor = torch.zeros(1, 1, padded_reduce_len, TILE_DIM, dtype=dtype)
                if tile_broadcast:
                    tensor[:, :, 0:reduce_len, :] = const_value
                else:
                    tensor[:, :, 0:reduce_len, 0] = const_value
        elif reduce_dim == len(input_shape) - 2:
            if type == "grouped_reduce_avg":
                groups = attr[1]
                keep_dims = attr[2]
                group_size = reduce_len // groups
                tensor = torch.zeros(1, 1, align_up_tile(groups if not keep_dims else reduce_len), padded_reduce_len, dtype=dtype)
                for g in range(groups):
                    if not keep_dims:
                        tensor[:, :, g, (g*group_size):((g+1)*group_size)] = const_value
                    else:
                        tensor[:, :, (g*group_size):((g+1)*group_size), (g*group_size):((g+1)*group_size)] = const_value
                
            else:
                tensor = torch.zeros(1, 1, TILE_DIM, padded_reduce_len, dtype=dtype)
                if tile_broadcast:
                    tensor[:, :, :, 0:reduce_len] = const_value
                else:
                    tensor[:, :, 0, 0:reduce_len] = const_value
        else:
            raise RuntimeError(f"Lowered reduce only supported for rows and columns. Reduce dim={reduce_dim}")

    const = lc.tensor(tensor)

    if reduce_dim == len(input_shape) - 2:
        # Row reduce
        matmul = lc.op("matmul", (const, ops[0]), tag = "reduce_r")
        inner_matmul_dim_tiles = pad_to_tile_dim(input_shape[len(input_shape) - 2]) // TILE_DIM
        if broadcast_const:
            lc.set_broadcast_dim(const, matmul, 3, inner_matmul_dim_tiles)
        if len(input_shape) >= 3 and input_shape[len(input_shape) - 3] > 1:
            lc.set_broadcast_dim(const, matmul, 1, input_shape[len(input_shape) - 3])

    elif reduce_dim == len(input_shape) - 1:
        # Column reduce
        matmul = lc.op("matmul", (ops[0], const), tag = "reduce_c")
        inner_matmul_dim_tiles = pad_to_tile_dim(input_shape[len(input_shape) - 1]) // TILE_DIM
        if broadcast_const:
            lc.set_broadcast_dim(const, matmul, 2, inner_matmul_dim_tiles)
        if len(input_shape) >= 3 and input_shape[len(input_shape) - 3] > 1:
            lc.set_broadcast_dim(const, matmul, 1, input_shape[len(input_shape) - 3])

    else:
        raise RuntimeError(f"Lowered reduce only supported for rows and columns. Reduce dim={reduce_dim}")

def backward(type, attr, ac, operand, inputs, output, grad):

    assert len(inputs) == 1, "Reduce should have one input"
    assert len(attr) == 1 or len(attr) == 2 and type == "reduce_max" or len(attr) == 3 and type == "grouped_reduce_avg", "Reduce should have one dim parameter, and optional stride attr OR mandatory groups attr for grouped reduce."

    if type == "reduce_max":
        in0 = inputs[0]
        if len(attr) == 2:
            stride = attr[1]
            fast = False
        else:
            stride = inputs[0].shape[attr[0]]

        one_torch = torch.tensor([1.0]).reshape([1] * len(in0.shape))
        one = ac.tensor(one_torch)
        threshold = 1.0
        fast = False
        if fast:
            # This version treats multiple maximal values equally (unlike pytorch)
            mask = ac.op("subtract", [in0, output])    # has 0.0 in max positions and < 0.0 everywhere else
            mask = ac.op("add", [mask, one])           # has 1.0 in max positions and < 1.0 everywhere else
            mask = ac.op("relu", [mask], (threshold, )) # has 1.0 in max posistions, 0.0 everywhere else
            return ac.op("multiply", [grad, mask])
        else:
            # This version takes only the first of multiple maximal values (like pytorch)
            dim = attr[0]
            neg_range_torch = -(torch.arange(in0.shape[dim]) - in0.shape[dim]).float()
            shape = [1] * len(in0.shape)
            shape[dim] = neg_range_torch.shape[0]
            neg_range_torch = neg_range_torch.reshape(shape)
            neg_range = ac.tensor(neg_range_torch)
            mask = ac.op("subtract", [in0, output])     # has 0.0 in max positions and < 0.0 everywhere else
            mask = ac.op("add", [mask, one])            # has 1.0 in max positions and < 1.0 everywhere else
            mask = ac.op("relu", [mask], (threshold, ))  # has 1.0 in max posistions, 0.0 everywhere else
            mask = ac.op("multiply", [mask, neg_range]) # puts range N...1 in max positions, 0.0 everywhere else
            redc = ac.op("reduce_max", [mask], (dim, stride))  # argmax
            mask = ac.op("subtract", [mask, redc])      # Orig range - argmax, 0.0 in FIRST max position
            mask = ac.op("add", [mask, one])            # has 1.0 is first max position, and < 1.0 everywhere else
            mask = ac.op("relu", [mask], (threshold, ))  # has 1.0 is first max position, and 0.0 everywhere else
            return ac.op("multiply", [grad, mask])

    if type == "reduce_sum":
        return ac.op("nop", (grad, )) # the broadcast will be implicitly figured out during shape calculations

    if type == "reduce_avg":
        dim = attr[0]
        size = ac.get_shape(inputs[0])[dim]
        return ac.op("multiply", (grad, ac.constant(1/size)))
    
    if type == "grouped_reduce_avg":
        dim = attr[0]
        groups = attr[1]
        keep_dims = attr[2]
        
        group_size = ac.get_shape(inputs[0])[dim] // groups
        
        if dim >= 0:
            dim -= len(grad.shape)
        
        cols = []
        rows = []
        if not keep_dims:
            for i in range(groups):
                cols.extend([i]*group_size)
            rows = list(range(len(cols)))
        else:
            for i in range(groups):
                cols.extend(list(range(i*group_size, (i+1)*group_size))*group_size)
                for j in range(group_size):
                    rows.extend([i*group_size + j]*group_size)

        sparse = torch.sparse_coo_tensor((rows, cols), torch.ones(len(cols)), (max(rows)+1, grad.shape[dim]), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sparse = torch.cat([sparse]*grad.shape[-3], dim=-3)
        spm = ac.tensor(sparse)
        
        if dim == -1:
            grad = ac.op(TransposeTM.create(-2, -1), [grad])
            
        grad = ac.op("sparse_matmul", (spm, grad))
        
        if dim == -1:
            grad = ac.op(TransposeTM.create(-2, -1), [grad])
        
        size = ac.get_shape(inputs[0])[dim] // groups
        return ac.op("multiply", (grad, ac.constant(1/size)))

    raise NotImplementedError("Unknown type of reduce")


def decompose(type, attr, dc, inputs):
    assert len(inputs) == 1, "Reduce should have one input"
    assert len(attr) == 1 or len(attr) == 2 and type == "reduce_max" or len(attr) == 3 and type == "grouped_reduce_avg", "Reduce should have one dim parameter, and optional stride attr OR mandatory groups attr for grouped reduce."

    if isinstance(attr[0], list):
        x = inputs[0]
        for dim in attr[0]:
            x = dc.op("reduce_avg", [x], (dim,))
        dc.fuse(x)
        return

    inp_shape = inputs[0].shape.as_list()
    if inp_shape[attr[0]] == 1:
        # This is a NOP
        result = dc.op("nop", inputs, ())
        dc.fuse(result)
    elif (type == "reduce_sum" or type == "reduce_avg"):
        dim = attr[0]

        if dim >= 0:
            dim -= len(inputs[0].shape)

        if dim == -4:
            result = dc.op(TransposeTM.create(0,1), inputs)
            result = dc.op(type, [result], (1,))
            result = dc.op(TransposeTM.create(0,1), [result])
            dc.fuse(result)


def decompose_post_autograd(op_type, attr, dc, inputs): 
    if (op_type == "reduce_sum" or op_type == "reduce_avg"):
        dim = attr[0]
        if dim >= 0:
            dim -= len(inputs[0].shape)

        if dim == -4:
            result = dc.op(TransposeTM.create(0,1), inputs)
            result = dc.op(op_type, [result], (1,))
            result = dc.op(TransposeTM.create(0,1), [result])
            dc.fuse(result)
            return


def initial_flops_estimate(type, attr, ops):
    flops = 0
    reduce_ops = ["reduce_max", "reduce_sum", "reduce_avg"]
    output_shape = shape(type, attr, ops)[0]
    if type in reduce_ops:
        flops = np.prod(output_shape)
    
    return flops

