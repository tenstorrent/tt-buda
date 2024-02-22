# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


from math import gamma
import numpy as np
import torch
import torch.nn.functional as F

from ..common import to_torch_operands
from . import reduce
from .exp import Exp
from .reciprocal import Reciprocal
from .log import Log


def eval(op_type, attr, ops):
    """
    Operator or module evaluation function. Evaluation is done using PyTorch ML library.

    Parameters
    ----------
    op_type:
        Type of the operation.

    attr:
        Operation attributes.

    ops:
        Operation operands.

    Returns
    -------
        Result of the evaluation.

    """
    
    if op_type == "softmax":
        
        assert len(ops) == 1, "Softmax should have one operand."
        assert len(attr) == 2, "Softmax should have two attributes."

        t_ops = to_torch_operands(*ops)
        input_ = t_ops[0]
        dim = attr[0]
        stable = attr[1]
        assert input_.dim() > dim, "Given dimension is out of the shape"
        return F.softmax(input_, dim=dim)

    if op_type == "log_softmax":
        
        assert len(ops) == 1, "LogSoftmax should have one operand."
        assert len(attr) == 2, "LogSoftmax should have two attributes."

        t_ops = to_torch_operands(*ops)
        input_ = t_ops[0]
        dim = attr[0]

        assert input_.dim() > dim, "Given dimension is out of the shape"
        
        result = F.log_softmax(input_, dim=dim)

        return result

    if op_type == "softmax_bw":

        assert len(ops) == 3, "Softmax backward should have three operands."
        assert len(attr) == 1, "Softmax backward should have one attribute."

        t_ops = to_torch_operands(*ops)

        input_ = t_ops[0]   # the input of the softmax
        output = t_ops[1]   # the output of the softmax function
        grad = t_ops[2]    # gradient from the previous layer
        dim = attr[0]  # the dimension by which we do softmax

        assert input_.dim() > dim and dim >= -output.dim(), "Given dimesnion is out of the shape"

        # assert output.dim() > dim and dim >= -output.dim(), "Given dimension is out of the shape"

        # result = (grad - torch.sum(grad * output, dim=dim, keepdim=True)) * output
        # return result

        input__ = input_.clone().detach()
        input__.requires_grad = True
        output = F.softmax(input__, dim=dim)
        output.backward(gradient=grad)

        return input__.grad

    if op_type == "layernorm":
        
        assert len(ops) == 3, "Layernorm should have three operands."
        assert len(attr) == 2, "Layernorm should have two attributes."

        t_ops = to_torch_operands(*ops)

        input_ = t_ops[0]   # Input tensor
        gamma = t_ops[1]   # weights, weight re-scaling parameter
        beta = t_ops[2]      # bias, weight re-centering parameter
        dim = attr[0]       # It should be the last dimension or last N dimensions ????
        epsilon = attr[1]

        assert dim == -1 or dim == input_.dim() - 1, "Normalization can be done only over the last dimension"
        assert gamma.shape[-1] == input_.shape[-1], "Weights shape must be the same as normalized shape."
        for gdim in gamma.shape[:-1]:
            assert gdim == 1, "All dimensions but the last one must be 1"
        assert beta.shape[-1] == input_.shape[-1], "Bias shape must be the same as normalized shape."
        for bdim in beta.shape[:-1]:
            assert bdim == 1, "All dimensions but the last one must be 1"

        return F.layer_norm(
                    input=input_,
                    normalized_shape=input_.shape[-1:],
                    weight=gamma.reshape(gamma.shape[-1:]),
                    bias=beta.reshape(beta.shape[-1:]),
                    eps=epsilon
                )

    if op_type == "layernorm_bw":
        
        # assert len(ops) == 4, "Layernorm should have four operands."
        assert len(ops) == 5, "Layernorm should have five operands."
        assert len(attr) == 3, "Layernorm should have three attributes."

        t_ops = to_torch_operands(*ops)

        input_ = t_ops[0]   # Input tensor
        gamma = t_ops[1]   # gamma, weight re-scaling parameter
        beta = t_ops[2]      # beta, weight re-centering parameter
        grad = t_ops[3]      # gradient from the prevoius layer
        dim = attr[0]       # It should be tha last dimension or last N dimensions ????
        epsilon = attr[1]
        operand = attr[2]   # Index of the operand, X - 0, weights - 1, bias - 2

        assert dim == -1 or dim == input_.dim() - 1, "Normalization can be done only over the last dimension."
        assert gamma.shape[-1] == input_.shape[-1], "Weights shape must be the same as normalized shape."
        for gdim in gamma.shape[:-1]:
            assert gdim == 1, "All dimensions but the last one must be 1"
        assert beta.shape[-1] == input_.shape[-1], "Bias shape must be the same as normalized shape."
        for bdim in beta.shape[:-1]:
            assert bdim == 1, "All dimensions but the last one must be 1"
        assert operand in range(3), "Operand index out of range."

        input__ = input_.clone().detach()
        gamma__ = gamma.clone().detach()
        beta__ = beta.clone().detach()
        input__.requires_grad = True
        gamma__.requires_grad = True
        beta__.requires_grad = True
        output = F.layer_norm(
                        input__,
                        gamma__.shape[-1:],
                        gamma__.reshape(gamma__.shape[-1:]),
                        beta__.reshape(beta__.shape[-1:]),
                        epsilon
                    )
        output.backward(gradient=grad)

        if operand == 0:
            return input__.grad
        if operand == 1:
            return gamma__.grad.reshape(gamma__.shape)
        if operand == 2:
            return beta__.grad.reshape(beta__.shape)

    if op_type == "batchnorm":
        
        assert len(ops) == 5, "batchnorm should have five operands."
        assert len(attr) == 1, "batchnorm should have one attributes."

        t_ops = to_torch_operands(*ops)

        input_ = t_ops[0]   # Input tensor
        weight = t_ops[1]   # weights, weight re-scaling parameter
        bias = t_ops[2]      # bias, weight re-centering parameter
        running_mean = t_ops[3]
        running_var = t_ops[4] 
        epsilon = attr[0]
 
        #assert gamma.shape[-1] == input_.shape[-1], "Weights shape must be the same as normalized shape."
        #for gdim in gamma.shape[:-1]:
        #    assert gdim == 1, "All dimensions but the last one must be 1"
        #assert beta.shape[-1] == input_.shape[-1], "Bias shape must be the same as normalized shape."
        #for bdim in beta.shape[:-1]:
        #    assert bdim == 1, "All dimensions but the last one must be 1"

        return F.batch_norm(
                    input=input_,
                    running_mean=running_mean.shape[-1:],
                    running_var=running_var.shape[-1:],
                    normalized_shape=input_.shape[-1:],
                    weight=weight.reshape(gamma.shape[-1:]),
                    bias=bias.reshape(beta.shape[-1:]),
                    eps=epsilon
                )

    assert False, f"{op_type} is not defined in nn eval."


def shape(op_type, attr, ops):
    """
    Computes output shapes for particular operation.

    Parameters
    ----------
    op_type:
        Type of the operation.

    attr:
        Operation attributes.

    ops:
        Operation operands.

    Returns
    -------
        Output shape after particular operation.

    """

    if op_type == "softmax":
        
        assert len(ops) == 1, "Softmax should have one operand."
        assert len(attr) == 2, "Softmax should have two attributes."

        return ops[0], []

    if op_type == "log_softmax":
        
        assert len(ops) == 1, "LogSoftmax should have one operand."
        assert len(attr) == 2, "LogSoftmax should have two attributes."

        return ops[0], []


    if op_type == "softmax_bw":

        assert len(ops) == 3, "Softmax should have three operands."
        assert len(attr) == 1, "Softmax backward should have one attribute."

        return ops[0], []

    if op_type == "layernorm":
        
        assert len(ops) == 3, "Layernorm should have three operands."
        assert len(attr) == 2, "Layernorm should have two attributes."

        return ops[0], []

    if op_type == "layernorm_bw":
        
        # assert len(ops) == 4, "Layernorm should have four operands."
        assert len(ops) == 5, "Layernorm should have five operands."
        assert len(attr) == 3, "Layernorm should have three attributes."

        operand = attr[2]   # Index of the operand, X - 0, weights - 1, bias - 2

        assert operand in range(3), "Operand index out of range"
        
        return ops[operand], []

    if op_type == "batchnorm":
        
        assert len(ops) == 5, "Layernorm should have five operands."
        assert len(attr) == 1, "Layernorm should have one attributes."

        return ops[0], []

    assert False, f"{op_type} is not defined in nn shape."


def lower(op_type, attr, lc, ops, outputs):
    """
    Translates complex operations into simpler buda operations.

    Parameters
    ----------
    op_type:
        Type of the operation.

    attr:
        Operator attributes.

    lc:
        Lowering Context, PyBuda C++ API for breaking 
        Pybuda graph/subgraph into Buda operations.

    ops:
        Input operands, tensors.

    Returns
    -------
        No return value.

    """
    pass


def backward(op_type, attr, ac, operand, inputs, output, grad):
    """
    Computes backward value for particular operation using derivative 
    of the current operation and gradient from the previous node.


    Parameters
    ----------
    op_type:
        Type of the operation.

    attr:
        Operation attributes.

    ac:
        Autograd Context, PyBuda C++ API for automatic gradient computation. 

    operand:
        Operation operands.

    inputs:
        Operation inputs.

    output:
        Operation outpus, i.e. reuslt of the operation, from forward pass. 
        If it's needed in backward it shouldn't be computed again, just re-used.
    
    grad:
        Gradient of the previous node.

    Returns
    -------
        Result of the backward pass.

    """
    
    if op_type == "softmax":

        assert len(inputs) == 1, "Softmax should have one operand."
        assert len(attr) == 2, "Softmax should have two attributes."

        dim = attr[0]

        return ac.op("softmax_bw", (inputs[0], output, grad), (dim, ))

    if op_type == "layernorm":
        
        assert len(inputs) == 3, "Layernorm should have three operands."
        assert len(attr) == 2, "Layernorm should have two attributes."
        assert operand in range(len(inputs)), "Operand index out of the input range."

        # Add operand indices to attributes
        attr = list(attr)
        attr.append(operand)
        attr = tuple(attr)
        
        # Add gradient and intermidate results from forward pass
        inputs = list(inputs)
        inputs += [grad, output]
        inputs = tuple(inputs)

        return ac.op("layernorm_bw", inputs, attr)

    if op_type == "batchnorm":
        raise NotImplementedError("Back propagation for Batchnorm op is not implemented yet")

    assert False, f"{op_type} is not defined in nn backward. "

def decompose(op_type, attr, dc, inputs):
    """
    Decompses the operator after backward pass.

    Parameters
    ----------
    op_type:
        Type of the operation.

    attr:
        Operation attributes.

    dc:
        Decomposing Context, PyBuda C++ API for breaking 
        Pybuda graph/subgraph to simpler, PyBuda graph, too.

    inputs:
        Operation inputs.

    Returns
    -------
        Result of the operation.
    
    """
    
    if op_type == "log_softmax":
        assert len(inputs) == 1, "Softmax should have one operand."
        assert len(attr) == 2, "Softmax should have two attributes."
        x = inputs[0]
        dim = attr[0]
        stable = attr[1]
        result = dc.op("softmax", (x, ), (dim, stable))
        result = dc.op(Log.create(), (result, ))
        dc.fuse(result)
        return

    if op_type == "batchnorm": 
        assert len(inputs) == 5, "Batchnorm should have five operands."
        assert len(attr) == 1, "Layernorm should have one attributes."

        input_ = inputs[0]
        weight = inputs[1]
        bias = inputs[2]
        running_mean = inputs[3]
        running_var = inputs[4]
        epsilon = attr[0]

        # const tensor
        eps_tensor = dc.tensor(torch.zeros(running_var.shape.as_list()) + epsilon)
        neg_one = dc.tensor(torch.zeros(running_mean.shape.as_list()) - 1.0)

        # decompose
        var_eps = dc.op("add", (running_var, eps_tensor), ())
        sqrt = dc.op("sqrt", (var_eps,), ())
        recipro = dc.op(Reciprocal.create(), (sqrt,), ())
        weighted = dc.op("multiply", (recipro, weight), ())
        neg_mean = dc.op("multiply", (neg_one, running_mean), ())
        weighted_mean = dc.op("multiply", (weighted, neg_mean), ())
        weighted_bias = dc.op("add", (weighted_mean, bias), ())
        weighted_bias = dc.op("unsqueeze", [weighted_bias], (1, len(weighted_bias.shape),))
        weighted_bias = dc.op("unsqueeze", [weighted_bias], (1, len(weighted_bias.shape),))
        weighted_var = dc.op("unsqueeze", [weighted], (1, len(weighted.shape),))
        weighted_var = dc.op("unsqueeze", [weighted_var], (1, len(weighted_var.shape),))
        scaled = dc.op("multiply", (input_, weighted_var), ())
        biased = dc.op("add", (scaled, weighted_bias), ())
        dc.fuse(biased)
        return

def decompose_post_autograd(op_type, attr, dc, inputs):
    """
    Decompses the operator after backward pass.

    Parameters
    ----------
    op_type:
        Type of the operation.

    attr:
        Operation attributes.

    dc:
        Decomposing Context, PyBuda C++ API for breaking 
        Pybuda graph/subgraph to simpler, PyBuda graph, too.

    inputs:
        Operation inputs.

    Returns
    -------
        Result of the operation.
    
    """
    
    if op_type == "softmax":
        
        assert len(inputs) == 1, "Softmax should have one operand."
        assert len(attr) == 2, "Softmax should have two attributes."
        x = inputs[0]
        dim = attr[0]
        stable = attr[1]

        if stable and dc.get_compiler_cfg().enable_stable_softmax:
            res_max = dc.op("reduce_max", (x, ), (dim, ))
            res_x_max = dc.op("subtract", (x, res_max), ())
            res_exp = dc.op(Exp.create(), (res_x_max, ), ())
        else:
            res_exp = dc.op(Exp.create(), (x, ), ())
            

        res_exp_sum = dc.op("reduce_sum", (res_exp, ), (dim, ))
        res_exp_sum = dc.op("add", (res_exp_sum, dc.tensor(torch.zeros(res_exp_sum.shape.as_list()) + 1e-10)), ())
        res_exp_sum_recip = dc.op(Reciprocal.create(), (res_exp_sum, ), ())
        result = dc.op("multiply", (res_exp, res_exp_sum_recip), ())
        dc.fuse(result)
        return

    if op_type == "softmax_bw":

        assert len(inputs) == 3, "Softmax backward should have three operands."
        assert len(attr) == 1, "Softmax backward should have one attribute."

        output = inputs[1]   # the output of the softmax function
        grad = inputs[2]    # gradient from the previous layer
        dim = attr[0]  # the dimension by which we do softmax
        out_shape = output.shape.as_list()

        assert len(out_shape) > dim and dim >= -len(out_shape), "Given dimension is out of the shape"

        grad_out = dc.op("multiply", (grad, output), ())
        gout_sum = dc.op("reduce_sum", (grad_out, ), (dim, ))
        gout_sub = dc.op("subtract", (grad, gout_sum), ())
        result = dc.op("multiply", (gout_sub, output), ())
        dc.fuse(result)
        return

    if op_type == "layernorm":
        
        assert len(inputs) == 3, "Layernorm should have three operands."
        assert len(attr) == 2, "Layernorm should have two attributes."

        input_ = inputs[0]
        weights = inputs[1]   # gamma, weight re-scaling parameter
        bias = inputs[2]      # beta, weight re-centering parameter
        dim = attr[0]       # It should be tha last dimension or last N dimensions ????
        epsilon = attr[1]

        input_shape = input_.shape.as_list()
        gamma_shape = weights.shape.as_list()
        beta_shape = bias.shape.as_list()

        assert dim == -1 or dim == len(input_shape) - 1, "Normalization can be done only over the last dimension."
        assert gamma_shape[-1] == input_shape[-1], "Weights shape must be the same as normalized shape."
        for gdim in gamma_shape[:-1]:
            assert gdim == 1, "All dimensions but the last one must be 1"
        assert beta_shape[-1] == input_shape[-1], "Bias shape must be the same as normalized shape."
        for bdim in beta_shape[:-1]:
            assert bdim == 1, "All dimensions but the last one must be 1"

        # mean = dc.op("reduce_avg", (input_, ), (dim, ))
        mu = dc.op("reduce_sum", (input_, ), (dim, ))
        divider = dc.tensor(torch.zeros(input_shape) + 1.0 / input_shape[dim])
        mu = dc.op("multiply", (divider, mu), ())

        # x_minus_mean = dc.op("subtract", (input_, mean), ())
        xmu = dc.op("subtract", (input_, mu), ())
        # squared = dc.op("multiply", (x_minus_mean, x_minus_mean), ())
        sq = dc.op("multiply", (xmu, xmu), ())

        # var = dc.op("reduce_avg", (squared, ), (dim, ))
        var = dc.op("reduce_sum", (sq, ), (dim, ))
        divider = dc.tensor(torch.zeros(var.shape.as_list()) + 1.0 / input_shape[dim])
        var = dc.op("multiply", (divider, var), ())

        epsilon_tensor = dc.tensor(torch.zeros(var.shape.as_list()) + epsilon)
        # var_plus_eps = dc.op("add", (var, epsilon_tensor), ())
        var_add = dc.op("add", (var, epsilon_tensor), ())
        # std = dc.op("sqrt", (var_plus_eps, ), ())
        std = dc.op("sqrt", (var_add, ), ())
        # recip = dc.op("reciprocal", (std, ), ())
        ivar = dc.op(Reciprocal.create(), (std, ), ())
        # normalized = dc.op("multiply", (x_minus_mean, recip), ())
        xhat = dc.op("multiply", (xmu, ivar), ())
        # normalized_weighted = dc.op("multiply", (normalized, weights), ())
        xhat_weighted = dc.op("multiply", (xhat, weights), ())
        # result = dc.op("add", (normalized_weighted, bias), ())
        result = dc.op("add", (xhat_weighted, bias), ())
        dc.fuse(result)
        return

    if op_type == "layernorm_bw":
        
        # assert len(inputs) == 4, "Layernorm should have four operands."
        assert len(inputs) == 5, "Layernorm should have five operands."
        assert len(attr) == 3, "Layernorm should have three attributes."

        input_ = inputs[0]  # Input tensor
        gamma = inputs[1]   # gamma, weight re-scaling parameter
        beta = inputs[2]    # beta, weight re-centering parameter
        grad = inputs[3]    # gradient from the previous layer
        output = inputs[4]

        dim = attr[0]       # It should be the last dimension or last N dimensions ????
        epsilon = attr[1]
        operand = attr[2]   # Index of the operand, X - 0, weights - 1, bias - 2

        input_shape = input_.shape.as_list()
        gamma_shape = gamma.shape.as_list() 
        beta_shape = beta.shape.as_list()

        assert dim == -1 or dim == len(input_shape) - 1, "Normalization can be done only over the last dimension."
        assert gamma_shape[-1] == input_shape[-1], "Weights shape must be the same as normalized shape."
        for gdim in gamma_shape[:-1]:
            assert gdim == 1, "All dimensions but the last one must be 1"
        assert beta_shape[-1] == input_shape[-1], "Bias shape must be the same as normalized shape."
        for bdim in beta_shape[:-1]:
            assert bdim == 1, "All dimensions but the last one must be 1"

        if operand == 2:
            dbeta = dc.op("reduce_sum", (grad, ), (-2, ))
            # dbeta = dc.op("reduce_sum", (grad, ), (0, ))
            # grad_shape = grad.shape.as_list()
            # for i in range(1, len(grad_shape) - 1):
            #     dbeta = dc.op("reduce_sum", (dbeta, ), (i, ))
            dc.fuse(dbeta)
            return

        xhat_weighted, _ = dc.get_operands(output)
        xhat, _ = dc.get_operands(xhat_weighted)
        _, ivar =  dc.get_operands(xhat)

        if operand == 1:
            xhat_grad = dc.op("multiply", (xhat, grad), ())
            dgamma = dc.op("reduce_sum", (xhat_grad, ), (-2, ))
            # dgamma = dc.op("reduce_sum", (xhat_grad, ), (0, ))
            # xhat_grad_shape = xhat_grad.shape.as_list()
            # for i in range(1, len(xhat_grad_shape) - 1):
            #     dgamma = dc.op("reduce_sum", (dgamma, ), (i, ))
            dc.fuse(dgamma)
            return

        if operand == 0:
            dxhat = dc.op("multiply", (grad, gamma), ())
            N = input_shape[dim]
            sum_1 = dc.op("reduce_sum", (dxhat, ), (dim, ))
            dxhat_xhat = dc.op("multiply", (dxhat, xhat), ())
            sum_2 = dc.op("reduce_sum", (dxhat_xhat, ), (dim, ))
            xhat_sum_2 = dc.op("multiply", (xhat, sum_2), ())
            sum_1_sum_2_add = dc.op("add", (sum_1, xhat_sum_2), ())
            N_recip = torch.zeros(sum_1_sum_2_add.shape.as_list()) + 1.0 / N
            N_recip_tensor = dc.tensor(N_recip)
            N_recip_add = dc.op("multiply", (N_recip_tensor, sum_1_sum_2_add), ())
            dxhat_add_sub = dc.op("subtract", (dxhat, N_recip_add), ())
            dx = dc.op("multiply", (ivar, dxhat_add_sub), ())
            dc.fuse(dx)
            return

    assert False, f"{op_type} is not defined in nn decompose_post_autograd. "

def initial_flops_estimate(type, attr, ops):
    flops = 0
    if type == "layernorm":
        LAYER_NORM_FLOPS = 5
        flops = LAYER_NORM_FLOPS
        flops *= ops[0][-1] * ops[0][-2]
        if len(ops[0]) > 2:
            flops *= ops[0][-3]
        if len(ops[0]) > 3:
            flops *= ops[0][-4]

    return flops

