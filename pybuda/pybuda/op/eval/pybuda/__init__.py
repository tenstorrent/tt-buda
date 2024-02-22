# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import importlib
from types import ModuleType
from functools import lru_cache
from .transpose import TransposeTM
from .exp import Exp
from .logical_not import LogicalNot
from .sine import Sine
from .cosine import Cosine
from .ethernet_datacopy import EthernetDatacopy
from .reciprocal import Reciprocal
from .abs import Abs
from .tanh import Tanh

op_to_module_map = {
        "add":           "eltwise_binary",
        "divide":        "eltwise_binary",
        "subtract":      "eltwise_binary",
        "multiply":      "eltwise_binary",
        "maximum":       "eltwise_binary",
        "minimum":       "eltwise_binary",
        "heaviside":     "eltwise_binary",
        "binary_stack":  "eltwise_binary",
        "power":         "eltwise_binary",
        "greater":       "eltwise_binary",
        "greater_equal": "eltwise_binary",
        "less":          "eltwise_binary",
        "less_equal":    "eltwise_binary",
        "equal":         "eltwise_binary",
        "not_equal":     "eltwise_binary",
        "logical_and":   "eltwise_binary",

        "nop":             "eltwise_unary",
        "buffer":          "eltwise_unary",
        "exp":              Exp,
        "reciprocal":      Reciprocal,
        "sqrt":            "eltwise_unary",
        "relu":            "eltwise_unary",
        "leaky_relu":      "eltwise_unary",
        "gelu":            "eltwise_unary",
        "gelu_derivative": "eltwise_unary",
        "log":             "eltwise_unary",
        "sigmoid":         "eltwise_unary",
        "clip":            "eltwise_unary",
        "sine":            Sine,
        "cosine":          Cosine,
        "abs":             Abs,
        "sine":            "eltwise_unary",
        "tile_broadcast":  "eltwise_unary",
        "argmax":          "eltwise_unary",
        "tanh":            Tanh,
        "cumsum":          "eltwise_unary",
        "logical_not":     LogicalNot,
        "dropout":         "eltwise_unary",
        "pow":             "eltwise_unary",
        "tilizer":          "eltwise_unary",
        "erf":             "eltwise_unary",

        "conv_sum":        "eltwise_nary",
        "concatenate":     "eltwise_nary",
        "where":           "eltwise_nary",
        "index_copy":      "eltwise_nary",
        "interleave":      "eltwise_nary",
        "stack":           "eltwise_nary",
        
        "matmul":        "matmul",
        "sparse_matmul": "matmul",

        "depthwise": "depthwise",

        "embedding": "embedding",
        "ethernet_datacopy": EthernetDatacopy,

        "transpose":                   TransposeTM,
        "adv_index":                   "tm",
        "reshape":                     "tm",
        "index":                       "tm",
        "select":                      "tm",
        "gather":                      "tm",
        "hslice":                      "tm",
        "hstack":                      "tm",
        "vslice":                      "tm",
        "vstack":                      "tm",
        "broadcast":                   "tm",
        "repeat":                      "tm",
        "repeat_dim":                  "tm",
        "conv2d_depthwise_weights":    "tm",
        "conv2d_depthwise_weights_bw": "tm",
        "conv2d_grouped_weights":      "tm",
        "conv2d_grouped_weights_bw":   "tm",
        "conv2d_prestride_act":        "tm",
        "conv2d_prestride_weights":    "tm",
        "pad_tile":                    "tm",
        "narrow":                      "tm",
        "pad":                         "tm",
        "unsqueeze":                   "tm",
        "squeeze":                     "tm",
        "pixel_shuffle":               "tm",
        "buda_pad":                 "tm",
        "buda_unpad":               "tm",

        "reduce_avg":         "reduce",
        "reduce_sum":         "reduce",
        "reduce_max":         "reduce",
        "grouped_reduce_avg": "reduce",

        "conv2d"            :        "convolution",
        "conv2d_transpose"  :        "convolution",
        "conv3d"            :        "convolution",

        "max_pool1d":    "pooling",
        "max_pool2d":    "pooling",
        "max_pool3d":    "pooling",
        "avg_pool1d":    "pooling",
        "avg_pool2d":    "pooling",

        "constant":  "constant",

        "resize2d":  "resize",
        "resize3d":  "resize",
        "dram_queue":  "dram_queue",

        "softmax": "nn",
        "log_softmax": "nn",
        "softmax_bw": "nn",
        "mask": "mask",
        "layernorm": "nn",
        "layernorm_bw": "nn",
        "batchnorm" : "nn", 

        "quantize"            : "quantize",
        "buda_quantize"       : "quantize",
        "dequantize"          : "quantize",
        "requantize"          : "quantize",
        "buda_requantize"     : "quantize",
        "buda_dequantize"     : "quantize",
}


def has_newstyle_interface(op_name):
    return type(op_to_module_map[op_name]) is not str

def is_tm(op_type):
    module_name_or_cls = op_to_module_map[op_type.op]
    if type(module_name_or_cls) is str:
        return module_name_or_cls == "tm"
    else:
        return module_name_or_cls(op_type).is_tm()

def is_eltwise(op_type):
    module_name_or_cls = op_to_module_map[op_type.op]
    if type(module_name_or_cls) is str:
        return "eltwise" in module_name_or_cls
    else:
        return module_name_or_cls(op_type).is_eltwise()

def is_eltwise_binary(op_type):
    module_name_or_cls = op_to_module_map[op_type.op]
    if type(module_name_or_cls) is str:
        return "eltwise_binary" == module_name_or_cls
    else:
        return module_name_or_cls(op_type).is_eltwise_binary()

def is_eltwise_unary(op_type):
    module_name_or_cls = op_to_module_map[op_type.op]
    if type(module_name_or_cls) is str:
        return "eltwise_unary" in module_name_or_cls
    else:
        return module_name_or_cls(op_type).is_eltwise_unary()

def is_eltwise_nary(op_type):
    module_name_or_cls = op_to_module_map[op_type.op]
    if type(module_name_or_cls) is str:
        return "eltwise_nary" in module_name_or_cls
    else:
        return module_name_or_cls(op_type).is_eltwise_nary()

@lru_cache(maxsize=len(op_to_module_map))
def _get_module_or_class(op_name):
    assert op_name in op_to_module_map, f"Pybuda op module not defined for {op_name}"
    module_name_or_cls = op_to_module_map[op_name]
    if type(module_name_or_cls) is str:
        return importlib.import_module("." + module_name_or_cls, package="pybuda.op.eval.pybuda")
    else:
        return module_name_or_cls

def get_f_instance(op_type):
    module_or_class = _get_module_or_class(op_type.op)
    assert not isinstance(module_or_class, ModuleType)
    return module_or_class(op_type)

def empty_function(*inputs):
    pass

def get_f_pybuda_backward(op_type):
    module_or_class = _get_module_or_class(op_type.op)
    if isinstance(module_or_class, ModuleType):
        return lambda *inputs : module_or_class.backward(op_type.op, op_type.attr, *inputs)
    else:
        return module_or_class(op_type).backward

def get_f_pybuda_shape(op_type):
    module_or_class = _get_module_or_class(op_type.op)
    if isinstance(module_or_class, ModuleType):
        return lambda *inputs : module_or_class.shape(op_type.op, op_type.attr, *inputs)
    else:
        return module_or_class(op_type).shape

def get_f_pybuda_eval(op_type):
    module_or_class = _get_module_or_class(op_type.op)
    if isinstance(module_or_class, ModuleType):
        return lambda *inputs : module_or_class.eval(op_type.op, op_type.attr, *inputs)
    else:
        return module_or_class(op_type).eval

def get_f_pybuda_lower(op_type):
    module_or_class = _get_module_or_class(op_type.op)
    if isinstance(module_or_class, ModuleType):
        if op_type.op == "matmul" or op_type.op == "sparse_matmul":
            return lambda *inputs : module_or_class.lower(op_type.op, op_type.attr, op_type.buda_attrs, *inputs)
        return lambda *inputs : module_or_class.lower(op_type.op, op_type.attr, *inputs)
    else:
        return module_or_class(op_type).lower

def get_f_pybuda_decompose(op_type):
    module_or_class = _get_module_or_class(op_type.op)
    if isinstance(module_or_class, ModuleType):
        if hasattr(module_or_class, "decompose"):
            return lambda *inputs : module_or_class.decompose(op_type.op, op_type.attr, *inputs)
        else:
            return empty_function
    else:
        return module_or_class(op_type).decompose

def get_f_pybuda_decompose_post_autograd(op_type):
    module_or_class = _get_module_or_class(op_type.op)
    if isinstance(module_or_class, ModuleType):
        if hasattr(module_or_class, "decompose_post_autograd"):
            return lambda *inputs : module_or_class.decompose_post_autograd(op_type.op, op_type.attr, *inputs)
        else:
            return empty_function
    else:
        return module_or_class(op_type).decompose_post_autograd

def get_f_pybuda_decompose_post_optimize(op_type):
    module_or_class = _get_module_or_class(op_type.op)
    if isinstance(module_or_class, ModuleType):
        if hasattr(module_or_class, "decompose_post_optimize"):
            return lambda *inputs : module_or_class.decompose_post_optimize(op_type.op, op_type.attr, *inputs)
        else:
            return empty_function
    else:
        return module_or_class(op_type).decompose_post_optimize

def get_f_pybuda_initial_flops_estimate(op_type):
    module_or_class = _get_module_or_class(op_type.op)
    if isinstance(module_or_class, ModuleType):
        if hasattr(module_or_class, "initial_flops_estimate"):
            return lambda *inputs : module_or_class.initial_flops_estimate(op_type.op, op_type.attr, *inputs)
        else:
            return empty_function
    else:
        return module_or_class(op_type).initial_flops_estimate
