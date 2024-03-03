# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import importlib
from types import ModuleType
from functools import lru_cache
from .transpose import TransposeTM
from .splice import Splice
from .exp import Exp
from .sine import Sine
from .cosine import Cosine
from .ethernet_datacopy import EthernetDatacopy
from .reciprocal import Reciprocal
from .abs import Abs
from .tanh import Tanh
from .log import Log
from .nop import Nop
from .buffer import Buffer
from .sqrt import Sqrt
from .tilizer import Tilizer
from .clip import Clip

op_to_module_map = {
        "add":             "eltwise_binary",
        "subtract":        "eltwise_binary",
        "multiply":        "eltwise_binary",
        "maximum":         "eltwise_binary",
        "minimum":         "eltwise_binary",
        "heaviside":       "eltwise_binary",
        "binary_vstack":   "eltwise_binary",
        "binary_hstack":   "eltwise_binary",
        "greater":         "eltwise_binary",
        "greater_equal":   "eltwise_binary",
        "less":            "eltwise_binary",
        "less_equal":      "eltwise_binary",
        "equal":           "eltwise_binary",
        "not_equal":       "eltwise_binary",
        "ethernet_datacopy": EthernetDatacopy,
        "exp":             Exp,
        "nop":             Nop,
        "buffer":          Buffer,
        "reciprocal":      Reciprocal,
        "sqrt":             Sqrt,
        "lrelu":           "eltwise_unary",
        "gelu":            "eltwise_unary",
        "gelu_derivative": "eltwise_unary",
        "log":             Log,
        "sigmoid":         "eltwise_unary",
        "clip":            Clip,
        "reduce":          "eltwise_unary",
        "abs":             Abs,
        "tanh":            Tanh,
        "dropout":         "eltwise_unary",
        "sine":            Sine,
        "cosine":          Cosine,
        "power":           "eltwise_unary",
        "tilizer":         Tilizer,

        "conv_sum":        "eltwise_nary",
        "hconcat":         "eltwise_nary",
        "vconcat":         "eltwise_nary",
        "concatenate":     "eltwise_nary",
        "index_copy":      "eltwise_nary",
        "splice":          Splice,

        "matmul":        "matmul",

        "depthwise": "depthwise",

        "embedding":     "embedding",

        "transpose":                TransposeTM,
        "reshape":                  "tm",
        "select":                   "tm",
        "gather":                   "tm",
        "hslice":                   "tm",
        "hstack":                   "tm",
        "vslice":                   "tm",
        "vstack":                   "tm",
        "broadcast":                "tm",
        "conv2d_grouped_weights":   "tm",
        "conv2d_prestride_act":     "tm",
        "tile_broadcast":           "tm",
        "buda_pad":                 "tm",
        "buda_unpad":               "tm",


        "constant":  "constant",
        "dram_queue":  "dram_queue",

        "fused_op": "fused_ops",


        "quantization"            : "quantize",
        "dequantization"          : "quantize",
        "requantization"          : "quantize",

        "void": "void", # void op for testing purposes
}

def has_newstyle_interface(op_name):
    return type(op_to_module_map[op_name]) is not str

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
        exceptions_to_unary = ["clip", "reduce"]
        return "eltwise_unary" in module_name_or_cls and op_type.op not in exceptions_to_unary
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
    assert op_name in op_to_module_map, f"Buda op module not defined for {op_name}"
    module_name_or_cls = op_to_module_map[op_name]
    if type(module_name_or_cls) is str:
        return importlib.import_module("." + module_name_or_cls, package="pybuda.op.eval.buda")
    else:
        return module_name_or_cls

def get_f_instance(op_type):
    module_or_class = _get_module_or_class(op_type.op)
    assert not isinstance(module_or_class, ModuleType)
    return module_or_class(op_type)

def get_f_pybuda_shape(op_type, tile_height, tile_width):
    module_or_class = _get_module_or_class(op_type.op)
    if isinstance(module_or_class, ModuleType):
        return lambda *args: module_or_class.shape(op_type.op, op_type.attr, *args, tile_height, tile_width)
    else:
        return lambda *args: module_or_class(op_type).shape(*args, tile_height, tile_width)

def get_f_pybuda_eval(op_type):
    module_or_class = _get_module_or_class(op_type.op)
    if isinstance(module_or_class, ModuleType):
        return lambda *args: module_or_class.eval(op_type.op, op_type.attr, *args)
    else:
        return module_or_class(op_type).eval

def get_f_pybuda_parallelization(op_type):
    module_or_class = _get_module_or_class(op_type.op)
    if isinstance(module_or_class, ModuleType):
        return lambda *args: module_or_class.parallelization(op_type.op, op_type.attr, *args)
    else:
        return module_or_class(op_type).parallelization

def get_f_pybuda_input_ublock_order(op_type, num_operands):
    module_or_class = _get_module_or_class(op_type.op)
    if isinstance(module_or_class, ModuleType):
        return module_or_class.input_ublock_order(op_type.op, op_type.attr, num_operands)
    else:
        return module_or_class(op_type).input_ublock_order(num_operands)

def get_f_pybuda_execution_cycles(op_type):
    module_or_class = _get_module_or_class(op_type.op)
    if isinstance(module_or_class, ModuleType):
        return lambda *args: module_or_class.execution_cycles(op_type.op, *args)
    else:
        return module_or_class(op_type).execution_cycles
