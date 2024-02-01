# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import tensorflow as tf
import torch
import numpy as np

tf_to_pt_type_map = {
    tf.bfloat16: torch.bfloat16,
    tf.bool: torch.bool,
    tf.complex128: torch.complex128,
    tf.complex64: torch.complex64,
    tf.double: torch.double,
    tf.float16: torch.float16,
    tf.float32: torch.float32,
    tf.float64: torch.float64,
    tf.half: torch.half,
    tf.int16: torch.int16,
    tf.int32: torch.int32,
    tf.int64: torch.int64,
    tf.int8: torch.int8,
    tf.qint16: torch.qint32, # No torch.qint16.
    tf.qint32: torch.qint32,
    tf.qint8: torch.qint8,
    tf.quint16: None, # No torch.quint16.
    tf.quint8: torch.quint8,
    tf.resource: None, # No torch.resource.
    tf.string: None, # No torch.string
    tf.uint16: None, # No torch.uint16
    tf.uint32: None, # No torch.uint16
    tf.uint64: None, # No torch.uint16
    tf.uint8: torch.uint8,
    tf.variant: None # No torch.uint16
}


def map_tf_dtype_to_pt(tf_dtype):
    pt_type = tf_to_pt_type_map[tf_dtype]
    assert pt_type is not None, f"TensorFlow DType {tf_dtype} has no PyTorch equivalent"
    return pt_type

def map_pt_dtype_to_tf(pt_dtype):
    pt_types = list(tf_to_pt_type_map.values())
    assert pt_dtype in pt_types, f"{pt_dtype} Tensorflow equivelant not defined"

    return list(tf_to_pt_type_map.keys())[pt_types.index(pt_dtype)]

def flatten_inputs(inputs, names=None, force_float32=False):
    from pybuda.tensor import Tensor

    new_inputs = []
    new_names = []
    flattened_name_map = {}

    if isinstance(inputs, (torch.Tensor, tf.Tensor, Tensor)):
        inputs = (inputs, )

    if names is None:
        names = [f"input_{i}" for i in range(len(inputs))]

    assert len(inputs) == len(names)

    for i in range(len(inputs)):
        inp = inputs[i]
        name = names[i]
        if isinstance(inp, (list, tuple)):
            sub_names = [f"{name}_{j}" for j in range(len(inp))]
            
            sub_inputs, sub_names, _ = flatten_inputs(inp, sub_names)
            new_inputs += sub_inputs
            new_names += sub_names

            flattened_name_map[name] = sub_names

        elif isinstance(inp, dict):
            sub_names = []
            sub_inputs = []
            for k, v in inp.items():
                sub_names.append(f"{name}_{k}")
                sub_inputs.append(v)
            
            sub_inputs, sub_names, _ = flatten_inputs(sub_inputs, sub_names)
            new_inputs += sub_inputs
            new_names += sub_names
            flattened_name_map[name] = sub_names

        elif isinstance(inp, (torch.Tensor, tf.Tensor, Tensor)):
            new_inputs.append(inp)
            new_names.append(name)
            flattened_name_map[name] = [name]
        
        elif inp is None:
            continue

        else:
            raise NotImplementedError(f"Unknown input type: {type(inp)}")

    return new_inputs, new_names, flattened_name_map

def flatten_structured_output(outputs):
    from pybuda.tensor import Tensor

    new_outputs = []

    for i in range(len(outputs)):
        out = outputs[i]

        if isinstance(out, (list, tuple)):
            sub_output = flatten_structured_output(out,)
            new_outputs += sub_output

        elif isinstance(out, dict):
            sub_output = []
            for k, v in out.items():
                sub_output.append(v)

            sub_output= flatten_structured_output(sub_output,)
            new_outputs += sub_output

        elif isinstance(out, (torch.Tensor, tf.Tensor, Tensor, np.ndarray)):
            new_outputs.append(out)

        elif out is None:
            continue
        else:
            raise NotImplementedError(f"Unknown output type: {type(out)}")

    return new_outputs
