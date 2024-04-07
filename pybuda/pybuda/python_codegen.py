# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

import torch
import numpy as np
import tensorflow as tf
from loguru import logger

def pybuda_df_str_from_str(df: str, name: str): 
        df = df.lower()
        
        if df == "bfp2":
            return "pybuda.DataFormat.Bfp2"
        elif df == "bfp2_b":
            return "pybuda.DataFormat.Bfp2_b"
        elif df == "bfp4":
            return "pybuda.DataFormat.Bfp4"
        elif df == "bfp4_b":
            return "pybuda.DataFormat.Bfp4_b"
        elif df == "bfp8":
            return "pybuda.DataFormat.Bfp8"
        elif df == "bfp8_b":
            return "pybuda.DataFormat.Bfp8_b"
        elif df == "float16":
            return "pybuda.DataFormat.Float16"
        elif df in ["float16_b", "bfloat16"]:
            return "pybuda.DataFormat.Float16_b"
        elif df == "float32":
            return "pybuda.DataFormat.Float32"
        elif df == "int8":
            return "pybuda.DataFormat.Int8"
        elif df == "invalid":
            return "pybuda.DataFormat.Invalid"
        elif df == "lf8":
            return "pybuda.DataFormat.Lf8"
        elif df == "raw_uint16":
            return "pybuda.DataFormat.RawUInt16"
        elif df == "raw_uint32":
            return "pybuda.DataFormat.RawUInt32"
        elif df == "raw_uint8":
            return "pybuda.DataFormat.RawUInt8"
        elif df == "uint16":
            return "pybuda.DataFormat.UInt16"
        elif df == "uint8":
            return "pybuda.DataFormat.UInt8"
        elif df == "int8":
            return "pybuda.DataFormat.Int8"
        elif df == "int32":
            return "pybuda.DataFormat.Int32"
        else:
            logger.warning(f"Invalid data format: {df} for constant/parameter \'{name}\', defaulting to float32")
            return "pybuda.DataFormat.Float32"
        
def pytorch_df_str_from_str(df: str, name): 
        df = df.lower()
        
        if df == "float16":
            return "torch.float16"
        elif df in ["float16_b", "bfloat16"]:
            return "torch.bfloat16"
        elif df == "float32":
            return "torch.float32"
        elif df == "float64":
            return "torch.float64"
        elif df == "uint8":
            return "torch.uint8"
        elif df == "int8":
            return "torch.int8"
        elif df == "int32":
            return "torch.int32"
        elif df == "int16":
            return "torch.int16"
        elif df == "int64":
            return "torch.int64"
        else:
            logger.warning(f"Invalid data format: {df} for constant/parameter \'{name}\', defaulting to float32")
            return "torch.float32"


class PythonWriter():
    def __init__(self, module_name, open_file=True):
        self.filename = module_name + ".py"

        self.module_directory = "generated_modules"
        os.makedirs(self.module_directory, exist_ok=True)
        if open_file:
            self.file = open(os.path.join(self.module_directory, self.filename), "w")
        self.indent = 0
        self.module_name = module_name
        self.class_name = module_name.title().replace("_", "")

    def wl(self, text):
        indent = self.indent * "    "
        self.file.write(indent + text + "\n")

    def breakpoint(self):
        self.wl("import pdb; pdb.set_trace()")

    def close_file(self):
        self.file.close()
    
    def import_module_path(self):
        return self.module_directory + f".{self.module_name}"

class PyBudaWriter(PythonWriter):
    incompatible_np_float_types = [tf.bfloat16, ]

    def __init__(self, module_name, framework, contains_incompatible_np_floats=False, delete_inputs=True):
        super().__init__(module_name)

        self.framework = framework
        self.param_names = []
        self.const_names = []
        self.num_submodels = 0
        self.contains_incompatible_np_floats = contains_incompatible_np_floats
        self.delete_inputs = delete_inputs
        self.dev = "TTDevice"

    def write_header(self):
        self.wl("import pybuda")
        self.wl("import pybuda.op")
        self.wl("from pybuda import PyBudaModule")
        self.wl("")

        self.wl("from loguru import logger")

        self.wl("import torch")
        if self.framework == "tensorflow":
            self.wl("import tensorflow as tf")
            self.wl("from pybuda.tvm_utils import map_tf_dtype_to_pt")

        if self.framework == "jax":
            self.wl("import flax")
            self.wl("import numpy as np")
            self.wl("import jax.numpy as jnp")
            self.wl("from collections.abc import MutableMapping")
            self.wl("from transformers.modeling_flax_utils import FlaxPreTrainedModel")

        self.wl("\n")

    def write_class_definition(self, params, constants, class_name=None, num_submodels=0, is_submodel=False):
        if class_name is None:
            class_name = self.class_name
        self.num_submodels = num_submodels
        self.wl(f"class {class_name}(PyBudaModule):")
        self.indent += 1
        self.wl("def __init__(self, name):")
        self.indent += 1
        self.wl(f"super().__init__(name)")
        if num_submodels > 0:
            self.wl(f"self.num_layers = {num_submodels}")
            self.wl("self.layers = []")
            self.wl("for i in range(self.num_layers):")
            self.indent += 1
            self.wl("self.layers.append(Submodel(f\"layer_{i}\"))")
            self.indent -= 1

        for param in params.values():
            name, shape, requires_grad, dtype = param
            if name in self.param_names:
                continue
            self.param_names.append(name)
            if is_submodel:
                self.wl(f"self.add_parameter(\"{name}\", pybuda.Parameter(*{shape}, requires_grad={requires_grad}, dev_data_format={pybuda_df_str_from_str(dtype, name)}), prepend_name=True)")
            else:
                self.wl(f"self.add_parameter(\"{name}\", pybuda.Parameter(*{shape}, requires_grad={requires_grad}, dev_data_format={pybuda_df_str_from_str(dtype, name)}))")


        for const in constants.values():
            name = const[0]
            shape = tuple(const[1])
            self.const_names.append(name)
            if is_submodel:
                self.wl(f"self.add_constant(\"{name}\", prepend_name=True, shape={shape})")
            else:
                self.wl(f"self.add_constant(\"{name}\", shape={shape})")

        self.indent = 0
        self.wl("")

    def get_op_input_names(self, op):
        input_names = []
        for name in op.input_names:
            if name in self.param_names:
                input_names.append("self.get_parameter(\"" + name + "\")")
            elif name in self.const_names:
                input_names.append("self.get_constant(\"" + name + "\")")
            else:
                input_names.append(name)

        return input_names


    def write_forward(self, ops, inputs, outputs):
        self.indent = 1
        activation_names = "".join([", " + name for name in [inputs[key] for key in sorted(inputs)]])
        self.wl("def forward(self" + activation_names + "):")
        self.indent += 1

        for key in sorted(ops):
            input_names = self.get_op_input_names(ops[key])
            activation_names = "".join([", " + name for name in input_names])
            if ops[key].is_submodule_call:
                activation_names = activation_names.lstrip(", ")
            if len(ops[key].args) == 0:
                arg_text = ""
            else:
                arg_text = "".join(
                    [
                        ", " + argument + "=" + value for argument, value in ops[key].args                    
                    ]
                )

            set_src_layer = ""
            if ops[key].src_layer:
                set_src_layer = f'.set_src_layer("{ops[key].src_layer}")'
            if ops[key].is_submodule_call:
                if len(ops[key].loop_with):
                    if len(ops[key].loop_with) + 1 == self.num_submodels:
                        loop_len = "self.num_layers"
                    else:
                        if ops[key].loop_start_index == 0:
                            loop_len = f"{len(ops[key].loop_with) + 1}"
                        else:
                            loop_len = f"{ops[key].loop_start_index}, {len(ops[key].loop_with) + ops[key].loop_start_index + 1}"
                        
                    self.wl(f"for i in range({loop_len}):") # +1 for current op
                    self.indent += 1
                    self.wl(f"{ops[key].output_name} = {ops[key].function_name}({activation_names}{arg_text}){set_src_layer}")
                    self.indent -= 1
                else:
                    self.wl(f"{ops[key].output_name} = {ops[key].function_name}({activation_names}{arg_text}){set_src_layer}")
            else:
                self.wl(f"{ops[key].output_name} = {ops[key].function_name}(\"{ops[key].node_name}\"{activation_names}{arg_text}){set_src_layer}")
                for name_to_del in ops[key].inputs_to_delete:
                    self.wl(f"{name_to_del}._value = None")

        outputs = list(outputs.values())
        if len(outputs) == 1:
            output_names = outputs[0]
        else:
            output_names = ", ".join(outputs)

        self.wl(f"return {output_names}")
        self.indent = 0
        self.wl("")


    def write_param_parser(self, param_names, param_file_name):
        self.indent = 1

        if self.framework == "pytorch":
            self.wl(f"def process_framework_parameters(self, model):")
            self.indent += 1
            self.wl(f"named_parameters = dict(model.state_dict().items())")
            if param_file_name is not None:
                self.wl(f"serialized_params = torch.load(\"{param_file_name}\")")
                self.wl(f"named_parameters.update(serialized_params)")
            self.wl("named_buffers = dict(model.named_buffers())")
            self.wl("named_parameters.update(named_buffers)")

            if len(param_names):
                self.wl("flattened_to_hierarchical_map = {")
                self.indent += 1
                for k, v in param_names.items():
                    self.wl(f"'{k}' : {v},")
                self.indent -= 1
                self.wl("}")

            # Loop over all named params
            self.wl("for name, torch_param in named_parameters.items():")
            self.indent += 1
            
            # Handle -inf and inf values
            self.wl("# Replace infinities with relevant numbers")
            self.wl("if torch.any(torch.isinf(torch_param)):")
            self.indent += 1
            self.wl("torch_param = torch.where(torch.isposinf(torch_param), torch.tensor(1e4, dtype=torch_param.dtype), torch_param)")
            self.wl("torch_param = torch.where(torch.isneginf(torch_param), torch.tensor(-1e4, dtype=torch_param.dtype), torch_param)")
            self.wl("logger.warning(f\"Replacing -inf and inf values in tensor param: {name}\")")
            self.indent -= 1
            
            self.wl("tensor = torch_param.data")

            if len(param_names):
                self.wl("if torch.numel(tensor) == 1 and len(tensor.shape) == 0:")
                self.indent += 1
                self.wl("tensor = tensor.reshape((1, 1))")
                self.indent -= 1
                self.wl("if name in flattened_to_hierarchical_map:")
                self.indent += 1
                self.wl("layer_name, param_name = flattened_to_hierarchical_map[name]")

                # If name in parameter dictionary
                self.wl("if param_name in self.get_submodules()[layer_name]._parameters:")
                self.indent += 1
                self.wl("tensor.requires_grad = True")
                self.wl("self.get_submodules()[layer_name].set_parameter(param_name, tensor)")
                self.indent -= 1

                # If name in constant dictionary
                self.wl("elif param_name in self.get_submodules()[layer_name]._constants:")
                self.indent += 1
                self.wl("if torch.numel(tensor) == 1 and len(tensor.shape) == 0:")
                self.indent += 1
                self.wl("tensor = tensor.reshape((1, 1))")
                self.indent -= 1
                self.wl("tensor.requires_grad = False")
                self.wl("self.get_submodules()[layer_name].set_constant(param_name, tensor)")
                self.indent -= 1

                self.wl("else:")
                self.indent += 1
                self.wl("logger.warning(f\"{name} not found in self._parameters and self._constants\")")
                self.indent -= 1

                self.indent -= 1
                self.wl("else:")
                self.indent += 1
            self.wl("if name in self._parameters:")
            self.indent += 1
            self.wl("tensor.requires_grad = torch.is_floating_point(tensor)")
            self.wl("self.set_parameter(name, tensor)")
            self.indent -= 1

            self.wl("elif name in self._constants:")
            self.indent += 1
            self.wl("if torch.numel(tensor) == 1 and len(tensor.shape) == 0:")
            self.indent += 1
            self.wl("tensor = tensor.reshape((1, 1))")
            self.indent -= 1
            self.wl("tensor.requires_grad = False")
            self.wl("if not torch.is_floating_point(tensor):")
            self.indent += 1
            self.wl("tensor = tensor.float()")
            self.indent -= 1
            self.wl("self.set_constant(name, tensor)")
            self.indent -= 1

            self.wl("else:")
            self.indent += 1
            self.wl("logger.warning(f\"{name} not found in self._parameters and self._constants\")")
            self.indent -= 1
            self.indent -= 1

            self.indent = 0

        elif self.framework == "tensorflow":
            self.wl(f"def process_framework_parameters(self, model):")
            self.indent += 1
            self.wl(f"weights = model.weights")
            self.wl("flattened_to_hierarchical_map = {")
            self.indent += 1
            for k, v in param_names.items():
                self.wl(f"'{k}' : {v},")
            self.indent -= 1
            self.wl("}")

            if self.contains_incompatible_np_floats:
                self.wl(f"incompatible_np_float_types = {PyBudaWriter.incompatible_np_float_types}")

            self.wl("for weight in weights:")
            self.indent += 1
            self.wl("name = weight.name")
            if self.contains_incompatible_np_floats:
                self.wl("# Some floating-point weights in the model havea a dtype that is incompatible with the .numpy() call.")
                self.wl("if weight.dtype in incompatible_np_float_types:")
                self.indent += 1
                self.wl("dtype = map_tf_dtype_to_pt(weight.dtype)")
                self.wl("weight = tf.cast(weight.value(), tf.float32)")
                self.wl("tensor = torch.tensor(weight.numpy(), dtype=dtype)")
                self.indent -= 1
                self.wl("else:")
                self.indent +=1
                self.wl("tensor = torch.tensor(weight.numpy())")
                self.indent -= 1
            else:
                self.wl("tensor = torch.tensor(weight.numpy())")
            
            self.wl("if name in flattened_to_hierarchical_map:")
            self.indent += 1
            self.wl("layer_name, param_name = flattened_to_hierarchical_map[name]")

            # If name in parameter dictionary
            self.wl("if param_name in self.get_submodules()[layer_name]._parameters:")
            self.indent += 1
            self.wl("tensor.requires_grad = True")
            self.wl("self.get_submodules()[layer_name].set_parameter(param_name, tensor)")
            self.indent -= 1

            self.wl("elif param_name in self.get_submodules()[layer_name]._constants:")
            self.indent += 1
            self.wl("if torch.numel(tensor) == 1 and len(tensor.shape) == 0:")
            self.indent += 1
            self.wl("tensor = tensor.reshape((1, 1))")
            self.indent -= 1
            self.wl("tensor.requires_grad = False")
            self.wl("self.get_submodules()[layer_name].set_constant(param_name, tensor)")
            self.indent -= 1

            self.wl("else:")
            self.indent += 1
            self.wl("logger.warning(f\"{name} not found in self._parameters and self._constants\")")
            self.indent -= 1
            self.indent -= 1

            self.wl("else:")
            self.indent += 1
            self.wl("if name in self._parameters:")
            self.indent += 1
            self.wl("tensor.requires_grad = True")
            self.wl("self.set_parameter(name, tensor)")
            self.indent -= 1

            self.wl("elif name in self._constants:")
            self.indent += 1
            self.wl("if torch.numel(tensor) == 1 and len(tensor.shape) == 0:")
            self.indent += 1
            self.wl("tensor = tensor.reshape((1, 1))")
            self.indent -= 1
            self.wl("tensor.requires_grad = False")
            self.wl("self.set_constant(name, tensor)")
            self.indent -= 1

            self.wl("else:")
            self.indent += 1
            self.wl("logger.warning(f\"{name} not found in self._parameters and self._constants\")")
            self.indent -= 1
            self.indent -= 1
            self.indent -= 1

            if param_file_name is not None:
                self.wl(f"serialized_params = torch.load(\"{param_file_name}\")")
                self.wl(f"for name, torch_param in serialized_params.items():")
                self.indent += 1
                self.wl("tensor = torch_param.data")
                self.wl("if name in flattened_to_hierarchical_map:")
                self.indent += 1
                self.wl("layer_name, param_name = flattened_to_hierarchical_map[name]")
                self.wl("if param_name in self.get_submodules()[layer_name]._parameters:")
                self.indent += 1
                self.wl("tensor.requires_grad = True")
                self.wl("self.get_submodules()[layer_name].set_parameter(param_name, tensor)")
                self.indent -= 1

                self.wl("elif param_name in self.get_submodules()[layer_name]._constants:")
                self.indent += 1
                self.wl("if torch.numel(tensor) == 1 and len(tensor.shape) == 0:")
                self.indent += 1
                self.wl("tensor = tensor.reshape((1, 1))")
                self.indent -= 1
                self.wl("tensor.requires_grad = False")
                self.wl("self.get_submodules()[layer_name].set_constant(param_name, tensor)")
                self.indent -= 1

                self.wl("else:")
                self.indent += 1
                self.wl("logger.warning(f\"{name} not found in self._parameters and self._constants\")")
                self.indent -= 1
                self.indent -= 1

                self.wl("else:")
                self.indent += 1
                self.wl("if name in self._parameters:")
                self.indent += 1
                self.wl("tensor.requires_grad = True")
                self.wl("self.set_parameter(name, tensor)")
                self.indent -= 1

                self.wl("elif name in self._constants:")
                self.indent += 1
                self.wl("if torch.numel(tensor) == 1 and len(tensor.shape) == 0:")
                self.indent += 1
                self.wl("tensor = tensor.reshape((1, 1))")
                self.indent -= 1
                self.wl("tensor.requires_grad = False")
                self.wl("self.set_constant(name, tensor)")
                self.indent -= 1

                self.wl("else:")
                self.indent += 1
                self.wl("logger.warning(f\"{name} not found in self._parameters and self._constants\")")
                self.indent -= 1
                self.indent -= 1
                self.indent -= 1
        elif self.framework == "tf_graphdef":
            self.wl(f"def process_framework_parameters(self, model):")
            self.indent += 1
            self.wl("flattened_to_hierarchical_map = {")
            self.indent += 1
            for k, v in param_names.items():
                self.wl(f"'{k}' : {v},")
            self.indent -= 1
            self.wl("}")

            self.wl(f"serialized_params = torch.load(\"{param_file_name}\")")
            self.wl(f"for name, torch_param in serialized_params.items():")
            self.indent += 1
            self.wl("tensor = torch_param.data")
            self.wl("if name in flattened_to_hierarchical_map:")
            self.indent += 1
            self.wl("layer_name, param_name = flattened_to_hierarchical_map[name]")
            self.wl("if param_name in self.get_submodules()[layer_name]._parameters:")
            self.indent += 1
            self.wl("tensor.requires_grad = True")
            self.wl("self.get_submodules()[layer_name].set_parameter(param_name, tensor)")
            self.indent -= 1

            self.wl("elif param_name in self.get_submodules()[layer_name]._constants:")
            self.indent += 1
            self.wl("if torch.numel(tensor) == 1 and len(tensor.shape) == 0:")
            self.indent += 1
            self.wl("tensor = tensor.reshape((1, 1))")
            self.indent -= 1
            self.wl("tensor.requires_grad = False")
            self.wl("self.get_submodules()[layer_name].set_constant(param_name, tensor)")
            self.indent -= 1

            self.wl("else:")
            self.indent += 1
            self.wl("logger.warning(f\"{name} not found in self._parameters and self._constants\")")
            self.indent -= 1
            self.indent -= 1

            self.wl("else:")
            self.indent += 1
            self.wl("if name in self._parameters:")
            self.indent += 1
            self.wl("tensor.requires_grad = True")
            self.wl("self.set_parameter(name, tensor)")
            self.indent -= 1

            self.wl("elif name in self._constants:")
            self.indent += 1
            self.wl("if torch.numel(tensor) == 1 and len(tensor.shape) == 0:")
            self.indent += 1
            self.wl("tensor = tensor.reshape((1, 1))")
            self.indent -= 1
            self.wl("tensor.requires_grad = False")
            self.wl("self.set_constant(name, tensor)")
            self.indent -= 1

            self.wl("else:")
            self.indent += 1
            self.wl("logger.warning(f\"{name} not found in self._parameters and self._constants\")")
            self.indent -= 1
            self.indent -= 1
            self.indent -= 1

        elif self.framework == "onnx":
            self.wl(f"def process_framework_parameters(self, model):")
            self.indent += 1
            self.wl(f"import onnx")
            self.wl(f"import onnx.numpy_helper")
            self.wl(f"import numpy as np")
            self.wl(f"weights = model.graph.initializer")
            self.wl("flattened_to_hierarchical_map = {")
            self.indent += 1
            for k, v in param_names.items():
                self.wl(f"'{k}' : {v},")
            self.indent -= 1
            self.wl("}")

            # Onnx will convert bfloat16 to float32 in numpy with numpy_helper call
            self.wl("# Onnx will convert bfloat16 to float32 in numpy with numpy_helper call")

            self.wl("for weight in weights:")
            self.indent += 1
            self.wl("name = weight.name")
            self.wl("weight_numpy = onnx.numpy_helper.to_array(weight)")


            self.wl("tensor = torch.tensor(weight_numpy)")
            
            self.wl("if name in flattened_to_hierarchical_map:")
            self.indent += 1
            self.wl("layer_name, param_name = flattened_to_hierarchical_map[name]")
            self.wl("if param_name in self.get_submodules()[layer_name]._parameters:")
            self.indent += 1
            self.wl("tensor.requires_grad = True")
            self.wl("self.get_submodules()[layer_name].set_parameter(param_name, tensor)")
            self.indent -= 1

            self.wl("elif param_name in self.get_submodules()[layer_name]._constants:")
            self.indent += 1
            self.wl("if torch.numel(tensor) == 1 and len(tensor.shape) == 0:")
            self.indent += 1
            self.wl("tensor = tensor.reshape((1, 1))")
            self.indent -= 1
            self.wl("tensor.requires_grad = False")
            self.wl("self.get_submodules()[layer_name].set_constant(param_name, tensor)")
            self.indent -= 1

            self.wl("else:")
            self.indent += 1
            self.wl("logger.warning(f\"{name} not found in self._parameters and self._constants\")")
            self.indent -= 1
            self.indent -= 1

            self.wl("else:")
            self.indent += 1
            self.wl("if name in self._parameters:")
            self.indent += 1
            self.wl("tensor.requires_grad = True")
            self.wl("self.set_parameter(name, tensor)")
            self.indent -= 1

            self.wl("elif name in self._constants:")
            self.indent += 1
            self.wl("if torch.numel(tensor) == 1 and len(tensor.shape) == 0:")
            self.indent += 1
            self.wl("tensor = tensor.reshape((1, 1))")
            self.indent -= 1
            self.wl("tensor.requires_grad = False")
            self.wl("self.set_constant(name, tensor)")
            self.indent -= 1

            self.wl("else:")
            self.indent += 1
            self.wl("logger.warning(f\"{name} not found in self._parameters and self._constants\")")
            self.indent -= 1
            self.indent -= 1
            self.indent -= 1
            if param_file_name is not None:
                self.wl(f"serialized_params = torch.load(\"{param_file_name}\")")
                self.wl(f"for name, torch_param in serialized_params.items():")
                self.indent += 1
                self.wl("tensor = torch_param.data")

                self.wl("if name in flattened_to_hierarchical_map:")
                self.indent += 1
                self.wl("layer_name, param_name = flattened_to_hierarchical_map[name]")
                self.wl("if param_name in self.get_submodules()[layer_name]._parameters:")
                self.indent += 1
                self.wl("tensor.requires_grad = True")
                self.wl("self.get_submodules()[layer_name].set_parameter(param_name, tensor)")
                self.indent -= 1

                self.wl("elif param_name in self.get_submodules()[layer_name]._constants:")
                self.indent += 1
                self.wl("if torch.numel(tensor) == 1 and len(tensor.shape) == 0:")
                self.indent += 1
                self.wl("tensor = tensor.reshape((1, 1))")
                self.indent -= 1
                self.wl("tensor.requires_grad = False")
                self.wl("self.get_submodules()[layer_name].set_constant(param_name, tensor)")
                self.indent -= 1

                self.wl("else:")
                self.indent += 1
                self.wl("logger.warning(f\"{name} not found in self._parameters and self._constants\")")
                self.indent -= 1
                self.indent -= 1

                self.wl("else:")
                self.indent += 1
                self.wl("if name in self._parameters:")
                self.indent += 1
                self.wl("tensor.requires_grad = True")
                self.wl("self.set_parameter(name, tensor)")
                self.indent -= 1

                self.wl("elif name in self._constants:")
                self.indent += 1
                self.wl("if torch.numel(tensor) == 1 and len(tensor.shape) == 0:")
                self.indent += 1
                self.wl("tensor = tensor.reshape((1, 1))")
                self.indent -= 1
                self.wl("tensor.requires_grad = False")
                self.wl("self.set_constant(name, tensor)")
                self.indent -= 1

                self.wl("else:")
                self.indent += 1
                self.wl("logger.warning(f\"{name} not found in self._parameters and self._constants\")")
                self.indent -= 1
                self.indent -= 1
                self.indent -= 1

        elif self.framework == "mxnet":
            self.wl(f"def process_framework_parameters(self, model):")
            self.indent += 1
            self.wl(f"import mxnet as mx")
            self.wl(f"import numpy as np")
            self.wl(f"weights = model.collect_params()")
            self.wl("flattened_to_hierarchical_map = {")
            self.indent += 1
            for k, v in param_names.items():
                self.wl(f"'{k}' : {v},")
            self.indent -= 1
            self.wl("}")

            # MXNet only has float16; conversion to numpy is handled by .asnumpy() call
            self.wl("# MXNet only has float16; conversion to numpy is handled by .asnumpy() call")

            self.wl("for name, weight in weights.items():")
            self.indent += 1
            self.wl("weight_numpy = weight.data().asnumpy()")


            self.wl("tensor = torch.tensor(weight_numpy)")
            
            self.wl("if name in flattened_to_hierarchical_map:")
            self.indent += 1
            self.wl("layer_name, param_name = flattened_to_hierarchical_map[name]")
            self.wl("if param_name in self.get_submodules()[layer_name]._parameters:")
            self.indent += 1
            self.wl("tensor.requires_grad = True")
            self.wl("self.get_submodules()[layer_name].set_parameter(param_name, tensor)")
            self.indent -= 1

            self.wl("elif param_name in self.get_submodules()[layer_name]._constants:")
            self.indent += 1
            self.wl("if torch.numel(tensor) == 1 and len(tensor.shape) == 0:")
            self.indent += 1
            self.wl("tensor = tensor.reshape((1, 1))")
            self.indent -= 1
            self.wl("tensor.requires_grad = False")
            self.wl("self.get_submodules()[layer_name].set_constant(param_name, tensor)")
            self.indent -= 1

            self.wl("else:")
            self.indent += 1
            self.wl("logger.warning(f\"{name} not found in self._parameters and self._constants\")")
            self.indent -= 1
            self.indent -= 1

            self.wl("else:")
            self.indent += 1
            self.wl("if name in self._parameters:")
            self.indent += 1
            self.wl("tensor.requires_grad = True")
            self.wl("self.set_parameter(name, tensor)")
            self.indent -= 1

            self.wl("elif name in self._constants:")
            self.indent += 1
            self.wl("if torch.numel(tensor) == 1 and len(tensor.shape) == 0:")
            self.indent += 1
            self.wl("tensor = tensor.reshape((1, 1))")
            self.indent -= 1
            self.wl("tensor.requires_grad = False")
            self.wl("self.set_constant(name, tensor)")
            self.indent -= 1

            self.wl("else:")
            self.indent += 1
            self.wl("logger.warning(f\"{name} not found in self._parameters and self._constants\")")
            self.indent -= 1
            self.indent -= 1
            self.indent -= 1
            if param_file_name is not None:
                self.wl(f"serialized_params = torch.load(\"{param_file_name}\")")
                self.wl(f"for name, torch_param in serialized_params.items():")
                self.indent += 1
                self.wl("tensor = torch_param.data")

                self.wl("if name in flattened_to_hierarchical_map:")
                self.indent += 1
                self.wl("layer_name, param_name = flattened_to_hierarchical_map[name]")
                self.wl("if param_name in self.get_submodules()[layer_name]._parameters:")
                self.indent += 1
                self.wl("tensor.requires_grad = True")
                self.wl("self.get_submodules()[layer_name].set_parameter(param_name, tensor)")
                self.indent -= 1

                self.wl("elif param_name in self.get_submodules()[layer_name]._constants:")
                self.indent += 1
                self.wl("if torch.numel(tensor) == 1 and len(tensor.shape) == 0:")
                self.indent += 1
                self.wl("tensor = tensor.reshape((1, 1))")
                self.indent -= 1
                self.wl("tensor.requires_grad = False")
                self.wl("self.get_submodules()[layer_name].set_constant(param_name, tensor)")
                self.indent -= 1

                self.wl("else:")
                self.indent += 1
                self.wl("logger.warning(f\"{name} not found in self._parameters and self._constants\")")
                self.indent -= 1
                self.indent -= 1

                self.wl("else:")
                self.indent += 1
                self.wl("if name in self._parameters:")
                self.indent += 1
                self.wl("tensor.requires_grad = True")
                self.wl("self.set_parameter(name, tensor)")
                self.indent -= 1

                self.wl("elif name in self._constants:")
                self.indent += 1
                self.wl("if torch.numel(tensor) == 1 and len(tensor.shape) == 0:")
                self.indent += 1
                self.wl("tensor = tensor.reshape((1, 1))")
                self.indent -= 1
                self.wl("tensor.requires_grad = False")
                self.wl("self.set_constant(name, tensor)")
                self.indent -= 1

                self.wl("else:")
                self.indent += 1
                self.wl("logger.warning(f\"{name} not found in self._parameters and self._constants\")")
                self.indent -= 1
                self.indent -= 1
                self.indent -= 1
        elif self.framework == "jax":
            self.wl(f"def process_framework_parameters(self, model):")
            self.indent += 1
            self.wl(f"def flatten_params(params, parent_key='', sep='.'):")
            self.indent += 1
            self.wl(f"items = []")
            self.wl(f"for key, val in params.items():")
            self.indent += 1
            self.wl(f"new_key = parent_key + sep + key if parent_key else key")
            self.wl(f"if isinstance(val, MutableMapping):")
            self.indent += 1
            self.wl(f"items.extend(flatten_params(val, new_key, sep=sep).items())")
            self.indent -= 1
            self.wl(f"else:")
            self.indent += 1
            self.wl(f"items.append((new_key, val))")
            self.indent -= 1
            self.indent -= 1
            self.wl(f"return dict(items)")
            self.indent -= 1
            self.wl(f"if isinstance(model, FlaxPreTrainedModel):")
            self.indent += 1
            self.wl(f"model_params = model.params")
            self.indent -= 1
            self.wl(f"elif isinstance(model, flax.linen.Module):")
            self.indent += 1
            self.wl("model_params = {}")
            self.wl("if hasattr(model, 'params'):")
            self.indent += 1
            self.wl(f"model_params = model.variables['params']._dict")
            self.indent -= 1
            self.indent -= 1
            if param_file_name is not None:
                self.wl(f"serialized_params = torch.load(\"{param_file_name}\")")
                self.wl(f"for key, val in serialized_params.items():")
                self.indent += 1
                self.wl(f"model_params[key] = jnp.array(val.data.numpy())")
                self.indent -= 1
            self.wl(f"model_params = flatten_params(model_params)")
            self.wl(f"for key, value in model_params.items():")
            self.indent += 1
            self.wl(f"name = key")
            self.wl(f"tensor = torch.Tensor(np.array(value))")

            self.wl(f"if name in self._parameters:")
            self.indent += 1
            self.wl(f"tensor.requires_grad = True")
            self.wl(f"self.set_parameter(name, tensor)")
            self.indent -= 1
            self.wl("elif name in self._constants:")
            self.indent += 1
            self.wl("if torch.numel(tensor) == 1 and len(tensor.shape) == 0:")
            self.indent += 1
            self.wl("tensor = tensor.reshape((1, 1))")
            self.indent -= 1
            self.wl("tensor.requires_grad = False")
            self.wl("self.set_constant(name, tensor)")
            self.indent -= 1
            self.wl(f"else:")
            self.indent += 1
            self.wl("logger.warning(f\"{name} not found in self._parameters and self._constants\")")
            self.indent -= 1
            self.indent -= 1
            self.indent -= 1            
        elif self.framework == "tflite":
            self.wl(f"def process_framework_parameters(self, model):")
            self.indent += 1
            self.wl("flattened_to_hierarchical_map = {")
            self.indent += 1
            for k, v in param_names.items():
                self.wl(f"'{k}' : {v},")
            self.indent -= 1
            self.wl("}")

            self.wl(f"serialized_params = torch.load(\"{param_file_name}\")")
            self.wl(f"for name, torch_param in serialized_params.items():")
            self.indent += 1
            self.wl("tensor = torch_param.data")
            self.wl("if name in flattened_to_hierarchical_map:")
            self.indent += 1
            self.wl("layer_name, param_name = flattened_to_hierarchical_map[name]")
            self.wl("if param_name in self.get_submodules()[layer_name]._parameters:")
            self.indent += 1
            self.wl("tensor.requires_grad = True")
            self.wl("self.get_submodules()[layer_name].set_parameter(param_name, tensor)")
            self.indent -= 1

            self.wl("elif param_name in self.get_submodules()[layer_name]._constants:")
            self.indent += 1
            self.wl("if torch.numel(tensor) == 1 and len(tensor.shape) == 0:")
            self.indent += 1
            self.wl("tensor = tensor.reshape((1, 1))")
            self.indent -= 1
            self.wl("tensor.requires_grad = False")
            self.wl("self.get_submodules()[layer_name].set_constant(param_name, tensor)")
            self.indent -= 1

            self.wl("else:")
            self.indent += 1
            self.wl("logger.warning(f\"{name} not found in self._parameters and self._constants\")")
            self.indent -= 1
            self.indent -= 1

            self.wl("else:")
            self.indent += 1
            self.wl("if name in self._parameters:")
            self.indent += 1
            self.wl("tensor.requires_grad = True")
            self.wl("self.set_parameter(name, tensor)")
            self.indent -= 1

            self.wl("elif name in self._constants:")
            self.indent += 1
            self.wl("if torch.numel(tensor) == 1 and len(tensor.shape) == 0:")
            self.indent += 1
            self.wl("tensor = tensor.reshape((1, 1))")
            self.indent -= 1
            self.wl("tensor.requires_grad = False")
            self.wl("self.set_constant(name, tensor)")
            self.indent -= 1

            self.wl("else:")
            self.indent += 1
            self.wl("logger.warning(f\"{name} not found in self._parameters and self._constants\")")
            self.indent -= 1
            self.indent -= 1
            self.indent -= 1
        else:
            assert False, "TODO: Add other framework param parsers"


class PyTorchWriter(PythonWriter):
    incompatible_np_float_types = [tf.bfloat16, ]

    def __init__(self, module_name, source_framework):
        super().__init__(module_name)

        self.framework = source_framework
        self.param_names = []
        self.const_names = []
        self.num_submodels = 0
        self.dev = "CPUDevice"
        self.submodules = []

    def write_header(self):
        self.wl("import torch")
        self.wl("from torch import nn")
        self.wl("\n")

    def clean_name(self, name):
        new_name = []
        for mod in name.split('.'):
            if mod.isdigit():
                new_name.append('l' + mod)
            else:
                new_name.append(mod)
        new_name = '.'.join(new_name)
        
        return new_name.replace('/', '.').replace(':', '_')

    def write_class_definition(self, params, constants, class_name=None, num_submodels=0, is_submodel=False):
        if class_name is None:
            class_name = self.class_name
        self.num_submodels = num_submodels
        self.wl(f"class {class_name}(nn.Module):")
        self.indent += 1
        self.wl("def __init__(self):")
        self.indent += 1
        self.wl(f"super().__init__()")

        for param in params.values():
            full_name, shape, requires_grad, dtype = param
            if dtype == "uint1":
                dtype = "bool"

            name = self.clean_name(full_name)
            self.param_names.append(name)
            prefix = ""
            while "." in name:
                mod, _, name = name.partition(".")
                submodule = prefix + f"{mod}"
                if submodule not in self.submodules:
                    self.wl(f"self.{submodule} = nn.Module()")
                    self.submodules.append(submodule)
                prefix = submodule + "."

            mod = self.clean_name(full_name).rpartition(name)[0]
            self.wl(f"self.{mod}register_parameter(\"{name}\", nn.Parameter(torch.zeros(*{shape}, dtype={pytorch_df_str_from_str(dtype, name)}), requires_grad={requires_grad}))")

        for const in constants.values():
            full_name, shape, dtype = const
            if dtype == "uint1":
                dtype = "bool"

            name = self.clean_name(full_name)
            self.const_names.append(name)
            prefix = ""
            while "." in name:
                mod, _, name = name.partition(".")
                submodule = prefix + f"{mod}"
                if submodule not in self.submodules:
                    self.wl(f"self.{submodule} = nn.Module()")
                    self.submodules.append(submodule)
                prefix = submodule + "."

            mod = self.clean_name(full_name).rpartition(name)[0]
            self.wl(f"self.{mod}register_buffer(\"{name}\", torch.zeros(*{shape}, dtype={pytorch_df_str_from_str(dtype, name)}))")


        self.indent = 0
        self.wl("")
        
    def get_op_output_structure(self, op):
        """
        Defines operand output structure.

        Args:
            op (Operation): Reference operation

        Returns:
            str: Operation output structure
        """
        op_out_structure = op.output_name
        
        # Handle special case PyTorch outputs
        if op.function_name == "torch.max":
            op_out_structure += ", _"
            
        return op_out_structure

    def get_op_input_arguments(self, op):
        """
        Constructs operand inputs in expected order. 
        
        As TVM recognize differences between operand inputs 
        and attributes (unlike PyTorch), we reconstruct those
        within PyTorchWriter and generate ordered arguments
        which can be used for relevant PyTorch function/operand.        

        Args:
            op (Operation): Reference operation

        Returns:
            str: Ordered function/operation attributes
        """
        op_arguments = []
        
        assert type(op.args) is dict, "Invalid op input type"

        # Structure TVM inputs
        for name in op.args['inp']:
            if self.clean_name(name) in self.param_names:
                op_arguments.append(f"self.get_parameter(\"{self.clean_name(name)}\")")
            elif self.clean_name(name) in self.const_names:
                op_arguments.append(f"self.get_buffer(\"{self.clean_name(name)}\")")
            else:
                op_arguments.append(name)
                
        # Handle special case TVM inputs - concatenate
        if op.function_name == "torch.cat":
            op_arguments = ["(" + ", ".join([arg for arg in op_arguments]) + ")"]
            
        # Handle special case TVM inputs - stack
        if op.function_name == "torch.stack":
            op_arguments = ["(" + ", ".join([arg for arg in op_arguments]) + ",)"]
            
        # Handle special case TVM inputs - where
        if op.function_name == "torch.where":
            # Handle when where represents adv_index op
            if "adv_index" in op.output_name:
                # Reverse order of arguments
                op_arguments = op_arguments[::-1]
                # Add empty tensor as last argument
                op_arguments.append(f'torch.zeros_like({op_arguments[0]}, dtype=torch.float32)')
                
            op_arguments[0] = op_arguments[0] + ".bool()"

        if op.function_name == "torch.index_select" and len(op_arguments) == 2:
            op_arguments[-1] = f"torch.squeeze({op_arguments[-1]}).long()"
        
        # Handle special case TVM inputs - broadcast_to_like
        if op.function_name == "torch.broadcast_to_like":
            op.function_name = "torch.broadcast_to"
            op_arguments[-1] = op_arguments[-1] + ".shape"
            
        # Structure TVM attributes
        for attr_name, attr_info in op.args['attr'].items():
            if 'as_named' in attr_info:
                op_arguments.insert(attr_info['inp_pos'], f"{attr_name}={attr_info['val']}")
            else:
                op_arguments.insert(attr_info['inp_pos'], f"{attr_info['val']}")
            
        # Construct operand arguments
        op_arguments = "".join([", " + name for name in op_arguments]).lstrip(", ")

        return op_arguments

    def _write_determine_shape_for_batch_dim(self):
        self.indent = 1
        self.wl(f"def determine_shape_for_batch_dim(self, old_shape, batch_dim):")
        self.indent += 1
        self.wl("old_shape = list(old_shape)")
        self.wl(f"if old_shape[0] == batch_dim:")
        self.indent += 1
        self.wl("return old_shape")
        self.indent -= 1
        self.wl("")
        
        self.wl(f"if old_shape[0] != 1:")
        self.indent += 1
        self.wl("old_shape.insert(0, 1)")
        self.indent -= 1
        self.wl("")
        self.wl(f"new_shape = [batch_dim]")
        self.wl(f"extracted_batch_dim = False")
        self.wl(f"for i in range(1, len(old_shape)):")
        self.indent += 1
        self.wl(f"if not extracted_batch_dim:")
        self.indent += 1
        self.wl(f"new_shape.append(old_shape[i] // batch_dim)")
        self.wl(f"extracted_batch_dim = True")
        self.indent -= 1
        self.wl(f"else:")
        self.indent += 1
        self.wl(f"new_shape.append(old_shape[i])")
        self.indent -= 1
        self.indent -= 1
        
        self.wl(f"return new_shape")
        self.indent = 0
        self.wl("")
    
    def write_forward(self, ops, inputs, outputs, force_batch_dim_outputs=[]):
        
        if len(force_batch_dim_outputs):
            self._write_determine_shape_for_batch_dim()
        
        self.indent = 1
        activation_names = "".join([", " + name for name in [inputs[key] for key in sorted(inputs)]])

        self.wl("def forward(self" + activation_names + "):")
        self.indent += 1

        for key in sorted(ops):
            op_attributes = self.get_op_input_arguments(ops[key])
            op_out_structure = self.get_op_output_structure(ops[key])

            if ops[key].args['inplace']:
                op_attributes = op_attributes.split(', ')
                lhs = op_attributes[0]
                rhs = ", ".join(op_attributes[1:])
                self.wl(f"{op_out_structure} = {lhs}.{ops[key].function_name}({rhs})")
            else:
                self.wl(f"{op_out_structure} = {ops[key].function_name}({op_attributes})")

        outputs = list(outputs.values())
        self.wl("")
        
        if len(force_batch_dim_outputs):
            self.wl("# Batch dim handling")
            # Force reshape to expose batch dimension
            self.wl(f"batch_dim = {inputs[0]}.shape[0]")
            for name in force_batch_dim_outputs:
                output_idx = outputs.index(name)
                self.wl(f"{outputs[output_idx]} = torch.reshape({name}, self.determine_shape_for_batch_dim({name}.shape, batch_dim))")

            self.wl("")
            
        if len(outputs) == 1:
            output_names = outputs[0]
        else:
            output_names = ", ".join(outputs)

        self.wl(f"return {output_names}")
        self.indent = 0
        self.wl("")


    def write_param_parser(self, param_names, param_file_name):
        self.indent = 1

        if self.framework == "pytorch":
            self.wl(f"def process_framework_parameters(self, model):")
            self.indent += 1

            self.wl("named_parameters = dict(model.named_parameters())")
            self.wl("my_params = dict(self.named_parameters())")
            self.wl("for name, parameter in named_parameters.items():")
            self.indent += 1
            self.wl("name = '.'.join(['l' + mod if mod.isdigit() else mod for mod in name.split('.')])")
            self.wl("if name in my_params:")
            self.indent += 1
            self.wl("module_path, _, param_name = name.rpartition(\".\")")
            self.wl("setattr(self.get_submodule(module_path), param_name, parameter)")
            self.indent -= 1
            self.indent -= 1
            self.wl("named_buffers = dict(model.named_buffers())")
            if param_file_name is not None:
                self.wl(f"serialized_params = torch.load(\"{param_file_name}\")")
                self.wl("named_buffers.update(serialized_params)")

            self.wl("self.load_state_dict(named_buffers, strict=False)")
            self.indent = 0
        elif self.framework == "onnx":
            self.wl(f"def process_framework_parameters(self, model):")
            self.indent += 1

            self.wl(f"import onnx")
            self.wl(f"import onnx.numpy_helper")
            self.wl(f"import numpy as np")
            self.wl(f"weights = model.graph.initializer")
            self.wl("named_parameters = dict(self.state_dict())")
            self.wl("named_buffers = dict(self.named_buffers())")
            self.wl("for weight in weights:")
            self.indent += 1
            self.wl("name = weight.name")
            self.wl("name = '.'.join(['l' + mod if mod.isdigit() else mod for mod in name.split('.')])")
            self.wl("weight_numpy = onnx.numpy_helper.to_array(weight)")
            self.wl("tensor = torch.tensor(weight_numpy)")
            # self.breakpoint()
            self.wl("tensor.requires_grad = issubclass(weight_numpy.dtype.type, np.floating) or issubclass(weight_numpy.dtype.type, np.complex)")

            self.wl("if name in named_parameters:")
            self.indent += 1
            self.wl("module_path, _, param_name = name.rpartition(\".\")")
            self.wl("getattr(self.get_submodule(module_path), param_name).data = tensor")
            self.indent -= 1
            self.wl("named_parameters[name] = tensor")
            
            self.indent -= 1

            if param_file_name is not None:
                self.wl("named_parameters.update(named_buffers)")
                self.wl(f"serialized_params = torch.load(\"{param_file_name}\")")
                self.wl("named_parameters.update(serialized_params)")

            self.wl("self.load_state_dict(named_parameters, strict=False)")

        elif self.framework == "tensorflow":
            self.wl(f"def process_framework_parameters(self, model):")
            self.indent += 1
            self.wl("named_parameters = {}")
            self.wl(f"for weight in model.weights:")
            self.indent += 1
            self.wl(f"name = weight.name.replace(\'/\', \'.\').replace(\':\', \'_\')")
            self.wl("name = '.'.join(['l' + mod if mod.isdigit() else mod for mod in name.split('.')])")
            self.wl(f"value = nn.Parameter(torch.from_numpy(weight.numpy()))")
            self.wl("named_parameters[name] = value\n")
            self.indent -= 1
            if param_file_name is not None:
                self.wl(f"serialized_params = torch.load(\"{param_file_name}\")")
                self.wl("serialized_params_cleaned = {}")
                self.wl(f"for key, value in serialized_params.items():")
                self.indent += 1
                self.wl(f"name = key.replace(\'/\', \'.\').replace(\':\', \'_\')")
                self.wl(f"serialized_params_cleaned[name] = value")
                self.indent -= 1
                self.wl("named_parameters.update(serialized_params_cleaned)\n")
            self.wl("self.load_state_dict(named_parameters, strict=False)")
        elif self.framework == "jax":
            self.wl(f"def process_framework_parameters(self, model):")
            self.indent += 1
            self.wl(f"import flax")
            self.wl(f"import numpy as np")
            self.wl(f"import jax.numpy as jnp")
            self.wl(f"from collections.abc import MutableMapping")
            self.wl(f"from transformers.modeling_flax_utils import FlaxPreTrainedModel")
            self.wl(f"def flatten_params(params, parent_key='', sep='.'):")
            self.indent += 1
            self.wl(f"items = []")
            self.wl(f"for key, val in params.items():")
            self.indent += 1
            self.wl(f"new_key = parent_key + sep + key if parent_key else key")
            self.wl(f"if isinstance(val, MutableMapping):")
            self.indent += 1
            self.wl(f"items.extend(flatten_params(val, new_key, sep=sep).items())")
            self.indent -= 1
            self.wl(f"else:")
            self.indent += 1
            self.wl(f"items.append((new_key, val))")
            self.indent -= 1
            self.indent -= 1
            self.wl(f"return dict(items)")
            self.indent -= 1
            self.wl(f"if isinstance(model, FlaxPreTrainedModel):")
            self.indent += 1
            self.wl(f"model_params = model.params")
            self.indent -= 1
            self.wl(f"elif isinstance(model, flax.linen.Module):")
            self.indent += 1
            self.wl("model_params = {}")
            self.wl("if hasattr(model, 'params'):")
            self.indent += 1
            self.wl(f"model_params = model.variables['params']._dict")
            self.indent -= 1
            self.indent -= 1
            self.wl("named_parameters = {}")
            if param_file_name is not None:
                self.wl(f"serialized_params = torch.load(\"{param_file_name}\")")
                self.wl(f"for key, val in serialized_params.items():")
                self.indent += 1
                self.wl(f"named_parameters[key] = val")
                self.indent -= 1
            self.wl(f"module_params = flatten_params(model_params)")
            self.wl(f"for key, value in module_params.items():")
            self.indent += 1
            self.wl(f"name = key")
            self.wl(f"value = nn.Parameter(torch.from_numpy(np.array(value)))")
            self.wl(f"value.requires_grad = True")
            self.wl(f"named_parameters[name] = value")
            self.indent -= 1
            self.wl(f"self.load_state_dict(named_parameters, strict=False)")
            self.indent -= 1
            self.indent -= 1

        elif self.framework == "tf_graphdef":
            self.wl(f"def process_framework_parameters(self, model):")
            self.indent += 1
            self.wl("named_parameters = {}")
            if param_file_name is not None:
                self.wl(f"serialized_params = torch.load(\"{param_file_name}\")")
                self.wl("serialized_params_cleaned = {}")
                self.wl(f"for key, value in serialized_params.items():")
                self.indent += 1
                self.wl(f"name = key.replace(\'/\', \'.\').replace(\':\', \'_\')")
                self.wl(f"serialized_params_cleaned[name] = value")
                self.indent -= 1
                self.wl("named_parameters.update(serialized_params_cleaned)\n")
            self.wl("self.load_state_dict(named_parameters, strict=False)")

        elif self.framework == "tflite":
            self.wl(f"def process_framework_parameters(self, model):")
            self.indent += 1
            self.wl("named_parameters = {}")
            if param_file_name is not None:
                self.wl(f"serialized_params = torch.load(\"{param_file_name}\")")
                self.wl("serialized_params_cleaned = {}")
                self.wl(f"for key, value in serialized_params.items():")
                self.indent += 1
                self.wl(f"name = key.replace(\'/\', \'.\').replace(\':\', \'_\')")
                self.wl(f"serialized_params_cleaned[name] = value")
                self.indent -= 1
                self.wl("named_parameters.update(serialized_params_cleaned)\n")
            self.wl("self.load_state_dict(named_parameters, strict=False)")
