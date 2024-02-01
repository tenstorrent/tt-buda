# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union

from ..tensor import Tensor
from ..parameter import Parameter
from pybuda.op.eval.pybuda import get_f_pybuda_eval, get_f_pybuda_shape
from pybuda._C import DataFormat
from pybuda._C.graph import OpType
import pybuda
from pybuda.pybudaglobal import get_unique_node_id, tracing

depracated_name_dict = {}
deprecated_op_id = 0
class PyBudaOp:

    def __init__(
        self,
        op_type: str,
        name: str,
        *operands: Union[Tensor, Parameter],
        attrs: Tuple[int, ...] = (),
        **named_attrs):
        """
        Create an op with given parameters.
        """
        self.op_type = op_type

        global deprecated_op_id, depracated_name_dict
        if tracing():
            if name != "":
                self.name = name
            else:
                unique_id = get_unique_node_id()
                self.name = f"{op_type}_{unique_id}"
                if (unique_id != deprecated_op_id):
                    depracated_name_dict[f"{op_type}_{deprecated_op_id}"] = self.name
        deprecated_op_id += 1

        operands = tuple(pybuda.op.Constant("", constant=operand) if isinstance(operand, (int, float)) else operand for operand in operands)
        self.operands = operands
        self.attrs = attrs
        self.named_attrs = named_attrs
        self.cpp_op_type = OpType(self.op_type, self.attrs, self.named_attrs)

    def get_tensor(self, out_df=None) -> Tensor:
        """
        Generate result tensor of the right shape, and if value is set, value.
        """
        #shapes = [o.shape.get_pytorch_shape() if isinstance(o, (Tensor, Parameter)) else o for o in self.operands]
        shapes = [o.shape.get_pytorch_shape() for o in self.operands]
        shape, self.operand_broadcast = get_f_pybuda_shape(self.cpp_op_type)(shapes)

        # TODO: pick data formats in some way when mismatched inputs are coming...
        if out_df is not None:
            data_format = out_df # User provided output dataformat

        # we should create a map that maps input dataformat to output dataformat 
        elif self.op_type in ["matmul", "conv2d"]:
            op0_df = self.operands[0].data_format
            op1_df = self.operands[1].data_format
            if op0_df == DataFormat.Int8 and op1_df == DataFormat.Int8:
                data_format = DataFormat.Int32
            else:
                data_format = self.operands[0].data_format
        elif len(self.operands) > 0:
            if self.op_type == "where":
                data_format = self.operands[1].data_format
            else:
                data_format = self.operands[0].data_format
        else:
            data_format = DataFormat.Float32 # what's correct here? TODO

        result = Tensor.create_from_trace(
            src_op = self,
            shape = shape,
            data_format = data_format
        )
        result.requires_grad = any([o.requires_grad for o in self.operands])

        # Calculate reference if there's one
        if all([o.has_value() if isinstance(o, (Tensor, Parameter)) else True for o in self.operands]):
            values = [o.value() if isinstance(o, (Tensor, Parameter)) else o for o in self.operands]
            result.set_value(get_f_pybuda_eval(self.cpp_op_type)(values))


        return result
