# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch
from loguru import logger

from .tensor import Tensor, TensorShape, TensorBase, pytorch_dtype_to_buda_dataformat, pad_pytorch_tensor_to_buda, buda_dataformat_to_pytorch_dtype
from .pybudaglobal import lazy_trace_data
from pybuda._C import DataFormat
import pybuda

class Parameter(TensorBase):
    """
    Module parameter
    """
    def __init__(
        self,
        *args,
        requires_grad: bool = True,
        name: str = None,
        dev_data_format: Optional[DataFormat] = None,
    ):
        """
        Create parameter of given shape.
        
        Parameters
        ----------
        *args: Union[int, torch.Tensor]
            Parameter dimensions, or, tensor value for the parameter.

        requires_grad: bool, optional
            Set to false if this is not a trainable parameter. True by default.

        name: str, optional
            Parameter name will be auto-set based on the module hierarchy and the name of the variable it is stored as
            in the module. However, this parameter allows the user to specify a custom name, ignoring the auto-naming
            path. In that case, it will be user's responsibility to ensure that all parameters have unique names.

        dev_data_format: DataFormat, optional
            If set, forces the data type on device. If not provided, the closest type to given value will be used.
        """

        if len(args) == 0:
            raise RuntimeError("Initiali value, or list of dimensions (i.e. shape) must be provided.")

        if isinstance(args[0], torch.Tensor):
            self._value = args[0]
            self._tensor_shape = TensorShape(*self._value.shape)
        else:
            self._value = None
            self._tensor_shape = TensorShape(*args)

        self._requires_grad: bool = requires_grad
        self.forced_name: str = name
        self.auto_name: str = None  # To be calculated at the time of compile

        self.empty_tensor = None
        self.fp32_fallback = DataFormat.Float16_b
        if dev_data_format is not None:
            self._data_format = dev_data_format
        elif self._value is not None:
            self._data_format = pytorch_dtype_to_buda_dataformat(self._value.dtype)
        else:
            self._data_format = DataFormat.Float32 # default

    def __repr__(self):
        ret = f"Buda Parameter {self.get_name()} {self.shape}"
        if self.has_value():
            ret = f"{ret}\n {self.value()}"
        return ret

    @property
    def shape(self) -> TensorShape:
        return self._tensor_shape

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    def set_requires_grad(self, value: bool):
        self._requires_grad = value

    def get_empty_tensor(self) -> torch.Tensor:
        """
        Return an empty tensor of the right shape.

        Returns
        -------
        torch.Tensor
            Parameter's tensor
        """
        if self.empty_tensor is None:
            if self._data_format is None:
                if self.fp32_fallback is None:
                    data_format = DataFormat.Float32
                else:
                    data_format = self.fp32_fallback
            else:
                data_format = self._data_format

            self.empty_tensor = Tensor.create(
                    self._tensor, 
                    requires_grad=self.requires_grad, 
                    data_format=data_format, 
                    parameter=True)

        return self.empty_tensor

    def get_name(self) -> str:
        """
        Returns parameter name. This could return None if called before compilation and no user-provided name is set.
        
        Returns
        -------
        str
            Parameter name
        """
        if self.forced_name is not None:
            return self.forced_name

        return self.auto_name

    def set_value(self, value: torch.Tensor):
        if len(value.shape) == 0:
            with torch.no_grad():
                for _ in range(len(self.shape)):
                    value = value.unsqueeze(0)
        assert all([dima == dimb] for dima, dimb in zip(value.shape, self.shape))
        """
        Set tensor value of the parmeter. The parameter will be copied to the target device (if needed) before running.

        Parameters
        ----------
        value: SomeTensor
            Parameter value
        """
        logger.trace("Setting parameter ({}) value to ".format(self.auto_name))
        lazy_trace_data(value)
        self._value = value
        if self._data_format is None:
            self._data_format = pytorch_dtype_to_buda_dataformat(self._value.dtype, fp32_fallback=self.fp32_fallback)


    def value(self, is_buda = False) -> torch.Tensor:
        """
        Return parameter value, optionally padded to buda dimensions.
        """

        if self._value is not None:
            # parameters are never tile-broadcast
            ret = pad_pytorch_tensor_to_buda(self._value, []) if is_buda else self._value
            ret.requires_grad = self.requires_grad
            return ret

        return None

    def has_value(self) -> bool:
        return self._value is not None

    def _set_auto_name(self, name: str):
        """
        Set automatic name from module __setattr__
        """
        self.auto_name = name

    def _set_fp32_fallback(self, fp32_fallback: DataFormat):
        """
        Set FP32 fallback format if this is running on a device that doesn't support FP32.
        """
        self.fp32_fallback = fp32_fallback

    @property
    def data_format(self) -> DataFormat:
        """
        Return this parameter's PyBuda data format
        """
        assert self._data_format is not None, "No data type set for parameter yet"
        return self._data_format

    @property
    def pt_data_format(self) -> torch.dtype:
        """
        Return this parameter's data format, using PyTorch types
        """
        assert self._data_format is not None, "No data type set for parameter yet"
        return buda_dataformat_to_pytorch_dtype(self._data_format)

    def set_data_format(self, df: DataFormat):
        """
        Set data format for the parameter.

        Parameters
        ----------
        df: DataFormat
            PyBuda data format
        """
        self._data_format = df
        if self._value is not None:
            self._value = self._value.type(buda_dataformat_to_pytorch_dtype(df))

    @classmethod
    def create_from_torch(cls, torch_tensor: torch.Tensor) -> "Parameter":
        """
        Create parameter from pytorch tensor, and set value
        """
        return Parameter(
            torch_tensor,
            requires_grad=torch_tensor.requires_grad,
        )

    def _create_const_tensor(self, value):
        assert isinstance(value, (int, float)), f"Automatic constant tensor creation for {type(value)} not supported"
        return pybuda.op.Constant("", constant=value)

    def _handle_binary_op(self, other, op, is_r=False):
        if not isinstance(other, (pybuda.Tensor, pybuda.Parameter)):
            other = self._create_const_tensor(other)
        if not is_r:
            return op("", self, other)
        else:
            return op("", other, self)

    def __add__(self, other):
        return self._handle_binary_op(other, pybuda.op.Add)

    def __radd__(self, other):
        return self._handle_binary_op(other, pybuda.op.Add, is_r=True)

    def __sub__(self, other):
        return self._handle_binary_op(other, pybuda.op.Subtract)

    def __rsub__(self, other):
        return self._handle_binary_op(other, pybuda.op.Subtract, is_r=True)

    def __mul__(self, other):
        return self._handle_binary_op(other, pybuda.op.Multiply)

    def __rmul__(self, other):
        return self._handle_binary_op(other, pybuda.op.Multiply, is_r=True)
