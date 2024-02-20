# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Training optimizers
"""

from typing import Dict, List, Optional, Tuple
import copy

import numpy as np
import torch

from pybuda.tensor import Tensor
from pybuda.parameter import Parameter
import pybuda.torch_optimizers
from pybuda.torch_optimizers import AdamNoBiasCorrection

class Optimizer:
    """
    Optimizer base class
    """
    def __init__(self, device_params: bool = False):
        """
        Create baseline optimizer. If device_params is set, no parameters are provided at the time of
        optimizer creation, but will be extracted from the device on which the optimizer is placed.
        """
        self.device_params = device_params

    def get_param_dict(self) -> Dict:
        """
        Return a dict of parameter node names and values to push to the device
        """
        raise RuntimeError("Subclasses should implement this.")

    def get_optimizer_params(self, parameter_name, is_buda) -> Optional[Dict[str, Tensor]]:
        raise RuntimeError("Subclasses should implement this.")

    def generate_op_trace(self, parameter, gradient):
        """
        Define the graph of ops involved in the optimizer eval.
        """
        raise RuntimeError("Subclasses should implement this.")

    def torch_parameter_update(self, *, parameter_name: str, parameter: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
        """
        Pytorch eval implementation for the optimizer

        Parameters
        ----------
        parameter : torch.Tensor
            parameter
        gradient : torch.Tensor
            gradient
        """
        raise RuntimeError("Subclasses should implement this.")

    def get_pytorch_optimizer(self, parameters: Dict[str, torch.Tensor]) -> torch.optim.Optimizer:
        raise RuntimeError("Subclasses should implement this.")

class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer

    Attributes
    ----------
    learning_rate : float
        learning_rate used by optimizer to adjust parameter
    parameter_to_opt_inputs : Dict[Parameter, Dict[str, Tensor]]
        Maps a Parameter with `requires_grad=True` to its associated
        optimizer_parameter_name -> Tensor dict.
    """

    def __init__(self, learning_rate: float, parameters: Optional[List[Parameter]] = None, device_params: bool = False):
        super().__init__(device_params)
        self.learning_rate: float = learning_rate
        self.parameter_to_opt_inputs: Dict[str, Dict[str, Tensor]] = {}

        if device_params:
            assert parameters is None

        else:
            assert parameters is not None
            self.set_parameters_to_optimize(parameters)

    def set_parameters_to_optimize(self, parameters: List[Parameter]):
        # For each Parameter, we register its associated set of optimizer parameters
        for parameter in parameters:
            if parameter.requires_grad:
                self.parameter_to_opt_inputs[parameter.get_name()] = self.get_param_dict(parameter.pt_data_format)

    def get_param_dict(self, dtype: torch.dtype) -> Dict:
        """
        Return a dict of optimizer parameter names to tensor
        """
        # Buda needs a pytorch array for now
        # TODO(jchu): modify these two lines when we deprecate the old path
        learning_rate_torch = torch.full( (1,), self.learning_rate, dtype=dtype)

        learning_rate = Tensor.create_from_torch(learning_rate_torch)
        return {"lr": learning_rate}

    def get_optimizer_state_keys(self) -> List:
        return []

    def get_optimizer_params(self, parameter_name, is_buda) -> Optional[Dict[str, Tensor]]:
        if parameter_name not in self.parameter_to_opt_inputs:
            return None

        ret = copy.copy(self.parameter_to_opt_inputs[parameter_name])
        if is_buda:
            for k, v in ret.items():
                # optimize params should always tile broadcast if they are scalar
                one_d = len(ret[k].shape) == 1 and ret[k].shape[0] == 1
                tile_broadcast_dims = [-1, -2] if one_d else []
                ret[k] = v.to_buda_shape(tile_broadcast_dims=tile_broadcast_dims, clone=True)
        return ret


    def get_type(self) -> str:
        return "sgd"

    def set_optimizer_parameters(self, learning_rate: Optional[float] = None) -> None:
        """
        Loop through every Parameter tensor with `requires_grad=True` and pushes
        `learning_rate` value to its associated optimizer parameter queue. By default,
        if no `learning_rate` is specified, the learning rate used to construct the
        sgd_optimizer is used.

        This method may be invoked multiple times as a way to
        adjust the updated optimizer parameter values across training loop iterations.

        Parameters
        ----------
        learning_rate : Optional[float]
            learning_rate

        Returns
        -------
        None

        """

        if learning_rate:
            self.learning_rate = learning_rate

        for parameter, opt_inputs in self.parameter_to_opt_inputs.items():
            learning_rate_tensor = torch.full( (1, ), self.learning_rate, dtype=opt_inputs["lr"].pt_data_format)
            opt_inputs["lr"] = Tensor.create_from_torch(learning_rate_tensor)

    def generate_op_trace(self, ac, parameter, gradient):
        lr = ac.input("lr", (1,))

        grad_times_lr = ac.op("multiply", (gradient, lr))
        param_minus_lr_times_grad = ac.op("subtract", (parameter, grad_times_lr))

        return param_minus_lr_times_grad

    def torch_parameter_update(self, *, parameter_name: str, parameter: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
        return parameter - self.learning_rate * gradient

    def get_pytorch_optimizer(self, parameters: Dict[str, torch.Tensor], lr = None) -> torch.optim.Optimizer:
        """
        Return an equivalent pytorch optimizer, used for verification.
        """
        if lr is not None:
            self.set_optimizer_parameters(learning_rate=lr)
        return torch.optim.SGD([p for p in parameters.values()], self.learning_rate)


class Adam(Optimizer):
    """
    Adam Optimizer

    Attributes
    ----------
    learning_rate : float
        learning_rate used by optimizer to adjust parameter
    parameter_to_opt_inputs : Dict[Parameter, Dict[str, Tensor]]
        Maps a Parameter with `requires_grad=True` to its associated
        optimizer_parameter_name -> Tensor dict.
    betas : (Tuple[float, float], optional)
        coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
    eps : (float, optional)
        term added to the denominator to improve numerical stability (default: 1e-8)
    weight_decay : (float, optional)
        weight decay (L2 penalty) (default: 0)
    bias_correction: (bool, optional)
        use bias correction
    """

    def __init__(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        bias_correction: bool = True,
        parameters: Optional[List[Parameter]] = None,
        device_params: bool = False,
        enable_adam_w: bool = False,
    ):
        super().__init__(device_params)
        # optimizer constants
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.bias_correction = bias_correction
        self.torch_optimizer = None

        self.learning_rate = learning_rate
        self.parameter_to_opt_inputs: Dict[str, Dict[str, Tensor]] = {}
        self.parameter_to_opt_torch_inputs: Dict[str, Dict[str, Tensor]] = {}
        self.enable_adam_w = enable_adam_w

        if device_params:
            assert parameters is None
        else:
            assert parameters is not None
            self.set_parameters_to_optimize(parameters)

    def get_cpu_param_dict(self, dtype: torch.dtype, shape: Tuple[int]) -> Dict:
        if self.bias_correction:
            return {
                "torch_mean": torch.full(shape, 0.0, dtype=dtype),
                "torch_variance": torch.full(shape, 0.0, dtype=dtype),
                "torch_beta1_pow": torch.full((1,), 1.0, dtype=dtype),
                "torch_beta2_pow": torch.full((1,), 1.0, dtype=dtype),
            }
        else:
            return {
                "torch_mean": torch.full(shape, 0.0, dtype=dtype),
                "torch_variance": torch.full(shape, 0.0, dtype=dtype),
            }

    def get_param_dict(self, dtype: torch.dtype, shape: Tuple[int]) -> Dict:
        """
        Return a dict of optimizer parameter names to tensor
        """
        torch_lr = torch.full((1,), self.learning_rate, dtype=dtype)
        if self.bias_correction:
            return {
                "lr": Tensor.create_from_torch(torch_lr),
                "mean": Tensor.create_from_torch(torch.full(shape, 0.0, dtype=dtype)),
                "variance": Tensor.create_from_torch(torch.full(shape, 0.0, dtype=dtype)),
                "beta1_pow": Tensor.create_from_torch(torch.full((1,), 1.0, dtype=dtype)),
                "beta2_pow": Tensor.create_from_torch(torch.full((1,), 1.0, dtype=dtype)),
            }
        else:
            return {
                "lr": Tensor.create_from_torch(torch_lr),
                "mean": Tensor.create_from_torch(torch.full(shape, 0.0, dtype=dtype)),
                "variance": Tensor.create_from_torch(torch.full(shape, 0.0, dtype=dtype)),
            }

    def get_optimizer_state_keys(self) -> List:
        if self.bias_correction:
            return ["mean", "variance", "beta1_pow", "beta2_pow"]
        else:
            return ["mean", "variance"]

    def get_type(self) -> str:
        return "adam"

    def set_parameters_to_optimize(self, parameters: List[Parameter]):
        # For each Parameter, we register its associated set of optimizer parameters
        for parameter in parameters:
            if parameter.requires_grad:
                self.parameter_to_opt_inputs[
                    parameter.get_name()
                ] = self.get_param_dict(parameter.pt_data_format, parameter.shape.get_pytorch_shape())
                self.parameter_to_opt_torch_inputs[
                    parameter.get_name()
                ] = self.get_cpu_param_dict(parameter.pt_data_format, parameter.shape.get_pytorch_shape())

    def get_optimizer_params(self, parameter_name, is_buda) -> Optional[Dict[str, Tensor]]:
        if parameter_name not in self.parameter_to_opt_inputs:
            return None

        ret = copy.copy(self.parameter_to_opt_inputs[parameter_name])
        if is_buda:
            for k, v in ret.items():
                # optimize params should always tile broadcast if they are scalar
                one_d = len(ret[k].shape) == 1 and ret[k].shape[0] == 1
                tile_broadcast_dims = [-1, -2] if one_d else []
                ret[k] = v.to_buda_shape(
                    tile_broadcast_dims=tile_broadcast_dims, clone=True
                )
        return ret

    def set_optimizer_parameters(self, learning_rate: Optional[float] = None) -> None:
        """
        Loop through every Parameter tensor with `requires_grad=True` and pushes
        `learning_rate` value to its associated optimizer parameter queue. By default,
        if no `learning_rate` is specified, the learning rate used to construct the
        sgd_optimizer is used.

        This method may be invoked multiple times as a way to
        adjust the updated optimizer parameter values across training loop iterations.

        Parameters
        ----------
        learning_rate : Optional[float]
            learning_rate

        Returns
        -------
        None

        """

        if learning_rate:
            self.learning_rate = learning_rate

        # {mean, variance} get updated in the loopback
        for parameter, opt_inputs in self.parameter_to_opt_inputs.items():
            torch_lr = torch.full(
                (1,), self.learning_rate, dtype=opt_inputs["lr"].pt_data_format
            )
            opt_inputs["lr"] = Tensor.create_from_torch(torch_lr)

    def generate_op_trace(self, ac, parameter, gradient):
        if self.weight_decay > 0.0:
            weight_decay = ac.constant(self.weight_decay)
        else:
            weight_decay = None
        ## import locally to avoid circular dependency from Dataformat, fix it later
        from pybuda.op.eval.pybuda.buffer import Buffer
        # we copy the grad accum. queue since it only accepts a single consumer/pop
        gradient_copy = ac.op(Buffer.create(), (gradient,))

        if weight_decay and not self.enable_adam_w:
            weight_decay_times_param = ac.op("multiply", (weight_decay, parameter))
            gradient_copy = ac.op("add", (gradient_copy, weight_decay_times_param))


        # self.mean = self.beta1 * self.mean + one_minus_beta1 * gradient
        mean = ac.input("mean", parameter.shape, copy_consteval_operations=True)
        beta1 = ac.constant(self.beta1)
        one_minus_beta1 = ac.constant(1 - self.beta1)
        mean_times_beta1 = ac.op("multiply", (mean, beta1))
        gradient_times_one_minus_beta1 = ac.op("multiply", (gradient_copy, one_minus_beta1))
        updated_mean = ac.op("add", (mean_times_beta1, gradient_times_one_minus_beta1))

        # self.variance = self.beta2 * self.variance + one_minus_beta2 * gradient**2
        variance = ac.input("variance", parameter.shape, copy_consteval_operations=True)
        beta2 = ac.constant(self.beta2)
        one_minus_beta2 = ac.constant(1 - self.beta2)
        variance_times_beta2 = ac.op("multiply", (variance, beta2))
        gradient_squared = ac.op("multiply", (gradient_copy, gradient_copy))
        gradient_squared_times_one_minus_beta2 = ac.op(
            "multiply", (gradient_squared, one_minus_beta2)
        )
        updated_variance = ac.op(
            "add", (variance_times_beta2, gradient_squared_times_one_minus_beta2)
        )
        from pybuda.op.eval.pybuda.reciprocal import Reciprocal
        #import Sqrt module locally to avoid circular dependency
        from pybuda.op.eval.pybuda.sqrt import Sqrt
        if self.bias_correction:
            # bias_correction1 = 1 - beta1 ** step
            beta1_one = ac.constant(1.0)
            beta1_pow = ac.input("beta1_pow", (1,), disable_consteval=True) # stores beta1 ** step
            updated_beta1_pow = ac.op("multiply", (beta1_pow, beta1))
            bias_correction1  = ac.op("subtract", (beta1_one, updated_beta1_pow))
            reciprocal_bias_correction1 = ac.op(Reciprocal.create(), (bias_correction1,))

            # bias_correction2 = 1 - beta2 ** step
            beta2_one = ac.constant(1.0)
            beta2_pow = ac.input("beta2_pow", (1,), disable_consteval=True) # stores beta2 ** step
            updated_beta2_pow = ac.op("multiply", (beta2_pow, beta2))
            bias_correction2 = ac.op("subtract", (beta2_one, updated_beta2_pow))
            sqrt_bias_correction2 = ac.op(Sqrt.create(), (bias_correction2,))
            reciprocal_sqrt_bias_correction2 = ac.op(Reciprocal.create(), (sqrt_bias_correction2,))

            # sqrt_of_variance / sqrt_bias_correction2
            sqrt_of_variance_biased = ac.op(Sqrt.create(), (updated_variance,))
            sqrt_of_variance = ac.op("multiply", (sqrt_of_variance_biased, reciprocal_sqrt_bias_correction2))
        else:
            sqrt_of_variance = ac.op(Sqrt.create(), (updated_variance,))

        epsilon = ac.constant(self.epsilon)
        sqrt_of_variance_plus_epsilon = ac.op("add", (sqrt_of_variance, epsilon))
        reciprocal_of_sqrt_of_variance_plus_epsilon = ac.op(
            Reciprocal.create(), (sqrt_of_variance_plus_epsilon,)
        )

        if self.bias_correction:
            # mean / bias_correction1
            updated_mean_unbiased = ac.op("multiply", (updated_mean, reciprocal_bias_correction1))
            mean_times_reciprocal_of_sqrt_of_variance_plus_epsilon = ac.op(
                "multiply", (updated_mean_unbiased, reciprocal_of_sqrt_of_variance_plus_epsilon)
            )
        else:
            mean_times_reciprocal_of_sqrt_of_variance_plus_epsilon = ac.op(
                "multiply", (updated_mean, reciprocal_of_sqrt_of_variance_plus_epsilon)
            )

        if weight_decay and self.enable_adam_w:
            # weight_decay * param + mean/sqrt(var)
            weight_decay_times_param = ac.op("multiply", (weight_decay, parameter))
            mean_times_reciprocal_of_sqrt_of_variance_plus_epsilon_plus_weight_decay_times_param = ac.op(
                "add", (mean_times_reciprocal_of_sqrt_of_variance_plus_epsilon, weight_decay_times_param)
            )
        else:
            mean_times_reciprocal_of_sqrt_of_variance_plus_epsilon_plus_weight_decay_times_param = mean_times_reciprocal_of_sqrt_of_variance_plus_epsilon

        lr = ac.input("lr", (1,))
        parameter_delta = ac.op(
            "multiply", (mean_times_reciprocal_of_sqrt_of_variance_plus_epsilon_plus_weight_decay_times_param, lr)
        )
        updated_parameter = ac.op("subtract", (parameter, parameter_delta))

        # in the old spatial1, there was a loopback for each of {updated_mean, updated_variance}
        ac.loopback(updated_mean, mean)
        ac.loopback(updated_variance, variance)
        if self.bias_correction:
            ac.loopback(updated_beta1_pow, beta1_pow)
            ac.loopback(updated_beta2_pow, beta2_pow)
        return updated_parameter

    def torch_parameter_update(
        self, *, parameter_name: str, parameter: torch.Tensor, gradient: torch.Tensor
    ) -> torch.Tensor:
        if not self.enable_adam_w and self.weight_decay > 0.0:
            gradient = gradient + self.weight_decay * parameter

        if self.bias_correction:
            updated_torch_mean = (self.beta1 * self.parameter_to_opt_torch_inputs[parameter_name]["torch_mean"] + (1 - self.beta1) * gradient) / (1 - self.beta1)
            updated_torch_variance = (self.beta2 * self.parameter_to_opt_torch_inputs[parameter_name]["torch_variance"] + (1 - self.beta2) * gradient**2) / (1 - self.beta2)
        else:
            updated_torch_mean = self.beta1 * self.parameter_to_opt_torch_inputs[parameter_name]["torch_mean"] + (1 - self.beta1) * gradient
            updated_torch_variance = self.beta2 * self.parameter_to_opt_torch_inputs[parameter_name]["torch_variance"] + (1 - self.beta2) * gradient**2

        updated_parameter = parameter - self.learning_rate * (
            updated_torch_mean / (torch.sqrt(updated_torch_variance) + self.epsilon)
            + (self.weight_decay * parameter if self.enable_adam_w and self.weight_decay > 0.0 else 0)
        )

        return updated_parameter


    def get_pytorch_optimizer(self, parameters: Dict[str, torch.Tensor], lr = None) -> torch.optim.Optimizer:
        """
        Return an equivalent pytorch optimizer, used for verification.
        """
        if lr is not None:
            self.set_optimizer_parameters(learning_rate=lr)
        # May want to initialize with initial learning_rate
        if not self.torch_optimizer:
            if self.bias_correction:
                optim = torch.optim.AdamW if self.enable_adam_w else torch.optim.Adam
                self.torch_optimizer = optim(
                    [p for p in parameters.values()],
                    lr=self.learning_rate,
                    betas=(self.beta1, self.beta2),
                    eps=self.epsilon,
                    weight_decay=self.weight_decay,
                )
            else:
                self.torch_optimizer = AdamNoBiasCorrection(
                    [p for p in parameters.values()],
                    lr=self.learning_rate,
                    betas=(self.beta1, self.beta2),
                    eps=self.epsilon,
                    weight_decay=self.weight_decay,
                    enable_adam_w=self.enable_adam_w
                )
        return self.torch_optimizer


def get_optimizer_type_from_string(type_string: str):
    # replace this implementation with one that inspects module classes
    string_to_type = {
        "sgd": SGD,
        "adam": Adam,
        "adamw": AdamW,
        "lars": LARS,
        "lamb": LAMB

    }
    return string_to_type[type_string]


class AdamW(Adam):
    """
    Implements weighted Adam optimizer.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        bias_correction: bool = True,
        parameters: Optional[List[Parameter]] = None,
        device_params: bool = False
    ):
        super().__init__(learning_rate, beta1, beta2, eps, weight_decay, bias_correction, parameters, device_params, enable_adam_w=True)


class LAMB(Optimizer):
    
    def __init__(
        self, 
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        correction: bool = False,
        clip_value = (0.0, 10.0),
        parameters: Optional[List[Parameter]] = None,
        device_params: bool = False
    ):
        super().__init__(device_params)
        
        # LAMB Optimizer Constants
        assert learning_rate >= 0.0, f"Invalid learning rate value: {learning_rate}"
        assert beta1 >= 0.0 and beta1 < 1.0, f"Invalid beta1 value: {beta1}"
        assert beta2 >= 0.0 and beta2 < 1.0, f"Invalid beta2 value: {beta2}"
        assert weight_decay >= 0.0, f"Invalid weight decay value: {weight_decay}"
        assert clip_value[0] >= 0.0 and clip_value[0] < 10.0, f"Invalid lower clip value: {clip_value[0]}"
        assert clip_value[1] > 0.0 and clip_value[1] <= 10.0, f"Invalid higher clip value: {clip_value[1]}"

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.correction = correction
        self.clip_value = clip_value

        self.torch_optimizer = None

        self.parameter_to_opt_inputs: Dict[str, Dict[str, Tensor]] = {}
        self.parameter_to_opt_torch_inputs: Dict[str, Dict[str, Tensor]] = {}

        if device_params:
            assert parameters is None
        else:
            assert parameters is not None
            self.set_parameters_to_optimize(parameters)

        self.device_params = device_params


    def set_parameters_to_optimize(self, parameters: List[Parameter]):
        # For each Parameter, we register its associated set of optimizer parameters
        for parameter in parameters:
            if parameter.requires_grad:
                # PyBuda
                self.parameter_to_opt_inputs[
                    parameter.get_name()
                ] = self.get_param_dict(
                        parameter.pt_data_format, 
                        parameter.shape.get_pytorch_shape()
                    )
                # PyTorch
                self.parameter_to_opt_torch_inputs[
                    parameter.get_name()
                ] = self.get_cpu_param_dict(
                        parameter.pt_data_format, 
                        parameter.shape.get_pytorch_shape()
                    )


    def set_optimizer_parameters(self, learning_rate: Optional[float] = None) -> None:
        """
        Loop through every Parameter tensor with `requires_grad=True` and pushes
        `learning_rate` value to its associated optimizer parameter queue. By default,
        if no `learning_rate` is specified, the learning rate used to construct the
        sgd_optimizer is used.

        This method may be invoked multiple times as a way to
        adjust the updated optimizer parameter values across training loop iterations.

        Parameters
        ----------
        learning_rate : Optional[float]
            learning_rate

        Returns
        -------
        None

        """

        if learning_rate:
            self.learning_rate = learning_rate

        # {mean, variance} get updated in the loopback
        for parameter, opt_inputs in self.parameter_to_opt_inputs.items():
            torch_lr = torch.full(
                (1,), self.learning_rate, dtype=opt_inputs["lr"].pt_data_format
            )
            opt_inputs["lr"] = Tensor.create_from_torch(torch_lr)


    def get_optimizer_state_keys(self) -> List:
        return []


    def get_cpu_param_dict(self, dtype: torch.dtype, shape: Tuple[int]) -> Dict:
        return {
            "torch_mean": torch.full(shape, 0.0, dtype=dtype),
            "torch_variance": torch.full(shape, 0.0, dtype=dtype),
        }


    def get_param_dict(self, dtype: torch.dtype, shape: Tuple[int]) -> Dict:

        torch_lr = torch.full((1,), self.learning_rate, dtype=dtype)
        return {
            "lr": Tensor.create_from_torch(torch_lr),
            "mean": Tensor.create_from_torch(torch.full(shape, 0.0, dtype=dtype)),
            "variance": Tensor.create_from_torch(torch.full(shape, 0.0, dtype=dtype)),
        }


    def get_optimizer_params(self, parameter_name, is_buda) -> Optional[Dict[str, Tensor]]:

        if parameter_name not in self.parameter_to_opt_inputs:
            return None

        ret = copy.copy(self.parameter_to_opt_inputs[parameter_name])
        if is_buda:
            for k, v in ret.items():
                # optimize params should always tile broadcast if they are scalar
                one_d = len(ret[k].shape) == 1 and ret[k].shape[0] == 1
                tile_broadcast_dims = [-1, -2] if one_d else []
                ret[k] = v.to_buda_shape(
                    tile_broadcast_dims=tile_broadcast_dims, clone=True
                )
        return ret


    def get_type(self) -> str:
        return "lamb"


    def generate_op_trace(self, ac, parameter, gradient):

        # get parameter shape
        param_shape = parameter.shape.as_list()

        # gradinet buffering

        # g(t) -> gradient at current timestep
        #temp fix to avoid circular dependency by importing locally
        from pybuda.op.eval.pybuda.buffer import Buffer
        grad = ac.op(Buffer.create(), (gradient, ))

        # m(t) <- beta1 * m(t - 1) + (1 - beta1) * g(t)
        # m(t)     : mean at current timestep
        # m(t - 1) : mean at previous timestep
        mean = ac.input("mean", parameter.shape)    # m(t)
        beta1_torch = torch.zeros(param_shape) + self.beta1
        beta1 = ac.tensor(beta1_torch)      # beta1

        mean_1 = ac.op("multiply", (beta1, mean))   
        mean_2_1 = ac.tensor(1 - beta1_torch)
        mean_2 = ac.op("multiply", (mean_2_1, grad))
        updated_mean = ac.op("add", (mean_1, mean_2))

        # v(t) <- beta2 * v(t -1) + (1 - beta2) * g(t) ^ 2
        # v(t)     : variance at current timestep
        # v(t - 1) : variance at previous timestep  
        variance = ac.input("variance", parameter.shape)
        beta2_torch = torch.zeros(param_shape) + self.beta2
        beta2 = ac.tensor(beta2_torch)
        
        variance_1 = ac.op("multiply", (beta2, variance))
        variance_2_1 = ac.tensor(1 - beta2_torch)
        variance_2_2 = ac.op("multiply", (grad, grad))
        variance_2 = ac.op("multiply", (variance_2_1, variance_2_2))
        updated_variance = ac.op("add", (variance_1, variance_2))

        '''
        if self.correction:
            # m_hat(t) <- m(t) / (1 - beta1 ^ t)
            mean_hat = None
            # v_hat(t) <- v(t) / (1 - beta2 ^ t)
            variance_hat = None
        '''

        # L2 normaliztion of weights
        phi_norm = ac.op("multiply", (parameter, parameter))
        phi_norm_shape = phi_norm.shape.as_list()
        if len(phi_norm_shape) > 1:
            phi_norm = ac.op("reduce_sum", (phi_norm, ), (-2, ))
        phi_norm = ac.op("reduce_sum", (phi_norm, ), (-1, ))

        #importing locally to avoid circular dependency from Dataformats
        from pybuda.op.eval.pybuda.sqrt import Sqrt
        phi_norm = ac.op(Sqrt.create(), (phi_norm, ))

        epsilon = ac.tensor(torch.zeros(param_shape) + self.eps)
        weight_decay = ac.tensor(torch.zeros(param_shape) + self.weight_decay)

        # adam ratio, ratio of corrected mean and corrected variance stabilized with epsilon
        r_t = ac.op(Sqrt.create(), (updated_variance, ))
        r_t = ac.op("add", (r_t, epsilon))
        from pybuda.op.eval.pybuda.reciprocal import Reciprocal
        r_t = ac.op("multiply", (updated_mean,  ac.op(Reciprocal.create(), (r_t, ))))

        if self.weight_decay != 0:
            decayed_param = ac.op("multiply", (parameter, weight_decay))
            r_t = ac.op("add", (r_t, decayed_param))

        r_t_norm = ac.op("multiply", (r_t, r_t))
        r_t_norm_shape = r_t_norm.shape.as_list()
        if len(r_t_norm_shape) > 1:
            r_t_norm = ac.op("reduce_sum", (r_t_norm, ), (-2, ))
        r_t_norm = ac.op("reduce_sum", (r_t_norm, ), (-1, ))
        r_t_norm = ac.op(Sqrt.create(), (r_t_norm, ))

        #
        #   IF phi_norm != 0 AND r_t_norm != 0:
        #       trust_ratio = phi_norm / r_t_norm
        #   ELSE
        #       trust_ratio = 1
        #
        # We don't support branching, so we compute both branches, let's decompose IF-ELSE block
        #
        #   IF phi_norm != 0:
        #       IF r_t_norm != 0:
        #           trust_ratio = phi_norm / r_t_norm
        #       ELSE
        #           trust_ratio = 1
        #   ELSE
        #       trust_ratio = 1
        #
        #
        # so, trust ratio is
        #
        #   trust_ratio = (phi_norm != 0) * ((r_t_norm != 0) * (phi_norm / r_t_norm) + (r_t_norm == 0)) + (phi_norm == 0)
        #

        zero = ac.tensor(torch.zeros(phi_norm_shape))
        phi_norm_ne = ac.op("not_equal", (phi_norm, zero))
        phi_norm_eq = ac.op("equal", (phi_norm, zero))
        r_t_norm_ne = ac.op("not_equal", (r_t_norm, zero))
        r_t_norm_eq = ac.op("equal", (r_t_norm, zero))
        trust_ratio = ac.op(Reciprocal.create(), (r_t_norm, ))
        trust_ratio = ac.op("multiply", (phi_norm, trust_ratio))
        trust_ratio = ac.op("clip", (trust_ratio, ), (self.clip_value[0], self.clip_value[1]))
        trust_ratio = ac.op("multiply", (trust_ratio, r_t_norm_ne))
        trust_ratio = ac.op("add", (trust_ratio, r_t_norm_eq))
        trust_ratio = ac.op("multiply", (trust_ratio, phi_norm_ne))
        trust_ratio = ac.op("add", (trust_ratio, phi_norm_eq))

        # w(t) = w(t - 1) - learning_rate * adam_ratio * trust_ratio
        learning_rate = ac.tensor(torch.zeros(param_shape) + self.learning_rate)
        updated_parameter = ac.op("multiply", (trust_ratio, r_t))
        updated_parameter = ac.op("multiply", (updated_parameter, learning_rate))
        updated_parameter = ac.op("subtract", (parameter, updated_parameter))

        # Update mean and variance
        ac.loopback(updated_mean, mean)
        ac.loopback(updated_variance, variance)
        
        return updated_parameter


    def torch_parameter_update(
        self, 
        *, 
        parameter_name: str, 
        parameter: torch.Tensor, 
        gradient: torch.Tensor
    ) -> torch.Tensor:

        torch_mean = self.parameter_to_opt_torch_inputs[parameter_name]["torch_mean"]
        torch_variance = self.parameter_to_opt_torch_inputs[parameter_name]["torch_variance"]
        
        # m(t) <- beta1 * m(t - 1) + (1 - beta1) * g(t)
        # m(t)     : mean at current timestep
        # m(t - 1) : mean at previous timestep
        mean = self.beta1 * torch_mean + (1 - self.beta1) * gradient

        # v(t) <- beta2 * v(t - 1) + (1 - beta2) * g(t) ^ 2  
        # v(t)     : variance at current timestep
        # v(t - 1) : variance at previous timestep
        variance = self.beta2 * torch_variance + (1 - self.beta2) * gradient ** 2

        # L2 normaliztion of weights
        phi_norm = torch.sqrt(torch.sum(parameter ** 2))
        
        # adam ratio, ratio of corrected mean and corrected variance stabilized with epsilon
        r_t = torch.sqrt(variance)
        r_t = mean / (torch.sqrt(variance) + self.eps)
        if self.weight_decay != 0:
            r_t += parameter * self.weight_decay
        r_t_norm = torch.sqrt(torch.sum(r_t ** 2))

        if phi_norm != 0 and r_t_norm != 0:
            trust_ratio = phi_norm / r_t_norm
        else:
            trust_ratio = torch.tensor([1.0])
        trust_ratio = torch.clamp(trust_ratio, self.clip_value[0], self.clip_value[1])

        updated_parameter = parameter - self.learning_rate * trust_ratio * r_t

        return updated_parameter


    def get_pytorch_optimizer(self, parameters: Dict[str, torch.Tensor], lr = None) -> torch.optim.Optimizer:
        """
        Return an equivalent pytorch optimizer, used for verification. 
        """
        if not self.torch_optimizer:
            self.torch_optimizer = pybuda.torch_optimizers.LAMB(
                params=[p for p in parameters.values()],
                lr=self.learning_rate,
                betas=(self.beta1, self.beta2),
                eps=self.eps,
                weight_decay=self.weight_decay                
            )
        return self.torch_optimizer



class LARS(Optimizer):
    """
    Implements LARS optimizer, Layer-Wise Adaptive Rate Scaling.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        momentum: float = 0.9,
        nesterov: bool = False,
        dampening: float = 0.0,
        weight_decay: float = 0.01,
        lars_coeff: float = 0.001,
        parameters: Optional[List[Parameter]] = None,
        device_params: bool = False,
        epsilon: float = 1e-8
    ):
        # LARS Optimizer Constants
        assert learning_rate >= 0.0, f"Invalid learning rate value: {learning_rate}"
        assert momentum >= 0.0, f"Invalid momentum value: {momentum}"
        assert weight_decay >= 0.0, f"Invalid weight decay: {weight_decay}"
        assert lars_coeff >= 0.0 and lars_coeff < 1.0, f"Invalid LARS coefficient: {lars_coeff}"

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.lars_coeff = lars_coeff
        self.epsilon = epsilon

        self.torch_optimizer = None

        self.parameter_to_opt_inputs: Dict[str, Dict[str, Tensor]] = {}
        self.parameter_to_opt_torch_inputs: Dict[str, Dict[str, Tensor]] = {}

        if device_params:
            assert parameters is None
        else:
            assert parameters is not None
            self.set_parameters_to_optimize(parameters)

        self.device_params = device_params


    def set_parameters_to_optimize(self, parameters: List[Parameter]):
        # For each Parameter, we register its associated set of optimizer parameters
        for parameter in parameters:
            if parameter.requires_grad:
                # PyBuda
                self.parameter_to_opt_inputs[
                    parameter.get_name()
                ] = self.get_param_dict(
                        parameter.pt_data_format, 
                        parameter.shape.get_pytorch_shape()
                    )
                # PyTorch
                self.parameter_to_opt_torch_inputs[
                    parameter.get_name()
                ] = self.get_cpu_param_dict(
                        parameter.pt_data_format, 
                        parameter.shape.get_pytorch_shape()
                    )


    def set_optimizer_parameters(self, learning_rate: Optional[float] = None) -> None:
        """
        Loop through every Parameter tensor with `requires_grad=True` and pushes
        `learning_rate` value to its associated optimizer parameter queue. By default,
        if no `learning_rate` is specified, the learning rate used to construct the
        sgd_optimizer is used.

        This method may be invoked multiple times as a way to
        adjust the updated optimizer parameter values across training loop iterations.

        Parameters
        ----------
        learning_rate : Optional[float]
            learning_rate

        Returns
        -------
        None

        """

        if learning_rate:
            self.learning_rate = learning_rate

        # {mean, variance} get updated in the loopback
        for parameter, opt_inputs in self.parameter_to_opt_inputs.items():
            torch_lr = torch.full(
                (1,), self.learning_rate, dtype=opt_inputs["lr"].pt_data_format
            )
            opt_inputs["lr"] = Tensor.create_from_torch(torch_lr)


    def get_optimizer_state_keys(self) -> List:
        return []


    def get_cpu_param_dict(self, dtype: torch.dtype, shape: Tuple[int]) -> Dict:
        return {
            "torch_momentum": torch.full(shape, 0.0, dtype=dtype),
        }


    def get_param_dict(self, dtype: torch.dtype, shape: Tuple[int]) -> Dict:
        torch_lr = torch.full((1,), self.learning_rate, dtype=dtype)
        return {
            "lr": Tensor.create_from_torch(torch_lr),
            "momentum": Tensor.create_from_torch(torch.full(shape, 0.0, dtype=dtype)),
        }


    def get_optimizer_params(self, parameter_name, is_buda) -> Optional[Dict[str, Tensor]]:

        if parameter_name not in self.parameter_to_opt_inputs:
            return None

        ret = copy.copy(self.parameter_to_opt_inputs[parameter_name])
        if is_buda:
            for k, v in ret.items():
                # optimize params should always tile broadcast if they are scalar
                one_d = len(ret[k].shape) == 1 and ret[k].shape[0] == 1
                tile_broadcast_dims = [-1, -2] if one_d else []
                ret[k] = v.to_buda_shape(
                    tile_broadcast_dims=tile_broadcast_dims, clone=True
                )
        return ret


    def get_type(self) -> str:
        return "lars"


    def generate_op_trace(self, ac, parameter, gradient):

        # parameter shape
        param_shape = parameter.shape.as_list()

        # gradinet buffering

        # g(t) -> gradient at current timestep
        #temp fix for circular dependency
        from pybuda.op.eval.pybuda.buffer import Buffer
        grad = ac.op(Buffer.create(), (gradient, ))

        # lambda <- || w(t) || / (|| g(t) || + beta * || w(t) ||)
        weight_norm = ac.op("multiply", (parameter, parameter))
        weight_norm_shape = weight_norm.shape.as_list()
        if len(weight_norm_shape) > 1:
            weight_norm = ac.op("reduce_sum", (weight_norm, ), (-2, ))
        weight_norm = ac.op("reduce_sum", (weight_norm, ), (-1, ))
        #importing locally to avoid circular dependency from Dataformats
        from pybuda.op.eval.pybuda.sqrt import Sqrt
        weight_norm = ac.op(Sqrt.create(), (weight_norm, ))

        grad_norm = ac.op("multiply", (grad, grad))
        grad_norm_shape = grad_norm.shape.as_list()
        if len(grad_norm_shape) > 1:
            grad_norm = ac.op("reduce_sum", (grad_norm, ), (-2, ))
        grad_norm = ac.op("reduce_sum", (grad_norm, ), (-1, ))
        grad_norm = ac.op(Sqrt.create(), (grad_norm, ))
 
        #
        #   IF weight_norm != 0 AND grad_norm != 0:
        #       local_lr = lars_coeff * weight_norm / (grad_norm + weight_decay * weight_norm + epsilon)
        #   ELSE
        #       local_lr = 1
        #
        # We don't support branching, so we compute both branches, let's decompose IF-ELSE block
        #
        #   IF weight_norm != 0:
        #       IF grad_norm != 0:
        #           local_lr = lars_coeff * weight_norm / (grad_norm + weight_decay * weight_norm + epsilon)
        #       ELSE
        #           local_lr = 1
        #   ELSE
        #       local_lr = 1
        #
        #
        # so, local learning rate is
        #   
        #   local_lr = lars_coeff * weight_norm / (grad_norm + weight_decay * weight_norm + epsilon)
        #   local_lr = (weight_norm != 0) * ((grad_norn != 0) * local_lr + (grad_norm == 0)) + (weight_norm == 0)
        #

        # Computation of conditions
        zero = ac.tensor(torch.zeros(param_shape))
        weight_norm_eq = ac.op("equal", (weight_norm, zero))
        weight_norm_ne = ac.op("not_equal", (weight_norm, zero))
        grad_norm_eq = ac.op("equal", (grad_norm, zero))
        grad_norm_ne = ac.op("not_equal", (grad_norm, zero))

        # Extracted parameters
        weight_decay = ac.tensor(torch.zeros(param_shape) + self.weight_decay)
        lars_coeff = ac.tensor(torch.zeros(param_shape) + self.lars_coeff)
        epsilon = ac.tensor(torch.zeros(param_shape) + self.epsilon)

        # Computing local learning rate without conditions
        local_learning_rate = ac.op("multiply", (weight_decay, weight_norm))
        local_learning_rate = ac.op("add", (grad_norm, local_learning_rate))
        local_learning_rate = ac.op("add", (epsilon, local_learning_rate))
        from pybuda.op.eval.pybuda.reciprocal import Reciprocal
        local_learning_rate = ac.op(Reciprocal.create(), (local_learning_rate, ))
        local_learning_rate = ac.op("multiply", (weight_norm, local_learning_rate))
        local_learning_rate = ac.op("multiply", (lars_coeff, local_learning_rate))

        # Computing local learning rate with conditions
        local_learning_rate = ac.op("multiply", (grad_norm_ne, local_learning_rate))
        local_learning_rate = ac.op("add", (grad_norm_eq, local_learning_rate))
        local_learning_rate = ac.op("multiply", (weight_norm_ne, local_learning_rate))
        local_learning_rate = ac.op("add", (weight_norm_eq, local_learning_rate))

        if self.weight_decay != 0:
            grad = ac.op("add", (grad, ac.op("multiply", (weight_decay, parameter))))

        # momentum = ac.input("momentum", parameter.shape)
        # if self.momentum != 0:
        #     # Get momentum and dampening from the input
        #     momentum_ = ac.constant(self.momentum)
        #     dampening = ac.constant(1 - self.dampening)
        #     # Compute current momentum based on the previous one and parameters
        #     current_momentum = ac.op("multiply", (momentum, momentum_))
        #     dampening_grad = ac.op("multiply", (dampening, grad))
        #     current_momentum = ac.op("add", (current_momentum, dampening_grad))
        #     # if self.nesterov:
        #     #     grad = ac.op("add", (grad, ac.op("multiply", (current_momentum, momentum_))))
        #     # else:
        #     #     grad = current_momentum
        # learning_rate = ac.constant(self.learning_rate)
        # updated_momentum = ac.op("multiply", (learning_rate, ac.op("multiply", (local_learning_rate, grad))))

        momentum = ac.input("momentum", parameter.shape)
        momentum_ = ac.tensor(torch.zeros(param_shape) + self.momentum)
        current_momentum = ac.op("multiply", (momentum, momentum_))
        learning_rate = ac.tensor(torch.zeros(param_shape) + self.learning_rate)
        updated_momentum = ac.op("add", (ac.op("multiply", (learning_rate, ac.op("multiply", (local_learning_rate, grad)))), current_momentum))

        # w(t + 1) <- w(t) - v(t + 1)
        updated_parameter = ac.op("subtract", (parameter, updated_momentum))

        ac.loopback(updated_momentum, momentum)

        return updated_parameter

    def torch_parameter_update(
        self, 
        *, 
        parameter_name: str, 
        parameter: torch.Tensor, 
        gradient: torch.Tensor
    ) -> torch.Tensor:
        
        torch_momentum = self.parameter_to_opt_torch_inputs[parameter_name]["torch_momentum"]

        weight_norm = torch.sqrt(torch.sum(parameter ** 2))
        gradient_norm = torch.sqrt(torch.sum(gradient ** 2))

        if weight_norm != 0 and gradient_norm != 0:
            local_learning_rate = self.lars_coeff * weight_norm / (gradient_norm + self.weight_decay * weight_norm + self.epsilon)
        else:
            local_learning_rate = 1

        if self.weight_decay != 0:
            gradient += self.weight_decay * parameter
            
        # if self.momentum != 0:
        #     previous_momentum = self.momentum * torch_momentum
        #     previous_momentum += (1 - self.dampening) * gradient
        #     if self.nesterov:
        #         gradient += previous_momentum * self.momentum
        #     else:
        #         gradient = previous_momentum
        # updated_parameter = parameter - self.learning_rate * local_learning_rate * gradient

        updated_momentum = self.momentum * torch_momentum + self.learning_rate * local_learning_rate * gradient
        updated_parameter = parameter - updated_momentum

        return updated_parameter


    def get_pytorch_optimizer(self, parameters: Dict[str, torch.Tensor], lr = None) -> torch.optim.Optimizer:
        """
        Return an equivalent pytorch optimizer, used for verification. 
        """
        if not self.torch_optimizer:
            self.torch_optimizer = pybuda.torch_optimizers.LARS(
                params=[p for p in parameters.values()],
                lr=self.learning_rate,
                momentum=self.momentum,
                lars_coeff=self.lars_coeff,
                weight_decay=self.weight_decay,
                dampening=self.dampening,
                nesterov=self.nesterov,
                eps=self.epsilon             
            )
        return self.torch_optimizer

