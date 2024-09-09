import torch
import pybuda
import pytest

from loguru import logger

from pybuda.verify import verify_module, VerifyConfig

from test.operators.utils import FailingReasons


class Matmul2ModelPyBuda(pybuda.PyBudaModule):
    '''PyBuda model with two matmul operations'''

    def __init__(self):
        super(Matmul2ModelPyBuda, self).__init__("Buda Test 1")

    def forward(self, in_value1: pybuda.Tensor, in_value2: pybuda.Tensor, in_value3: pybuda.Tensor) -> torch.Tensor:
        
        v = pybuda.op.Matmul("op1", in_value1, in_value2)
        v = pybuda.op.Matmul("op2", v, in_value3)

        return v


class Matmul1ModelPyTorch(torch.nn.Module):
    '''PyTorch model with only one matmul operation'''

    def __init__(self):
        super(Matmul1ModelPyTorch, self).__init__()

    def forward(self, in_value1: torch.Tensor, in_value2: torch.Tensor) -> torch.Tensor:
        
        v = torch.matmul(in_value1, in_value2)

        return v


class Matmul2ModelPyTorch(torch.nn.Module):
    '''PyTorch model with two matmul operations'''

    def __init__(self):
        super(Matmul2ModelPyTorch, self).__init__()

    def forward(self, in_value1: torch.Tensor, in_value2: torch.Tensor, in_value3: torch.Tensor) -> torch.Tensor:
        
        v = torch.matmul(in_value1, in_value2)
        v = torch.matmul(v, in_value3)

        return v


class MatmulAddModelPyTorch(torch.nn.Module):
    '''PyTorch model with add and matmul operations'''

    def __init__(self):
        super(MatmulAddModelPyTorch, self).__init__()

    def forward(self, in_value1: torch.Tensor, in_value2: torch.Tensor, in_value3: torch.Tensor) -> torch.Tensor:
        
        v = torch.add(in_value1, in_value2)
        v = torch.matmul(v, in_value3)

        return v


@pytest.mark.parametrize("model_type, input_shapes", [

    # always pass if only 1 matmul is in pytorch model

    pytest.param(Matmul1ModelPyTorch, [
        (1, 5, 3),
        (1, 3, 2),
    ]),

    pytest.param(Matmul1ModelPyTorch, [
        (1, 64, 32),
        (1, 32, 64),
    ]),

    pytest.param(Matmul1ModelPyTorch, [
        (1, 2, 5, 3),
        (1, 2, 3, 2),
    ]),

    # if fails via PyTorch when operand source for matmul is another matmul and inputs are 3-dimensional tensors

    # 3-dimensional tensors are not working via pytorch if operand source is another matmul ?
    pytest.param(Matmul2ModelPyTorch, [
        (1, 5, 3),
        (1, 3, 2),
        (1, 2, 4),
    ], marks=pytest.mark.xfail(reason=FailingReasons.UNSUPPORTED_DIMENSION)),
    # Errors:
    #  - Failed on "ExplicateTranspose" TVM callback
    #  - ValueError: The type checker has not populated the checked_type for this node

    # 3-dimensional tensors are not working via pytorch if operand source is another matmul ?
    # size of shpae dimensions is not important
    pytest.param(Matmul2ModelPyTorch, [
        (1, 64, 32),
        (1, 32, 64),
        (1, 64, 128),
    ], marks=pytest.mark.xfail(reason=FailingReasons.UNSUPPORTED_DIMENSION)),
    # Errors:
    #  - Failed on "ExplicateTranspose" TVM callback
    #  - ValueError: The type checker has not populated the checked_type for this node

    # 4-dimensional tensors are working fine via pytorch
    pytest.param(Matmul2ModelPyTorch, [
        (1, 2, 5, 3),
        (1, 2, 3, 2),
        (1, 2, 2, 4),
    ]),

    # always pass in PyBuda even operand source for matmul is another matmul

    pytest.param(Matmul2ModelPyBuda, [
        (1, 5, 3),
        (1, 3, 2),
        (1, 2, 4),
    ]),

    pytest.param(Matmul2ModelPyBuda, [
        (1, 64, 32),
        (1, 32, 64),
        (1, 64, 128),
    ]),

    pytest.param(Matmul2ModelPyBuda, [
        (1, 2, 5, 3),
        (1, 2, 3, 2),
        (1, 2, 2, 4),
    ]),

    # it pass also via PyTorch if operand source is another operator other than matmul, like add

    pytest.param(MatmulAddModelPyTorch, [
        (1, 5, 2),
        (1, 5, 2),
        (1, 2, 4),
    ]),

])
def test_matmul_shapes_3dim(model_type: torch.nn.Module, input_shapes, test_device):

    input_shapes_str = "_".join(["x".join(map(str, list(input_shape))) for input_shape in input_shapes])
    modelname = f"test_matmul_shapes_3dim_{model_type.__name__}_{input_shapes_str}"
    print(f"modelname = {modelname}")

    if issubclass(model_type, torch.nn.Module):
        # instantiate pytorch model first
        pytorch_model = model_type()

        # input tensors for pytorch
        in_values = [torch.randn(input_shapes1) for input_shapes1 in input_shapes]

        for index, in_value in enumerate(in_values):
            logger.debug(f"in_values[{index}]: {in_value.shape}")

        # invoke pytorch directly
        pytorch_result = pytorch_model(*in_values)

        logger.debug(f"pytorch_result: {pytorch_result.shape}")

        assert True, "Always passing pytorch invoke"

        # instantiate pybuda model from pytorch model
        model = pybuda.PyTorchModule(modelname, pytorch_model)
    elif issubclass(model_type, pybuda.PyBudaModule):
        # instantiate pybuda model directly
        model = model_type()
        model.name = modelname
    else:
        assert False, "Unknown model type"

    # invoke via pybuda
    verify_module(model, input_shapes, VerifyConfig(devtype=test_device.devtype, arch=test_device.arch))
