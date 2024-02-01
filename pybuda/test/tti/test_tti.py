# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from loguru import logger
from pybuda.device import Device
from pybuda.module import PyBudaModule
from pybuda.pybudaglobal import pybuda_reset
import pytest

import queue
import random
import torch
import os

import pybuda
import pybuda.op
from pybuda import PyTorchModule, TTDeviceImage, VerifyConfig
from pybuda._C.backend_api import BackendDevice, BackendType, DeviceMode, detect_available_silicon_devices
from transformers import BertModel, BertConfig
from ..common import ModuleBuilder, TestDevice, run, ModuleBuilder, device
from test.bert.modules import PyBudaFeedForward 
from pybuda.ttdevice import get_device_config
from pybuda.config import _get_global_compiler_config
from test.utils import download_model

class PyBudaTestModule(pybuda.PyBudaModule):
    shape = (1, 1, 32, 32)

    def __init__(self, name):
        super().__init__(name)
        self.weights1 = pybuda.Parameter(
            torch.rand(*PyBudaTestModule.shape, requires_grad=True)
        )
        self.weights2 = pybuda.Parameter(
            torch.rand(*PyBudaTestModule.shape, requires_grad=True)
        )

    def forward(self, act1, act2):
        m1 = pybuda.op.Matmul("matmul1", act1, self.weights1)
        m2 = pybuda.op.Matmul("matmul2", act2, self.weights2)

        add = pybuda.op.Add("add_mm", m1, m2)
        constant = pybuda.op.Constant("constant", constant=2.0)

        add_constant = pybuda.op.Add("add_constant", add, constant)

        return add_constant

    @staticmethod
    def _save_device_image(
        *,
        device_name,
        arch,
        backend_type,
        device_mode,
        module,
        input_shapes,
        target_shapes=tuple(),
        enable_training=False,
        enable_optimizer=False,
        num_chips=1,
        chip_ids=[]
    ):
        optimizer = (
            pybuda.optimizers.SGD(learning_rate=0.1, device_params=True)
            if enable_optimizer
            else None
        )

        # Create a TT device
        tt0 = pybuda.TTDevice(
            get_device_name(device_name, backend_type),
            arch=arch, 
            devtype=backend_type,
            optimizer=optimizer,
            num_chips=num_chips,
            chip_ids=chip_ids,
        )

        # Place a module on the device
        tt0.place_module(module)

        if enable_training:
            tt0.place_loss_module(pybuda.op.loss.L1Loss("l1_loss"))

        sample_inputs = [torch.rand(*shape) for shape in input_shapes]
        sample_targets = [torch.rand(*shape) for shape in target_shapes]

        if device_mode == DeviceMode.CompileAndRun and enable_training:
            loss_q, checkpoint_q = mp_context.Queue(), mp_context.Queue()
            tt0.push_to_inputs(sample_inputs)
            tt0.push_to_target_inputs(sample_targets)
            pybuda.run_training(checkpoint_queue=checkpoint_q, loss_queue=loss_q)
        elif device_mode == DeviceMode.CompileAndRun:
            tt0.push_to_inputs(sample_inputs)
            output_q = pybuda.run_inference()
            output = _safe_read(output_q)

        # save device_image
        device_img = tt0.compile_to_image(
            img_path=f"device_images/{get_device_name(device_name, backend_type)}.tti",
            training=enable_training,
            sample_inputs=sample_inputs,
            sample_targets=sample_targets,
        )
        pybuda_reset()  # NB: note the reset; invoke to clear the global state that lingers around
        return device_img

    @staticmethod
    def _load_device_image(
        device_name, backend_type, *, set_module_params=False, enable_training=False
    ):
        # load device_image
        img_path = f"device_images/{get_device_name(device_name, backend_type)}.tti"
        img = TTDeviceImage.load_from_disk(img_path)
        tt1 = pybuda.TTDevice.load_image(img=img)

        if set_module_params:
            module = tt1.modules[-1]
            module.set_parameter(
                "weights1", torch.rand(*PyBudaTestModule.shape, requires_grad=True)
            )
            module.set_parameter(
                "weights2", torch.rand(*PyBudaTestModule.shape, requires_grad=True)
            )

        loss_q = mp_context.Queue()
        checkpoint_q = mp_context.Queue()

        inputs = [torch.rand(shape) for shape in img.get_input_shapes()]
        tt1.push_to_inputs(inputs)

        if enable_training:
            targets = [torch.rand(shape) for shape in img.get_target_shapes()]
            tt1.push_to_target_inputs(*targets)
            pybuda.run_training(checkpoint_queue=checkpoint_q, loss_queue=loss_q)
            print("checkpoint: ", _safe_read(checkpoint_q))
            print("loss: ", _safe_read(loss_q))
        else:
            output_q = pybuda.run_inference()
            output = _safe_read(output_q)
            print(output)


class MyLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(64, 128, bias=True)

    def forward(self, act):
        x = self.lin(act)
        return x

class MatMulRelu(PyBudaModule):
    shape = (1, 128, 128)
    def __init__(self, name):
        super().__init__(name)
        self.weights1 = pybuda.Parameter(
            torch.rand(*MatMulRelu.shape, requires_grad=True)
        )

    def forward(self, act):
        matmul = pybuda.op.Matmul("matmul1", act, self.weights1)
        relu = pybuda.op.Relu(f"relu", matmul)
        return relu 


def get_bert_encoder_module():
    config = download_model(BertConfig.from_pretrained, "prajjwal1/bert-tiny")
    config.num_hidden_layers = 1
    model = BertModel(config=config)
    module = PyTorchModule("bert_encoder", model.encoder)
    return module


# https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork
mp_context = torch.multiprocessing.get_context("spawn")


def _safe_read(q):
    """
    Read a queue, but return None if an error was raised in the meantime, preventing a hang on error.
    """
    while True:
        try:
            data = q.get(timeout=0.5)
            return data
        except queue.Empty as _:
            if pybuda.error_raised():
                raise RuntimeError("Error raised in pybuda")
        except KeyboardInterrupt:
            return None


def get_device_name(device_name, backend_type: BackendType):
    if backend_type == BackendType.Silicon:
        backend_type_suffix = "silicon"
    else:
        backend_type_suffix = "golden"
    return f"{device_name}_{backend_type_suffix}"


@pytest.fixture
def pybuda_module():
    return PyBudaTestModule("test_pybuda_module")

def test_inference_compile_to_image_and_run_then_rerun_from_image(test_device, pybuda_module):
    pybuda_module._save_device_image(
        device_name = "tt0",
        arch = test_device.arch,
        backend_type = BackendType.Golden,
        device_mode = DeviceMode.CompileAndRun,
        module = pybuda_module,
        input_shapes = [(4, 32, 32), (4, 32, 32)],
    )
    pybuda_module._load_device_image("tt0", BackendType.Golden)    
    
def test_inference_compile_only_then_run_from_image(test_device, pybuda_module):
    pybuda_module._save_device_image(
        device_name = "tt1",
        arch = test_device.arch,
        backend_type = BackendType.Golden,
        device_mode = DeviceMode.CompileOnly,
        module = pybuda_module,
        input_shapes = [(4, 32, 32), (4, 32, 32)],
    )
    pybuda_module._load_device_image("tt1", BackendType.Golden)


def test_inference_compile_only_silicon_target_device(test_device, pybuda_module):
    pybuda_module._save_device_image(
        device_name = "tt2",
        arch = test_device.arch,
        backend_type = BackendType.Silicon,
        device_mode = DeviceMode.CompileOnly,
        module = pybuda_module,
        input_shapes = [(4, 32, 32), (4, 32, 32)],
    )

@pytest.mark.parametrize("num_harvested_rows", [x+1 for x in range(3)])
def test_inference_compile_only_silicon_target_device_harvested_manual(test_device, num_harvested_rows, pybuda_module):     
    compiler_cfg = _get_global_compiler_config()
    dev_cfg = get_device_config(test_device.arch, [0], compiler_cfg.backend_cluster_descriptor_path, compiler_cfg.backend_runtime_params_path, compiler_cfg.store_backend_db_to_yaml, BackendType.Golden)
    if num_harvested_rows < 10 - dev_cfg.grid_size.r:
        pytest.skip("Simulated harveted rows are less than actually harvested rows on the silicon")    
    
    detected_harvested_rows = []
    harvesting_rows_available = [1,2,3,4,5,7,8,9,10,11]
    if dev_cfg.grid_size.r < 10:
        row_coordinate = 0
        harvesting_mask = dev_cfg.get_harvested_cfg()
        while harvesting_mask: 
            if (harvesting_mask & 1):
                detected_harvested_rows.append(row_coordinate)
                harvesting_rows_available.remove(row_coordinate)
            harvesting_mask = (harvesting_mask >> 1)
            row_coordinate += 1 
    harvested_rows = random.sample(harvesting_rows_available, num_harvested_rows-len(detected_harvested_rows))+detected_harvested_rows
    pybuda.set_configuration_options(harvested_rows=harvested_rows) 
    device_image = pybuda_module._save_device_image(
        device_name = "tt2-harvested-manual",
        arch = test_device.arch,
        backend_type = BackendType.Silicon,
        device_mode = DeviceMode.CompileOnly,
        module = pybuda_module,
        input_shapes = [(4, 32, 32), (4, 32, 32)],
    )
    pybuda_reset() 
    device_image.info()


@pytest.mark.skip(reason="currently local testing only")
def test_inference_compile_only_silicon_target_device_harvested_auto(test_device, pybuda_module):
    device_image = pybuda_module._save_device_image(
        device_name = "tt2-harvested-auto",
        arch = test_device.arch,
        backend_type = BackendType.Golden,
        device_mode = DeviceMode.CompileOnly,
        module = pybuda_module,
        input_shapes = [(4, 32, 32), (4, 32, 32)],
    ) 
    device_image.info()


def test_inference_compile_only_module_params_unset_on_save(test_device, pybuda_module):
    pybuda_module._save_device_image(
        device_name = "tt3",
        arch = test_device.arch,
        backend_type = BackendType.Golden,
        device_mode = DeviceMode.CompileOnly,
        module = pybuda_module,
        input_shapes = [(4, 32, 32), (4, 32, 32)],
    )
    pybuda_module._load_device_image("tt3", BackendType.Golden, set_module_params=True)


def test_inference_output_host_tms(test_device, pybuda_module):
    """
    def gelu_relu(act):
        op0 = pybuda.op.Gelu(f"op0", act)
        op1 = pybuda.op.Relu(f"op1", op0)
        return op1
    module = ModuleBuilder(gelu_relu)
    """
    module = MatMulRelu("matmul_relu")

    pybuda_module._save_device_image(
        device_name = "output_host_tms",
        arch = test_device.arch,
        backend_type = BackendType.Golden,
        device_mode = DeviceMode.CompileOnly,
        module = module,
        input_shapes = [(1, 128, 128),],
    )
    pybuda_module._load_device_image("output_host_tms", BackendType.Golden)


def test_training_compile_to_image_and_run_then_rerun_from_image(test_device, pybuda_module):
    pybuda_module._save_device_image(
        device_name = "tt4",
        arch = test_device.arch,
        backend_type = BackendType.Golden,
        device_mode = DeviceMode.CompileAndRun,
        enable_training = True,
        module = pybuda_module,
        input_shapes = [(4, 32, 32), (4, 32, 32)],
        target_shapes = [
            (4, 32, 32),
        ],
    )
    pybuda_module._load_device_image("tt4", BackendType.Golden, enable_training=True)


def test_training_compile_only_then_run_from_image(test_device, pybuda_module):
    pybuda_module._save_device_image(
        device_name = "tt5",
        arch = test_device.arch,
        backend_type = BackendType.Golden,
        device_mode = DeviceMode.CompileOnly,
        enable_training = True,
        module = pybuda_module,
        input_shapes = [(4, 32, 32), (4, 32, 32)],
        target_shapes = [
            (4, 32, 32),
        ],
    )
    pybuda_module._load_device_image("tt5", BackendType.Golden, enable_training=True)


def test_training_compile_only_silicon_target_device(test_device, pybuda_module):
    pybuda_module._save_device_image(
        device_name = "tt6",
        arch = test_device.arch,
        backend_type = BackendType.Silicon,
        device_mode = DeviceMode.CompileOnly,
        module = pybuda_module,
        input_shapes = [(4, 32, 32), (4, 32, 32)],
    )


def test_training_compile_only_then_run_from_image_with_optimizer(test_device, pybuda_module):
    pybuda_module._save_device_image(
        device_name = "tt7",
        arch = test_device.arch,
        backend_type = BackendType.Golden,
        device_mode = DeviceMode.CompileOnly,
        enable_optimizer = True,
        enable_training = True,
        module = pybuda_module,
        input_shapes = [(4, 32, 32), (4, 32, 32)],
        target_shapes = [
            (4, 32, 32),
        ],
    )
    pybuda_module._load_device_image("tt7", BackendType.Golden, enable_training=True)


def test_device_image_apis(test_device, pybuda_module):
    inference_img = pybuda_module._save_device_image(
        device_name = "tt_inference",
        arch = test_device.arch,
        backend_type = BackendType.Golden,
        device_mode = DeviceMode.CompileOnly,
        module = pybuda_module,
        input_shapes = [(4, 32, 32), (4, 32, 32)],
    )
    training_img = pybuda_module._save_device_image(
        device_name = "tt_training",
        arch = test_device.arch,
        backend_type = BackendType.Golden,
        device_mode = DeviceMode.CompileOnly,
        enable_training = True,
        module = pybuda_module,
        input_shapes = [(4, 32, 32), (4, 32, 32)],
        target_shapes = [
            (4, 32, 32),
        ],
    )

    assert not inference_img.is_compiled_for_training()
    assert training_img.is_compiled_for_training()


def test_const_eval_save_and_load(test_device, pybuda_module):
    pybuda_module._save_device_image(
        device_name = "tt9",
        arch = test_device.arch,
        backend_type = BackendType.Golden,
        device_mode = DeviceMode.CompileAndRun,
        module = PyTorchModule("pt_linear", MyLinear()),
        input_shapes = [
            (1, 128, 64),
        ],
    )
    pybuda_module._load_device_image("tt9", BackendType.Golden)


def test_pt_encoder_silicon_save_and_inspect(test_device, pybuda_module):
    device_img = pybuda_module._save_device_image(
        device_name = "tt10",
        arch = test_device.arch,
        backend_type = BackendType.Silicon,
        device_mode = DeviceMode.CompileOnly,
        module = get_bert_encoder_module(),
        input_shapes = [(1, 128, 128), (1, 1, 128, 128)],
    )
    device_img.info()


def test_pt_encoder_golden_save_and_load(test_device, pybuda_module):
    pybuda_module._save_device_image(
        device_name = "tt11",
        arch = test_device.arch,
        backend_type = BackendType.Golden,
        device_mode = DeviceMode.CompileOnly,
        module = get_bert_encoder_module(),
        input_shapes = [(1, 128, 128), (1, 1, 128, 128)],
    )
    pybuda_module._load_device_image("tt11", BackendType.Golden)


def test_tti_buffer_queue(test_device, pybuda_module):
    pybuda.config.set_configuration_options()
    pybuda.config.override_op_size("matmul_22", (2,2))
    pybuda.config.insert_buffering_nop("matmul_22", "matmul_29", hoist_tms = False)
    pybuda.config.override_dram_queue_placement("layer.0.attention.self.query.weight", channel=4)
    pybuda_module._save_device_image(
        device_name = "tti_buffer_queue",
        arch = test_device.arch,
        backend_type = test_device.devtype,
        device_mode = DeviceMode.CompileOnly,
        module = get_bert_encoder_module(),
        input_shapes = [(1, 128, 128), (1, 1, 128, 128)],
    )
    pybuda_module._load_device_image("tti_buffer_queue", test_device.devtype)


class simple_matmul(PyBudaModule):
        def __init__(self, name):
            super().__init__(name)
            self.weights = pybuda.Parameter(*(1,1,32,128), requires_grad=True)

        def forward(self, x):
            x = pybuda.op.Matmul("", x, self.weights)
            return x

@pytest.mark.parametrize("two_rows_harvested", [True, False])
def test_tti_save_load_verify_module(test_device, two_rows_harvested):
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip("Under development")

    tt0 = simple_matmul("test_tti_save_load_verify_module")
    input_shape = (1, 1, 32, 32)
    pybuda.verify.verify_module(
        tt0, 
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=pybuda.verify.config.TestKind.INFERENCE,
        ),
    )


def test_tti_bert_encoder(test_device):
    input_shapes = [(1, 128, 128), (1, 1, 128, 128)],
    pybuda.verify.verify_module(
        get_bert_encoder_module(), 
        *input_shapes,
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=pybuda.verify.config.TestKind.INFERENCE,
        ),
    )

if __name__ == "__main__":
    import os

    os.environ["LOGURU_LEVEL"] = "TRACE"

    test_pt_encoder_golden_save_and_load()
