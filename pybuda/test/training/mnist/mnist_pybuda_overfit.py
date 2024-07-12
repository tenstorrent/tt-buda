import torch
import pytest
from torch import nn

import pybuda
from pybuda import (
    CPUDevice,
    PyTorchModule,
)
from .utils import (
    MNISTLinear,
    Identity,
    load_tb_writer,
    load_dataset,
)
from pybuda.config import _get_global_compiler_config


def main(loss_on_cpu=True):
    torch.manual_seed(0)

    # Config
    num_steps = 250
    batch_size = 1
    learning_rate = 0.01
    sequential = True

    # Load dataset
    test_loader, train_loader = load_dataset(batch_size)

    # Load TensorBoard writer
    writer = load_tb_writer()

    # Dataset sample input
    sample_input = (test_loader.dataset[0][0].repeat(batch_size, 1),)
    sample_target = (
        nn.functional.one_hot(torch.tensor(test_loader.dataset[0][1]), num_classes=10)
        .float()
        .repeat(batch_size, 1)
    )

    # Initialize model
    framework_model = MNISTLinear()
    tt_model = pybuda.PyTorchModule("mnist_linear", framework_model)

    tt_optimizer = pybuda.optimizers.SGD(
        learning_rate=learning_rate, device_params=True
    )
    tt0 = pybuda.TTDevice("tt0", module=tt_model, optimizer=tt_optimizer)

    if loss_on_cpu:
        cpu0 = CPUDevice("cpu0", module=PyTorchModule("identity", Identity()))
        cpu0.place_loss_module(pybuda.PyTorchModule("l1_loss", torch.nn.MSELoss()))
    else:
        tt_loss = pybuda.PyTorchModule("l1_loss", torch.nn.MSELoss())
        tt0.place_loss_module(tt_loss)

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_auto_fusing = False

    if not loss_on_cpu:
        sample_target = (sample_target,)

    checkpoint_queue = pybuda.initialize_pipeline(
        training=True,
        sample_inputs=sample_input,
        sample_targets=sample_target,
        _sequential=sequential,
    )

    for step in range(num_steps):
        tt0.push_to_inputs(sample_input)
        if loss_on_cpu:
            cpu0.push_to_target_inputs(sample_target)
        else:
            tt0.push_to_target_inputs(sample_target)

        pybuda.run_forward(input_count=1, _sequential=sequential)
        pybuda.run_backward(input_count=1, zero_grad=True, _sequential=sequential)
        pybuda.run_optimizer(checkpoint=True, _sequential=sequential)

    loss_q = pybuda.run.get_loss_queue()

    step = 0
    while not loss_q.empty():
        if loss_on_cpu:
            writer.add_scalar("Loss/PyBuda/overfit", loss_q.get()[0], step)
        else:
            writer.add_scalar("Loss/PyBuda/overfit", loss_q.get()[0].value()[0], step)
        step += 1


loss_on_cpu = [True, False]
@pytest.mark.parametrize("loss_on_cpu", loss_on_cpu, ids=loss_on_cpu)
def test_mnist_pybuda_overfit(loss_on_cpu):
    main(loss_on_cpu)


if __name__ == "__main__":
    main()
