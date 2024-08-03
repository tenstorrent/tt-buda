import torch
from torch import nn

from utils import (
    MNISTLinear,
    Identity,
    load_tb_writer,
    load_dataset,
)


def main():
    torch.manual_seed(0)

    # Training configurations
    num_steps = 200
    batch_size = 1
    learning_rate = 0.01

    # Load dataset
    test_loader, train_loader = load_dataset(batch_size)

    # Load TensorBoard writer
    writer = load_tb_writer()

    # Dataset sample input
    sample_input = test_loader.dataset[0][0].repeat(batch_size, 1)
    sample_target = (
        nn.functional.one_hot(torch.tensor(test_loader.dataset[0][1]), num_classes=10)
        .float()
        .repeat(batch_size, 1)
    )

    # Initialize model
    framework_model = MNISTLinear()

    # Initialize optimizer and loos function
    optimizer = torch.optim.SGD(framework_model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.L1Loss()

    # Training loop
    for step in range(num_steps):
        optimizer.zero_grad()

        outputs = framework_model(sample_input)

        loss = loss_fn(outputs, sample_target)
        loss.backward()

        optimizer.step()

        # Log loss
        writer.add_scalar("Loss/PyTorch/overfit", loss.item(), step)


if __name__ == "__main__":
    main()
