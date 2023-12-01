# Linear Autoencoder Demo Script

import pybuda
import torch
import torchvision.transforms as transforms
from datasets import load_dataset


class LinearAE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder_lin1 = torch.nn.Linear(784, 128)
        self.encoder_lin2 = torch.nn.Linear(128, 64)
        self.encoder_lin3 = torch.nn.Linear(64, 12)
        self.encoder_lin4 = torch.nn.Linear(12, 3)

        # Decoder
        self.decoder_lin1 = torch.nn.Linear(3, 12)
        self.decoder_lin2 = torch.nn.Linear(12, 64)
        self.decoder_lin3 = torch.nn.Linear(64, 128)
        self.decoder_lin4 = torch.nn.Linear(128, 784)

        # Activation Function
        self.act_fun = torch.nn.ReLU()

    def forward(self, x):
        # Encode
        act = self.encoder_lin1(x)
        act = self.act_fun(act)
        act = self.encoder_lin2(act)
        act = self.act_fun(act)
        act = self.encoder_lin3(act)
        act = self.act_fun(act)
        act = self.encoder_lin4(act)

        # Decode
        act = self.decoder_lin1(act)
        act = self.act_fun(act)
        act = self.decoder_lin2(act)
        act = self.act_fun(act)
        act = self.decoder_lin3(act)
        act = self.act_fun(act)
        act = self.decoder_lin4(act)

        return act


def run_linear_ae_pytorch():

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    # Instantiate model
    # NOTE: The model has not been pre-trained or fine-tuned.
    # This is for demonstration purposes only.
    model = LinearAE()

    # Define transform to normalize data
    transform = transforms.Compose(
        [
            transforms.Resize((1, 784)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # Load sample from MNIST dataset
    dataset = load_dataset("mnist")
    sample = dataset["train"][0]["image"]
    sample_tensor = transform(sample).squeeze(0)

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(
        pybuda.PyTorchModule("pt_linear_ae", model),
        inputs=[sample_tensor],
    )
    output = output_q.get()

    # Print output
    print("Output:", output[0].value())


if __name__ == "__main__":
    run_linear_ae_pytorch()
