# Convolutional Autoencoder Demo Script

import pybuda
import torch
import torchvision.transforms as transforms
from datasets import load_dataset


class ConvAE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder_conv2d_1 = torch.nn.Conv2d(1, 16, 3, padding=1)
        self.encoder_conv2d_2 = torch.nn.Conv2d(16, 4, 3, padding=1)
        self.encoder_max_pool2d = torch.nn.MaxPool2d(2, 2)

        # Decoder
        self.decoder_conv2d_1 = torch.nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.decoder_conv2d_2 = torch.nn.ConvTranspose2d(16, 1, 2, stride=2)

        # Activation Function
        self.act_fun = torch.nn.ReLU()

    def forward(self, x):
        # Encode
        act = self.encoder_conv2d_1(x)
        act = self.act_fun(act)
        act = self.encoder_max_pool2d(act)
        act = self.encoder_conv2d_2(act)
        act = self.act_fun(act)
        act = self.encoder_max_pool2d(act)

        # Decode
        act = self.decoder_conv2d_1(act)
        act = self.act_fun(act)
        act = self.decoder_conv2d_2(act)

        return act


def run_conv_ae_pytorch():

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    # Instantiate model
    # NOTE: The model has not been pre-trained or fine-tuned.
    # This is for demonstration purposes only.
    model = ConvAE()

    # Define transform to normalize data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # Load sample from MNIST dataset
    dataset = load_dataset("mnist")
    sample = dataset["train"][0]["image"]
    sample_tensor = transform(sample).unsqueeze(0)

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(
        pybuda.PyTorchModule("pt_conv_ae", model),
        inputs=[sample_tensor],
    )
    output = output_q.get()

    # Print output
    print("Output:", output[0].value())


if __name__ == "__main__":
    run_conv_ae_pytorch()
