# GoogLeNet Demo Script

import urllib

import pybuda
import requests
import torch
from PIL import Image
from pybuda._C.backend_api import BackendDevice
from torchvision import models, transforms


def run_googlenet_pytorch():

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    # Device specific configurations
    available_devices = pybuda.detect_available_devices()
    if available_devices:
        if available_devices[0] == BackendDevice.Wormhole_B0:
            compiler_cfg.balancer_op_override(
                "max_pool2d_125.dc.sparse_matmul.5.dc.sparse_matmul.1.lc2",
                "t_stream_shape",
                (1, 1),
            )
            compiler_cfg.balancer_op_override(
                "max_pool2d_294.dc.sparse_matmul.5.dc.sparse_matmul.1.lc2",
                "t_stream_shape",
                (1, 1),
            )
            compiler_cfg.balancer_op_override(
                "max_pool2d_546.dc.sparse_matmul.5.dc.sparse_matmul.1.lc2",
                "t_stream_shape",
                (1, 1),
            )

    # Create PyBuda module from PyTorch model
    # Two ways to load the same model
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
    model = models.googlenet(pretrained=True)
    model.eval()
    tt_model = pybuda.PyTorchModule("pt_googlenet", model)

    # Image preprocessing
    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    input_image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=([input_batch]))
    output = output_q.get(timeout=0.5)[0].value()

    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get ImageNet class mappings
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    image_classes = urllib.request.urlopen(url)
    categories = [s.decode("utf-8").strip() for s in image_classes.readlines()]

    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    result = {}
    for i in range(top5_prob.size(0)):
        result[categories[top5_catid[i]]] = top5_prob[i].item()
    print(result)


if __name__ == "__main__":
    run_googlenet_pytorch()
