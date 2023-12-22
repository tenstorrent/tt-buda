# ResNet Demo Script - ONNX
# Uses torch and torchvision for data pre- and post-processing;
# can use other frameworks such as MXNet, TensorFlow or Numpy

import os
import urllib

import onnx
import pybuda
import requests
import torch
from PIL import Image
from torchvision import transforms


def preprocess(image: Image) -> torch.tensor:
    """Image preprocessing for ResNet50

    Parameters
    ----------
    image : PIL.Image
        PIL Image sample

    Returns
    -------
    torch.tensor
        Preprocessed input tensor
    """
    transform_fn = transforms.Compose(
        [
            transforms.Resize([256, 256]),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    pixel_values = transform_fn(image).unsqueeze(0)

    return pixel_values


def postprocess(predictions: torch.tensor) -> tuple:
    """Model prediction postprocessing for ResNet50

    Parameters
    ----------
    predictions : torch.tensor
        Model predictions

    Returns
    -------
    tuple
        topk probability and category ID
    """

    # Get probabilities
    probabilities = torch.nn.functional.softmax(predictions, dim=0)

    # Get top-k prediction
    top1_prob, top1_catid = torch.topk(probabilities, 1)

    return top1_prob, top1_catid


def run_resnet_onnx():

    # Download model weights
    url = "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-7.onnx?download="
    load_path = "cv_demos/resnet/" + url.split("/")[-1]
    response = requests.get(url, stream=True)
    with open(load_path, "wb") as f:
        f.write(response.content)

    # Load ResNet feature extractor and model checkpoint from HuggingFace
    model = onnx.load(load_path)

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.enable_auto_fusing = False
    os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

    # Load data sample
    url = "https://datasets-server.huggingface.co/assets/imagenet-1k/--/default/train/18/image/image.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    label = "tiger"

    # Data preprocessing
    pixel_values = preprocess(image)

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(
        pybuda.OnnxModule("onnx_resnet50", model, load_path),
        inputs=[(pixel_values,)],
    )
    output = output_q.get()

    # Data postprocessing
    top1_prob, top1_catid = postprocess(output[0].value()[0])

    # Get ImageNet class mappings
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    image_classes = urllib.request.urlopen(url)
    categories = [s.decode("utf-8").strip() for s in image_classes.readlines()]
    predicted_label = categories[top1_catid]

    # Results
    print(f"True Label: {label} | Predicted Label: {predicted_label} | Predicted Probability: {top1_prob.item():.2f}")

    # Remove weight file
    os.remove(load_path)


if __name__ == "__main__":
    run_resnet_onnx()
