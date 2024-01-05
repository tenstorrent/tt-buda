# LW-OpenPose 2D Demo Script

import pybuda
import requests
from PIL import Image
from pytorchcv.model_provider import get_model as ptcv_get_model
from torchvision import transforms


def get_image_tensor():
    # Image processing
    url = "https://raw.githubusercontent.com/axinc-ai/ailia-models/master/pose_estimation_3d/blazepose-fullbody/girl-5204299_640.jpg"
    input_image = Image.open(requests.get(url, stream=True).raw)
    preprocess = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    return input_batch


def run_lwopenpose_3d_osmr_pytorch():

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.enable_auto_fusing = False
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16

    # Create PyBuda module from PyTorch model
    model = ptcv_get_model("lwopenpose3d_mobilenet_cmupan_coco", pretrained=True)
    model.eval()
    tt_model = pybuda.PyTorchModule("pt_lwopenpose_3d_osmr", model)

    # Get sample input
    input_batch = get_image_tensor()

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=([input_batch]))
    output = output_q.get()[0].value()

    # Print output
    print(output)


if __name__ == "__main__":
    run_lwopenpose_3d_osmr_pytorch()
