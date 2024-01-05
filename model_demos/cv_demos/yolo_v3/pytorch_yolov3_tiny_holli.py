import os

import pybuda
import requests
from PIL import Image

from cv_demos.yolo_v3.holli_src import utils
from cv_demos.yolo_v3.holli_src.yolo_layer import *
from cv_demos.yolo_v3.holli_src.yolov3_tiny import *


def run_yolov3_tiny_holli_pytorch():

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.enable_auto_fusing = False
    compiler_cfg.default_df_override = pybuda._C.Float16_b

    # Download model weights
    url = "https://www.ollihuotari.com/data/yolov3_pytorch/yolov3_tiny_coco_01.h5"
    load_path = "cv_demos/yolo_v3/" + url.split("/")[-1]
    response = requests.get(url, stream=True)
    with open(load_path, "wb") as f:
        f.write(response.content)

    # Load model
    model = Yolov3Tiny(num_classes=80, use_wrong_previous_anchors=True)
    model.load_state_dict(
        torch.load(
            load_path,
            map_location=torch.device("cpu"),
        )
    )
    model.eval()

    # Create PyBuda module from PyTorch model
    tt_model = pybuda.PyTorchModule("pytorch_yolov3_tiny_holli", model)

    sz = 512
    image_url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    img_org = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    img_resized = img_org.resize((sz, sz))
    img_tensor = utils.image2torch(img_resized)

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=([img_tensor]))
    output = output_q.get()
    print(output)

    # Remove weight file
    os.remove(load_path)


if __name__ == "__main__":
    run_yolov3_tiny_holli_pytorch()
