# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig, PyTorchModule
from pybuda._C.backend_api import BackendType, BackendDevice
from pybuda.verify.config import TestKind

import pybuda
import os

import torch
import torchvision

from PIL import Image
from torchvision import transforms
torch.multiprocessing.set_sharing_strategy("file_system") 
 
def get_image():
    os.system(
        "wget -nc https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    )
    torch.hub.download_url_to_file(
        "https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg"
    )
    input_image = Image.open("dog.jpg")
    preprocess = transforms.Compose(
        [
            transforms.Resize(800),
            transforms.CenterCrop(800),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    img_tensor = preprocess(input_image)
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor

   
   
class RetinaNetModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image_tensors):
        
        images, targets = self.model.transform(image_tensors, None)
        features = self.model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
            
        features = list(features.values())
        head_outputs = self.model.head(features)
        # import pdb; pdb.set_trace()
        return image_tensors, features[0], features[1], features[2], features[3], features[4], head_outputs['cls_logits'], head_outputs['bbox_regression']

from torchvision.models.detection.image_list import ImageList  
class RetinaNetModelPostProcessing(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image_tensors, feat0, feat1, feat2, feat3, feat4, cls_logits, bbox_regression):        
        # get the original image sizes
        original_image_sizes: List[Tuple[int, int]] = []
        for img in image_tensors:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        
        features = [feat0, feat1, feat2, feat3, feat4]
        head_outputs = {'cls_logits': cls_logits, 'bbox_regression': bbox_regression}
        image_sizes = [tuple(img.shape[-2:]) for img in image_tensors]
        images = ImageList(image_tensors, image_sizes)
        anchors = self.model.anchor_generator(images, features)
        
        detections: List[Dict[str, Tensor]] = []
        # recover level sizes
        num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
        HW = 0
        for v in num_anchors_per_level:
            HW += v
        HWA = head_outputs['cls_logits'].size(1)
        A = HWA // HW
        num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

        # split outputs per level
        split_head_outputs: Dict[str, List[Tensor]] = {}
        for k in head_outputs:
            split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
        split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

        # compute the detections
        detections = self.model.postprocess_detections(split_head_outputs, split_anchors, images.image_sizes)
        detections = self.model.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        
        return detections

def test_retinanet_r50_fpn_v1_torchvision_pytorch(test_device):
    pytest.skip("Under development")

    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Bfp8_b
    compiler_cfg.amp_level = 2
    compiler_cfg.enable_auto_fusing = False
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_8"] = 7
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_167"] = 3
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_219"] = 3
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_259"] = 3
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_299"] = 3
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_717"] = 3
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_723"] = 3
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_725"] = 3
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_727"] = 3
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_729"] = 3
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_789"] = 3
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_791"] = 3
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_793"] = 3
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_795"] = 3
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_797"] = 3
    os.environ["PYBUDA_GRAPHSOLVER_SELF_CUT_TYPE"] = "ConsumerOperandDataEdgesFirst"

    # STEP 2: Create PyBuda module from PyTorch model 
    model = download_model(torchvision.models.detection.retinanet_resnet50_fpn, pretrained=True)
    model.eval()
    tt_model = pybuda.PyTorchModule("retinanet_v1_pt", RetinaNetModelWrapper(model))
    # import pdb; pdb.set_trace()
    # STEP 3: Run inference on Tenstorrent device 
    img_tensor = get_image()
    output = model(img_tensor)

    tt_model = RetinaNetModelWrapper(model)
    cpu_model = RetinaNetModelPostProcessing(model)
    tt_output = cpu_model(*tt_model(img_tensor))
    tt0 = pybuda.TTDevice("tt0", devtype=test_device.devtype, arch=test_device.arch, module=PyTorchModule("retinanet_pt", tt_model))
    cpu1 = pybuda.CPUDevice("cpu1", module=PyTorchModule("retinanet_pt", cpu_model))
    
    tt0.push_to_inputs(img_tensor)
    output_q = pybuda.run_inference(_verify_cfg=VerifyConfig(relative_atol=0.3), _sequential=True)
    
