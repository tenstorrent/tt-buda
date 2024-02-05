# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
import pytest

import torch
import importlib
import urllib

from pybuda import (
    PyTorchModule,
    VerifyConfig,
)
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind

def test_ghostnet(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()  # Backward is currently unsupported

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    #Fusing disabled due to tenstorrent/pybuda#800
    compiler_cfg.enable_auto_fusing=False

    # tenstorrent/pybuda#392
    import os
    os.environ["PYBUDA_DISABLE_CONSTANT_FOLDING"] = "1"
    
    # model = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=True)
    # Top file from torch hub depends on cuda import, so just get the model directly. 
    localfile, _ = urllib.request.urlretrieve("https://github.com/huawei-noah/ghostnet/raw/master/ghostnet_pytorch/ghostnet.py")
    ghostnet_module = importlib.machinery.SourceFileLoader("ghostnet", localfile).load_module()
    state_dict_url = 'https://github.com/huawei-noah/ghostnet/raw/master/ghostnet_pytorch/models/state_dict_73.98.pth'
    model = ghostnet_module.ghostnet(num_classes=1000, width=1.0, dropout=0.2)
    state_dict = torch.hub.load_state_dict_from_url(state_dict_url, progress=True)
    model.load_state_dict(state_dict)
    module = PyTorchModule("ghostnet", model.float())

    input_shape = (1, 3, 256, 256)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            pcc=0.9,
        ),
    )

def test_ghostnet_v2(test_kind, test_device):
    
    pytest.skip("Needs padding")
    
     # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = _get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "CNN"
 
    # Model load
    localfile, _ = urllib.request.urlretrieve("https://github.com/huawei-noah/ghostnet/raw/master/ghostnetv2_pytorch/model/ghostnetv2_torch.py")
    ghostnetv2_module = importlib.machinery.SourceFileLoader("ghostnetv2", localfile).load_module()
    model = ghostnetv2_module.ghostnetv2(num_classes=1000, width=1.6, dropout=0.2, args=None) # width = 1 | 1.3 | 1.6
    model.eval()

    module = PyTorchModule("pt_ghostnet_v2", model.float())

    input_shape = (1, 3, 256, 256)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
