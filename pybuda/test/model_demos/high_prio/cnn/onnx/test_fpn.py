import torch
import pybuda
import onnx
import os
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig


def test_fpn_onnx(test_device, test_kind):
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.Float16_b

    os.environ["PYBUDA_RIBBON2"] = "1"
    os.environ["PYBUDA_FORCE_EMULATE_HARVESTED"] = "1"

    # Load FPN model
    onnx_model_path = os.path.dirname(os.path.realpath(__file__)) + "/weights/fpn.onnx"
    model = onnx.load(onnx_model_path)
    tt_model = pybuda.OnnxModule("onnx_fpn", model, onnx_model_path)

    feat0 = torch.rand(1, 10, 64, 64)
    feat1 = torch.rand(1, 20, 16, 16)
    feat2 = torch.rand(1, 30, 8, 8)

    verify_module(
        tt_model,
        input_shapes=[feat0.shape, feat1.shape, feat2.shape],
        inputs=[(feat0, feat1, feat2)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=test_kind,
        ),
    )
