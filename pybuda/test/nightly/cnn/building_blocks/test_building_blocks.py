# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import subprocess
import os
import pytest
import sys


expected_to_pass_1 = [
    "test_mobilenet.py::test_mobilenet_v1_depthwise_separable_conv", 
    "test_mobilenet.py::test_mobilenet_v2_inverted_residual",
    "test_resnet.py::test_resnet_input_block",
    "test_resnet.py::test_resnet_output_block",
    "test_resnet.py::test_resnet_basic_block",
    "test_resnet.py::test_resnet_bottleneck_block",
    "test_resnet.py::test_resnext_bottleneck_block",
]

expected_to_pass_2 = [
    "test_unet.py::test_unet_double_conv_batchnorm_relu",
    "test_unet.py::test_unet_double_conv_relu",
    "test_unet.py::test_unet_double_conv_batchnorm_relu_maxpool",
    "test_unet.py::test_unet_maxpool",
]

expected_to_pass_3 = [
    "test_unet.py::test_unet_upconv",
    "test_unet.py::test_unet_upconv_double_conv_relu",
    "test_unet.py::test_unet_concat",
    "test_unet.py::test_unet_unityconv",
]

expected_to_pass_4 = [
    "test_vit.py::test_vit_encoder",
    "test_vit.py::test_vit_pooler"
]

expected_to_fail = [
    "test_mobilenet.py::test_mobilenet_v1_depthwise_separable_conv_xfail", 
    "test_mobilenet.py::test_mobilenet_v2_inverted_residual_xfail",
    "test_resnet.py::test_resnet_basic_block_xfail",
    "test_resnet.py::test_resnet_bottleneck_block_xfail",
    "test_resnet.py::test_resnext_bottleneck_block_xfail",
    "test_unet.py::test_unet_double_conv_batchnorm_relu_xfail",
    "test_unet.py::test_unet_double_conv_relu_xfail",
    "test_vit.py::test_vit_encoder_xfail"
]

def get_path_to_test(test_name):
    """Util function which returns path from pybuda repo root to the function pytest receives."""
    def normalize_string(string):
        return string.decode("utf-8").strip()
    
    git_root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True)

    if git_root.returncode == 0:
        git_root = str(normalize_string(git_root.stdout)) + "/"
        path_to_current_folder = os.path.dirname(os.path.realpath(__file__))
        return f"{path_to_current_folder[len(git_root):]}/{test_name}"
    
    raise RuntimeError("Must be run from inside git repo!")

def test_expected_to_pass(group_id: str):
    """Run tests that are expected to pass."""
    if group_id in ["1", "2", "3", "4"]:
        pytest.main(["-sv", *[get_path_to_test(test) for test in eval(f"expected_to_pass_{group_id}")]])
    else:
        raise ValueError("Unsupported expected_to_pass group! Must be one of [1, 2, 3, 4]")

def test_expected_to_fail():
    """Run tests that are expected to fail due to known issues."""
    pytest.main(["-sv", *[get_path_to_test(test) for test in expected_to_fail]])


if __name__ == "__main__":
    if sys.argv[1].startswith("--expected_to_pass"):
        test_expected_to_pass(sys.argv[1][-1])
    elif sys.argv[1] == "--expected_to_fail":
        test_expected_to_fail()
    else:
        print("Must pass `--expected_to_pass` or `--expected_to_fail`!")