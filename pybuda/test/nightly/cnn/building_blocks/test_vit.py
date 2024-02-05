# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""Script with pytests for ViT.

Read: https://arxiv.org/pdf/2010.11929.pdf

Input images to ViT are shaped HxWxC. Since standard transformer receives as input
1D sequence of token embeddings, to handle images, we reshape them into a sequence
of flattened 2D patches of shape Nx(P*P*C), where N=(H*W)/(P*P). Class embedding 
is prepended to this sequence, thus resulting in input shape (N+1)x(P*P*C). This
is refered to as "patch embeddings" in the paper, and we currently do it on host
CPU. Thus, we won't be using input shape of input images, but rather the shape
of patch embeddings as input to our encoder.

MLP block is a fully connected layer which has one hidden layer, three in total. 
Input->hidden layer makes hidden_size->intermediate_size connections to hidden 
layer with hidden_act fnct, after which comes a Dropout with hidden_dropout_prob,
after which hidden->output layer makes intermediate_size->hidden_size connections.
We choose `intermed_expansion_factor` such that intermediate_size % hidden_size == 0.

See https://arxiv.org/pdf/2010.11929.pdf section 3.1 for more info.
"""

import pytest
from transformers import ViTModel, ViTConfig

from pybuda import (
    PyTorchModule,
    VerifyConfig,
)
from pybuda.config import _get_global_compiler_config
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind
from pybuda._C.backend_api import BackendType, BackendDevice


# TODO probably needs to be broken down into smaller building blocks since encoder itself is a 
# complex block.
@pytest.mark.parametrize("image_size", [224, 256])
@pytest.mark.parametrize("num_channels", [1, 3])
@pytest.mark.parametrize("patch_size", [16, 32])
@pytest.mark.parametrize("num_hidden_layers", [6, 8])
@pytest.mark.parametrize("num_attention_heads", [8, 16])
@pytest.mark.parametrize("intermed_expansion_factor", [3, 4])
@pytest.mark.parametrize("arch", [BackendDevice.Grayskull, BackendDevice.Wormhole])
def test_vit_encoder(
    image_size,
    num_channels,
    patch_size,
    num_hidden_layers,
    num_attention_heads,
    intermed_expansion_factor,
    arch
):  
    expected_to_fail = [
        (256, 3, 32, 6, 8, 3, BackendDevice.Wormhole),
        (224, 3, 32, 8, 8, 3, BackendDevice.Wormhole),
        (256, 3, 32, 6, 16, 3, BackendDevice.Wormhole),
        (224, 3, 32, 8, 16, 3, BackendDevice.Wormhole),
        (256, 3, 32, 6, 8, 4, BackendDevice.Wormhole),
        (224, 3, 32, 8, 8, 4, BackendDevice.Wormhole),
        (256, 3, 32, 8, 8, 4, BackendDevice.Wormhole),
        (224, 3, 32, 8, 16, 4, BackendDevice.Wormhole),
        (256, 3, 32, 8, 16, 4, BackendDevice.Wormhole),
        (256, 3, 32, 6, 16, 4, BackendDevice.Wormhole)
    ]

    if (image_size, num_channels, patch_size, num_hidden_layers, num_attention_heads, 
        intermed_expansion_factor, arch) in expected_to_fail:
        pytest.skip(msg="This combination is expected to fail, moved to _xfail version of the function.")

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    hidden_size = patch_size * patch_size * num_channels
    num_patches = (image_size**2)//(patch_size**2)
    input_shape = (1, num_patches+1, hidden_size)
    intermediate_size = intermed_expansion_factor * hidden_size

    config = ViTConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        image_size=image_size,
        patch_size=patch_size,
    )
    model = ViTModel(config)
    module = PyTorchModule("ViTEncoder", model.encoder)

    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=arch,
            devtype=BackendType.Golden,
            test_kind=TestKind.INFERENCE,
            waive_gradient_errors={"layer.0.attention.attention.key.bias"},
            pcc=0.89
        ),
    )

@pytest.mark.xfail(reason="XFAIL due to: "
                   "tenstorrent/pybuda#424")
@pytest.mark.parametrize(
    "image_size, num_channels, patch_size, num_hidden_layers, num_attention_heads, intermed_expansion_factor, arch",
    [
        (256, 3, 32, 6, 8, 3, BackendDevice.Wormhole),
        (224, 3, 32, 8, 8, 3, BackendDevice.Wormhole),
        (256, 3, 32, 6, 16, 3, BackendDevice.Wormhole),
        (224, 3, 32, 8, 16, 3, BackendDevice.Wormhole),
        (256, 3, 32, 6, 8, 4, BackendDevice.Wormhole),
        (224, 3, 32, 8, 8, 4, BackendDevice.Wormhole),
        (256, 3, 32, 8, 8, 4, BackendDevice.Wormhole),
        (224, 3, 32, 8, 16, 4, BackendDevice.Wormhole),
        (256, 3, 32, 8, 16, 4, BackendDevice.Wormhole),
        (256, 3, 32, 6, 16, 4, BackendDevice.Wormhole)
    ]
)
def test_vit_encoder_xfail(
    image_size,
    num_channels,
    patch_size,
    num_hidden_layers,
    num_attention_heads,
    intermed_expansion_factor,
    arch
):  
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    hidden_size = patch_size * patch_size * num_channels
    num_patches = (image_size**2)//(patch_size**2)
    input_shape = (1, num_patches+1, hidden_size)
    intermediate_size = intermed_expansion_factor * hidden_size

    config = ViTConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        image_size=image_size,
        patch_size=patch_size,
    )
    model = ViTModel(config)
    module = PyTorchModule("ViTEncoder", model.encoder)

    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=arch,
            devtype=BackendType.Golden,
            test_kind=TestKind.INFERENCE,
            waive_gradient_errors={"layer.0.attention.attention.key.bias"},
            pcc=0.89
        ),
    )

# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("image_size", [224, 256, 288, 320])
@pytest.mark.parametrize("num_channels", [3, 1])
@pytest.mark.parametrize("patch_size", [16, 32])
@pytest.mark.parametrize("num_attention_heads", [4, 8, 16])
@pytest.mark.parametrize("arch", [BackendDevice.Grayskull, BackendDevice.Wormhole])
def test_vit_pooler(
    image_size,
    num_channels,
    patch_size,
    num_attention_heads, 
    arch
):   
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    hidden_size = patch_size * patch_size * num_channels
    num_patches = (image_size**2)//(patch_size**2)
    input_shape = (1, num_patches+1, hidden_size)

    config = ViTConfig(hidden_size=hidden_size, num_attention_heads=num_attention_heads)
    model = ViTModel(config)
    module = PyTorchModule("ViTPooler", model.pooler)

    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=arch,
            devtype=BackendType.Golden,
            test_kind=TestKind.INFERENCE,
            pcc=0.89
        ),
    )
