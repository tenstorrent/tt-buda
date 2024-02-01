# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
from codeop import Compile
import timm
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.mnasnet import MNASNet
from transformers import (
    T5Config,
    T5Model,
    GPTJConfig,
    GPT2Model,
    GPT2Config,
    GPTNeoModel,
    GPTNeoConfig,
    BertModel,
    AlbertModel,
    DetrModel,
    LongformerModel,
    Wav2Vec2Model,
    UniSpeechModel,
    ConvNextModel,
)
from transformers.models.gptj.modeling_gptj import GPTJBlock
from pybuda import (
    CompileDepth,
    CompilerConfig,
)

from test.legacy_tests.clip_guided_diffusion.unet.pytorch_unet import  create_UNet
from test.legacy_tests.clip_guided_diffusion.clip.clip_torch import create_CLIP
from test.tvm.clip_guided_diffusion.UNet.test_UNet_blocks import default_res_block_config, init_res_and_attention_blocks, default_attention_block_config
from test.legacy_tests.clip_guided_diffusion.unet.pytorch_unet import QKVAttentionLegacy
from test.legacy_tests.clip_guided_diffusion.unet.test_attention_block import init_attention_block
from test.tvm.cnn.pytorch.tests_B.test_mobilenet_v1 import MobileNetV1
from test.tvm.cnn.pytorch.tests_B.test_dense_depth import DenseDepth
from test.tvm.cnn.pytorch.tests_B.test_deepconn import DeepCoNN, Config as DeepCoNNConfig
from test.tvm.cnn.pytorch.tests_B.test_monodepth import MonoDepthNet
from test.tvm.cnn.pytorch.tests_B.test_openpose import OpenPoseBodyModel, OpenPoseHandModel


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        self.f = -1
        self.i = -1

    def forward(self, x):
        return x


def get_vgg_model(training, recompute):
    model = torch.hub.load(
        "pytorch/vision:v0.10.0", "vgg11", pretrained=True, force_reload=True
    )
    input_shape = (1, 3, 224, 224)
    act1 = torch.rand(*input_shape)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER,
    )
    
    return model, [
        act1,
    ], compile_cfg


def get_mobilenetv2_model(training, recompute):
    model = torch.hub.load(
        "pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=True, force_reload=True
    )
    input_shape = (1, 3, 224, 224)
    act1 = torch.rand(*input_shape)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER,
    )
    
    return model, [
        act1,
    ], compile_cfg


def get_unet_model(training, recompute):
    model = torch.hub.load(
        "mateuszbuda/brain-segmentation-pytorch",
        "unet",
        in_channels=3,
        out_channels=1,
        init_features=32,
        pretrained=True,
        force_reload=True,
    )
    input_shape = (1, 3, 256, 256)
    act1 = torch.rand(*input_shape)
    
    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER,
    )

    return model, [
        act1,
    ], compile_cfg


def get_densenet_block(training, recompute):
    model = torch.hub.load("pytorch/vision:v0.10.0", "densenet121", pretrained=True)
    input_shape = (1, 64, 256, 256)
    act1 = torch.rand(*input_shape)
    
    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER,
    )

    return model.features.denseblock1, [
        act1,
    ], compile_cfg


def get_yolov5_model(training, recompute):
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

    # Setup export for the last Detect module
    # model.model.model.model[-1].inplace = False
    # model.model.model.model[-1].export = True

    # The last Detect module in yolov5 has 5D shapes. Skip for now
    model.model.model.model[-1] = Identity()
    input_shape = (1, 3, 320, 320)
    act1 = torch.rand(*input_shape)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
    )
    
    return model, [
        act1,
    ], compile_cfg


def get_resnet_model(training, recompute):
    model = torch.hub.load(
        "pytorch/vision:v0.10.0", "resnet18", pretrained=True, force_reload=True
    )
    input_shape = (1, 3, 320, 320)
    act1 = torch.rand(*input_shape)
    
    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER,
    )

    return model, [
        act1,
    ], compile_cfg


def get_t5_small(training, recompute):
    config = T5Config.from_pretrained("t5-small", torchscript=True)
    model = T5Model(config)
    input_shape = (1, 128, 512)
    act1 = torch.rand(*input_shape)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
    )
    
    return model.decoder.block[0], [
        act1,
    ], compile_cfg


def get_MNIST_model(training, recompute):
    class MNIST(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=5 // 2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=5 // 2)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(1024, 320)
            self.fc2 = nn.Linear(320, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 3, padding=3 // 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 3, padding=3 // 2))
            x = x.view(-1, 1024)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, self.training)
            x = self.fc2(x)
            smx = F.softmax(x, dim=-1)
            return torch.log(smx)

    model = MNIST()
    input_shape = (1, 1, 32, 32)
    act1 = torch.rand(*input_shape)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
    )
    
    return model, [
        act1,
    ], compile_cfg


def get_gptj_block(training, recompute):
    config = GPTJConfig(n_layer=1)  # for faster loading
    config.activation_function = "gelu"
    config.rotary_dim = 64
    model = GPTJBlock(config)
    input_shape = (1, 128, 4096)
    act1 = torch.rand(*input_shape)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER,
    )
    
    return model, [
        act1,
    ], compile_cfg


def get_gpt2_block(training, recompute):
    config = GPT2Config()
    config.activation_function = "gelu"
    model = GPT2Model(config)
    input_shape = (1, 64, 768)
    act1 = torch.rand(*input_shape)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        enable_tvm_constant_prop=True,
        tvm_constnat_prop_mask={"attn.c_attn.weight", "attn.c_attn.bias"}
    )

    return model.h[0], [
        act1,
    ], compile_cfg, "c_attn.bias_1"


def get_gptneo_125M_block(training, recompute):
    config = GPTNeoConfig.from_pretrained("EleutherAI/gpt-neo-125M", torchscript=True)
    config.activation_function = "gelu"
    model = GPTNeoModel(config)
    input_shape = (1, 64, 768)
    act1 = torch.rand(*input_shape)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER,
    )
    
    return model.h[0], [
        act1,
    ], compile_cfg


def get_gptneo_13B_block(training, recompute):
    config = GPTNeoConfig.from_pretrained("EleutherAI/gpt-neo-1.3B", torchscript=True)
    config.activation_function = "gelu"
    model = GPTNeoModel(config)
    input_shape = (1, 64, 2048)
    act1 = torch.rand(*input_shape)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
    )
    
    return model.h[0], [
        act1,
    ], compile_cfg


def get_gptneo_27B_block(training, recompute):
    config = GPTNeoConfig.from_pretrained("EleutherAI/gpt-neo-2.7B", torchscript=True)
    config.activation_function = "gelu"
    model = GPTNeoModel(config)
    input_shape = (1, 64, 2560)
    act1 = torch.rand(*input_shape)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
    )
    
    return model.h[0], [
        act1,
    ], compile_cfg


def get_efficientnet_layer(training, recompute):
    model = torch.hub.load(
        "NVIDIA/DeepLearningExamples:torchhub",
        "nvidia_efficientnet_b0",
        pretrained=True,
        force_reload=True,
    )
    input_shape = (1, 32, 112, 112)
    act1 = torch.rand(*input_shape)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER,
    )
    
    return model.layers[0], [
        act1,
    ], compile_cfg


def get_bert_encoder(training, recompute):
    model = BertModel.from_pretrained("prajjwal1/bert-tiny", torchscript=True)
    input_shape = (1, 128, 128)
    act1 = torch.rand(*input_shape)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER,
    )
    
    return model.encoder, [
        act1,
    ], compile_cfg


def get_albert_attention(training, recompute):
    model = AlbertModel.from_pretrained("albert-base-v1", torchscript=True)
    input_shape = (1, 768, 768)
    act1 = torch.rand(*input_shape)
    
    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER,
    )
    
    return model.encoder.albert_layer_groups[0].albert_layers[0].attention, [
        act1,
    ], compile_cfg


def get_alexnet_model(training, recompute):
    model = torch.hub.load(
        "pytorch/vision:v0.10.0",
        "alexnet",
        pretrained=True,
        force_reload=True,
    )
    input_shape = (1, 3, 256, 256)
    act1 = torch.rand(*input_shape)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER,
    )
    
    return model, [
        act1,
    ], compile_cfg

def get_fcn_torch(training, recompute):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True, force_reload=True)
    input_shape = (1, 3, 224, 224)
    act1 = torch.rand(*input_shape)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER,
    )
    
    return model, [
        act1,
    ], compile_cfg


def get_detr(training, recompute):
    model = DetrModel.from_pretrained("facebook/detr-resnet-50", torchscript=True)
    input_shape = (1, 3, 256, 256)
    act1 = torch.rand(*input_shape)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
    )
    
    return model, [
        act1,
    ], compile_cfg


def get_deeplabv3_torch(training, recompute):
    model = torch.hub.load(
        "pytorch/vision:v0.10.0", "deeplabv3_resnet50", pretrained=True
    )
    input_shape = (1, 3, 224, 224)
    act1 = torch.rand(*input_shape)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        compile_depth=CompileDepth.PRE_LOWERING_PASS,
    )
    
    return model, [
        act1,
    ], compile_cfg


def get_clip_unet_torch(training, recompute):
    UNet_embeddings, UNet_no_emb, UNetTorchModel, UNet_config = create_UNet()
    act1_shape = (1, 3, 256, 256)
    act1 = torch.rand(*act1_shape)
    timesteps_shape = (1, )
    timesteps =  torch.randint(0, 1, size=timesteps_shape, requires_grad=False).float()
    
    act1, embedded_res = UNet_embeddings(timesteps, act1)
    
    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        compile_depth=CompileDepth.POST_INITIAL_GRAPH_PASS,
    )
    
    return UNet_no_emb, [act1, embedded_res], compile_cfg

def get_unet_resblock_upsample_resblock(training, recompute):
    ch = 768
    out_ch = 512
    
    first_res_block_config = default_res_block_config()
    first_res_block_config.update(dict(channels=ch, out_channels=out_ch))
    second_res_block_config = default_res_block_config()
    second_res_block_config.update(dict(channels=out_ch, out_channels=out_ch, up=True))
    model = init_res_and_attention_blocks(
        first_res_block_config,
        add_second_res_block=True,
        second_res_block_config=second_res_block_config,
    )
    
    act1 = torch.randn(1, 768, 64, 64)
    torch_emb = torch.randn(1, 1024)
    
    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER,
    )
    
    return model, [act1, torch_emb], compile_cfg

def get_unet_resblock_attention_block(training, recompute):
    
    ch = 512
    out_ch = 512
    
    first_res_block_config = default_res_block_config()
    first_res_block_config.update(dict(channels=ch, out_channels=out_ch))
    attention_block_config = default_attention_block_config()
    attention_block_config.update(dict(channels=out_ch))
    model = init_res_and_attention_blocks(
        first_res_block_config,
        add_attention_block=True,
        attention_config=attention_block_config,
    )
    
    act1 = torch.randn(1, 512, 32, 32)
    torch_emb = torch.randn(1, 1024)
    
    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER,
    )
    
    return model, [act1, torch_emb], compile_cfg

def get_unet_attention_block(training, recompute):
    channels = 512
    if channels == 512:
        act1 = torch.randn(1, 512, 32, 32)
    else:
        assert channels == 1024
        act1 = torch.randn(1, 1024, 8, 8)
    
    model = init_attention_block(channels)
    
    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER,
    )
    
    return model, [act1], compile_cfg

def get_unet_qkv_attention(training, recompute):
    num_heads = 4
    model = QKVAttentionLegacy(num_heads)
    
    acts = torch.randn(1, 1536, 1024)
    
    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
    )
    
    return model, [acts], compile_cfg

def get_clip(training, recompute):
    
    pytorch_clip_text_encoder, pytorch_clip_without_text_encoder, pytorch_clip, clip_config = create_CLIP()
    encoded_text = pytorch_clip_text_encoder()
    encoded_text = encoded_text.detach()
    image_shape = (1, 3, 224, 224)
    image = torch.rand(image_shape)
    
    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        compile_depth=CompileDepth.GENERATE_INITIAL_GRAPH,
    )
    
    return pytorch_clip_without_text_encoder, [image, encoded_text], compile_cfg


def get_MIDAS_torch(training, recompute):
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    act1 = torch.rand((1, 3, 384, 384))

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
    )
    
    return model, [
        act1,
    ], compile_cfg


def get_longformer_torch(training, recompute):
    model = LongformerModel.from_pretrained("allenai/longformer-base-4096", torchscript=True)

    input_shape = (1, 8 * 512)
    act = torch.randint(0, 9000, input_shape, dtype=torch.long)
    attention_mask = torch.ones(1, 8 * 512, dtype=torch.long)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
    )
    
    return model, [
        act,
        attention_mask,
    ], compile_cfg


def get_wav2vec2_torch(training, recompute):
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", torchscript=True)

    input_shape = (1, 512)
    act = torch.rand(input_shape)

    return model, [
        act,
    ]


def get_resnext_torch(training, recompute):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
    act = torch.rand((1, 3, 224, 224))
    
    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER,
    )
    
    return model, [act,], compile_cfg


def get_unispeech_torch(training, recompute):
    model = UniSpeechModel.from_pretrained("microsoft/unispeech-sat-base", torchscript=True)

    input_shape = (1, 512)
    act = torch.rand(input_shape)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
    )
    
    return model, [
        act,
    ], compile_cfg


def get_ssd_torch(training, recompute):
    from test.tvm.cnn.pytorch.tests_B.SSD.ssd import SSD
    model = SSD()
    
    input_shape = (1, 3, 300, 300)
    act = torch.rand(input_shape)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        compile_depth=CompileDepth.GENERATE_INITIAL_GRAPH,
    )
    
    return model, [
        act,
    ], compile_cfg


def get_mnasnet_torch(training, recompute):
    model = MNASNet(1.0)
    
    input_shape = (1, 3, 64, 64)
    act = torch.rand(input_shape)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER,
    )
    
    return model, [
        act,
    ], compile_cfg, ['layers.4.bias']


def get_mobilenet_v1_torch(training, recompute):
    model = MobileNetV1(9)
    
    input_shape = (1, 3, 64, 64)
    act = torch.rand(input_shape)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER,
    )
    
    return model, [
        act,
    ], compile_cfg, ['model.1.bias']


def get_convnext_torch(training, recompute):
    model = ConvNextModel.from_pretrained(
        "facebook/convnext-tiny-224", torchscript=True
    )
    
    input_shape = (1, 3, 64, 64)
    act = torch.rand(input_shape)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER,
    )
    
    return model, [
        act,
    ], compile_cfg


def get_hrnet_torch(training, recompute):
    model = timm.create_model('hrnet_w18')
    
    input_shape = (1, 3, 64, 64)
    act = torch.rand(input_shape)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER,
    )
    
    return model, [
        act,
    ], compile_cfg


def get_dense_depth_torch(training, recompute):
    model = DenseDepth()
    
    input_shape = (1, 3, 64, 64)
    act = torch.rand(input_shape)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER,
    )
    
    return model, [
        act,
    ], compile_cfg


def get_deep_conn_torch(training, recompute):
    config = DeepCoNNConfig()
    model = DeepCoNN(config)
    
    input_shape = (1, 10, 100)
    act1 = torch.randint(0, 9001, input_shape, dtype=torch.long)
    act2 = torch.randint(0, 9001, input_shape, dtype=torch.long)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
    )
    
    return model, [
        act1,
        act2,
    ], compile_cfg


def get_mono_depth_torch(training, recompute):
    model = MonoDepthNet(3, 2)
    
    input_shape = (1, 3, 256, 512)
    act = torch.rand(input_shape)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER,
    )
    
    return model, [
        act,
    ], compile_cfg


def get_open_pose_body_torch(training, recompute):
    model = OpenPoseBodyModel()
    
    input_shape = (1, 3, 64, 64)
    act = torch.rand(input_shape)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER,
    )
    
    return model, [
        act,
    ], compile_cfg


def get_open_pose_hand_torch(training, recompute):
    model = OpenPoseHandModel()
    
    input_shape = (1, 3, 64, 64)
    act = torch.rand(input_shape)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER,
    )
    
    return model, [
        act,
    ], compile_cfg


pytorch_model_name_to_pybuda_model = {
    "albert_attention_torch": get_albert_attention,
    "alexnet_torch": get_alexnet_model,
    "bert_encoder_torch": get_bert_encoder,
    "CLIP_guided_diffusion_attention": get_unet_attention_block,
    "CLIP_guided_diffusion_clip": get_clip,
    "CLIP_guided_diffusion_qkv_attention": get_unet_qkv_attention,
    "CLIP_guided_diffusion_resblock_attention": get_unet_resblock_attention_block,
    "CLIP_guided_diffusion_unet": get_clip_unet_torch,
    "CLIP_guided_diffusion_upsample_resblock": get_unet_resblock_upsample_resblock,
    "convnext_torch": get_convnext_torch,
    # "deep_conn_torch": get_deep_conn_torch, # Embeddings not supported
    "deeplabv3_torch": get_deeplabv3_torch,
    "dense_depth_torch": get_dense_depth_torch,
    "densenet_block_torch": get_densenet_block,
    "detr_torch": get_detr,
    "efficientnet_layer_torch": get_efficientnet_layer,
    "fcn_torch": get_fcn_torch,
    "get_unispeech_torch": get_unispeech_torch,
    "gpt2_block_torch": get_gpt2_block,
    "gptj_block_torch": get_gptj_block,
    "gptneo_125M_block_torch": get_gptneo_125M_block,
    "gptneo_13B_block_torch": get_gptneo_13B_block,
    "gptneo_27B_block_torch": get_gptneo_27B_block,
    "hrnet_torch": get_hrnet_torch,
    # "MIDAS_torch": get_MIDAS_torch, # FileNotFoundError: [Errno 2] No such file or directory: '.../.cache/torch/hub/intel-isl_MiDaS_master/hubconf.py'
    "mnasnet_torch": get_mnasnet_torch,
    "mnist_torch": get_MNIST_model,
    "mobilenet_v1_torch": get_mobilenet_v1_torch,
    "mobilenetv2_torch": get_mobilenetv2_model,
    "mono_depth_torch": get_mono_depth_torch,
    "open_pose_body_torch": get_open_pose_body_torch,
    "open_pose_hand_torch": get_open_pose_hand_torch,
    "resnet_torch": get_resnet_model,
    "resnext_torch": get_resnext_torch,
    "ssd_torch": get_ssd_torch,
    "t5_small_block_torch": get_t5_small,
    "unet_model_torch": get_unet_model,
    "vgg11_torch": get_vgg_model,
    "wav2vec2_torch": get_wav2vec2_torch,
    "yolov5_model_torch": get_yolov5_model,
}


passing_pytorch_model_name_to_pybuda_model_inference = [
    "albert_attention_torch",
    "alexnet_torch",
    "bert_encoder_torch",
    "CLIP_guided_diffusion_attention",
    "CLIP_guided_diffusion_clip",
    "CLIP_guided_diffusion_qkv_attention",
    "CLIP_guided_diffusion_unet",
    "CLIP_guided_diffusion_upsample_resblock",
    "convnext_torch",
    "deeplabv3_torch", # Takes too much memory + crashes
    "dense_depth_torch",
    "densenet_block_torch",
    "efficientnet_layer_torch",
    "fcn_torch",
    "gpt2_block_torch",
    "gptj_block_torch", # tenstorrent/pybuda#63
    "gptneo_125M_block_torch",
    "gptneo_13B_block_torch",
    "gptneo_27B_block_torch",
    "hrnet_torch",
    "mnasnet_torch",
    "mnist_torch",
    "mobilenet_v1_torch",
    "mobilenetv2_torch",
    "mono_depth_torch",
    "open_pose_body_torch",
    "open_pose_hand_torch",
    "resnet_torch",
    "resnext_torch",
    "ssd_torch",
    "unet_model_torch",
    "vgg11_torch",
    # "yolov5_model_torch", # The test is not setup properly: Identity has no attribute 'stride'
    # "resnext_torch",
    # "t5_small_block_torch", # Const eval issue,
]


passing_pytorch_model_name_to_pybuda_model_training = [
    "albert_attention_torch",
    "bert_encoder_torch",
    "CLIP_guided_diffusion_qkv_attention",
    "convnext_torch",
    "efficientnet_layer_torch",
    "gpt2_block_torch",
    "gptneo_125M_block_torch",
    "mnasnet_torch",
    "mobilenet_v1_torch",
    # "gptneo_13B_block_torch", # Error: TT_ASSERT @ pybuda/csrc/passes/balancer_error_handlers.cpp:46: op_node->is_matmul()
    # "gptneo_27B_block_torch", # Error: TT_ASSERT @ pybuda/csrc/passes/balancer_error_handlers.cpp:46: op_node->is_matmul()
]
