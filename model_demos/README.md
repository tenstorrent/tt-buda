# Model Demos

Short demos for a broad range of NLP and CV models.

## Setup Instructions

### Install requirements

First, create either a Python virtual environment with PyBUDA installed or execute from a Docker container with PyBUDA installed.

Installation instructions can be found at [Install TT-BUDA](../first_5_steps/1_install_tt_buda.md).

Then on the machine running the install:

```bash
pip install -r requirements.txt
```

## Quick Start

With an activate Python environment and all dependencies installed, run:

```bash
export PYTHONPATH=.
python cv_demos/resnet/pytorch_resnet.py
```

## Models Table

| **Model** | **Supported Hardware** <br /> GS - Grayskull <br /> WH - Wormhole |
|-------------------------------------------|:--------:|
|   [ALBERT](nlp_demos/albert/)            |     GS, WH   |
|   [Autoencoder](cv_demos/autoencoder/)  |     GS, WH   |
|   [BERT](nlp_demos/bert/)                |     GS, WH   |
|   [CLIP](cv_demos/clip/)                |     GS, WH   |
|   [CodeGen](nlp_demos/codegen/)          |     GS, WH   |
|   [DeiT](cv_demos/deit/)                |     GS, WH   |
|   [DenseNet](cv_demos/densenet/)        |     GS, WH   |
|   [DistilBERT](nlp_demos/distilbert/)    |     GS, WH   |
|   [DPR](nlp_demos/dpr/)                  |     GS, WH   |
|   [Falcon](nlp_demos/falcon/)               |    WH   |
|   [FLAN-T5](nlp_demos/flant5/)           |     GS, WH   |
|   [GoogLeNet](cv_demos/googlenet/)      |     GS, WH   |
|   [GPT-2](nlp_demos/gpt2/)               |     GS, WH   |
|   [GPT Neo](nlp_demos/gptneo/)           |     GS, WH   |
|   [HRNet](cv_demos/hrnet/)              |     GS, WH   |
|   [Inception-v4](cv_demos/inceptionv4/) |    GS, WH   |
|   [MobileNetV1](cv_demos/mobilenetv1/)  |     GS, WH   |
|   [MobileNetV2](cv_demos/mobilenetv2/)  |     GS, WH   |
|   [MobileNetV3](cv_demos/mobilenetv3/)  |     GS, WH   |
|   [OPT](nlp_demos/opt/)                  |     GS, WH   |
|   [ResNet](cv_demos/resnet/)            |     GS, WH   |
|   [ResNeXt](cv_demos/resnext/)          |     GS, WH   |
|   [RoBERTa](nlp_demos/roberta/)          |     GS, WH   |
|   [SqueezeBERT](nlp_demos/squeezebert/)  |     GS, WH   |
|   [Stable Diffusion](cv_demos/stable_diffusion/)    |    WH   |
|   [T5](nlp_demos/t5/)                    |     GS, WH   |
|   [U-Net](cv_demos/unet/)               |    GS, WH   |
|   [VGG](cv_demos/vgg/)                  |     GS, WH   |
|   [ViT](cv_demos/vit/)                  |     GS, WH   |
|   [VoVNet](cv_demos/vovnet/)            |     GS, WH   |
|   [Whisper](audio_demos/whisper/)          |     GS, WH   |
|   [XGLM](nlp_demos/xglm/)                |     GS, WH   |
|   [YOLOv5](cv_demos/yolov5/)            |     GS, WH   |
