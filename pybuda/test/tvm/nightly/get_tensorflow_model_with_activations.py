# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
import tensorflow as tf
from transformers import (
    T5Config,
    TFT5Model,
    GPTJConfig,
    GPT2Config,
    BertConfig,
    AlbertConfig,
    TFLongformerModel,
    TFWav2Vec2Model,
    TFRobertaModel
)
from transformers.models.gptj.modeling_tf_gptj import TFGPTJModel
from transformers.models.gpt2.modeling_tf_gpt2 import TFBlock as TFGPT2Block
from transformers.models.bert.modeling_tf_bert import TFBertLayer
from transformers.models.albert.modeling_tf_albert import TFAlbertAttention

from test.tvm.nlp.tensorflow.detr.config import TrainingConfig
from test.tvm.nlp.tensorflow.detr.detr import get_detr_model
from pybuda.config import CompileDepth, CompilerConfig

class Identity(tf.keras.Model):
    def __init__(self):
        super(Identity, self).__init__()
        self.f = -1
        self.i = -1

    def forward(self, x):
        return x


def get_vgg_model(training, recompute):
    input_shape = (1, 224, 224, 3)
    model = tf.keras.applications.VGG16(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
    )
    
    act1 = tf.random.uniform(input_shape)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER,
    )

    return model, [
        act1,
    ], compile_cfg


def get_mobilenetv2_model(training, recompute):
    input_shape = (1, 224, 224, 3)
    model = tf.keras.applications.MobileNetV2 (
        input_shape=input_shape[1:]
    )
    
    act1 = tf.random.uniform(input_shape)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER,
    )
    
    return model, [
        act1,
    ], compile_cfg


def get_resnet_model(training, recompute):
    model = tf.keras.applications.resnet50.ResNet50(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
    )
    input_shape = (1, 3, 320, 320)
    act1 = tf.random.uniform(input_shape)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
    )
    
    return model, [
        act1,
    ], compile_cfg


def get_t5_small(training, recompute):
    config = T5Config.from_pretrained("t5-small", torchscript=True)
    t5_model = TFT5Model(config)
    input_shape = (1, 128, 512)
    act1 = tf.random.uniform(input_shape)

    class T5block(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.layer = t5_model.get_decoder().block[0]
        def call(self, *x):
            return self.layer(*x, False, False, False)[0]

    model = T5block()
    inputs = [act1] + [None]*8
    
    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
    )
    
    return model, inputs, compile_cfg


def get_MNIST_model(training, recompute):
    class MNIST(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=5, padding='same')
            self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=5, padding='same')
            # self.conv2_drop = tf.keras.layers.Dropout(0.5)
            self.fc1 = tf.keras.layers.Dense(320)
            self.fc2 = tf.keras.layers.Dense(10)

        def call(self, x):
            x = tf.nn.relu(tf.nn.max_pool2d(self.conv1(x), 3, strides=3, padding=[[0, 0], [1, 1], [1, 1], [0, 0]]))
            x = tf.nn.relu(tf.nn.max_pool2d(self.conv2(x), 3, strides=3, padding=[[0, 0], [1, 1], [1, 1], [0, 0]]))
            x = tf.reshape(x, [1, 1024])
            x = tf.nn.relu(self.fc1(x))
            # x = tf.nn.dropout(x, 0.5)
            x = self.fc2(x)
            smx = tf.nn.softmax(x, axis=-1)
            return tf.math.log(smx)

    model = MNIST()
    input_shape = (1, 32, 32, 1)
    act1 = tf.random.uniform(input_shape)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
    )
    
    return model, [
        act1,
    ], compile_cfg


def get_gptj_full(training, recompute):
    configuration = GPTJConfig(n_layer=1)
    configuration.rotary_dim = 64
    configuration.use_cache = False
    model = TFGPTJModel(configuration)

    input_shape = (1, 128)
    act1 = tf.random.uniform(input_shape, minval=1, maxval=50257, dtype=tf.int32)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
    )
    return model, [
        act1,
    ], compile_cfg


def get_gpt2_block(training, recompute):
    config = GPT2Config()
    config.activation_function = "gelu"
    class TF_GPT2Block(tf.keras.Model):
        def __init__(self, config):
            super().__init__()
            self.layer = TFGPT2Block(config)

        def call(self, *x):
            return self.layer(*x)

    model = TF_GPT2Block(config)
    input_shape = (1, 64, 768)
    act1 = tf.random.uniform(input_shape)
    inputs = [act1] + [None]*7
    
    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
    )
    
    return model, inputs, compile_cfg

def get_efficientnet_layer(training, recompute):
    input_shape = (1, 112, 112, 32)
    blocks_args = [{
        'kernel_size': 3,
        'repeats': 1,
        'filters_in': 32,
        'filters_out': 16,
        'expand_ratio': 1,
        'id_skip': True,
        'strides': 1,
        'se_ratio': 0.25
    }]
    class EfficientNetB0Layer(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.layer = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=input_shape[1:], blocks_args=blocks_args, weights=None)

        def call(self, x):
            return self.layer(x)

    model = EfficientNetB0Layer()
    act1 = tf.random.uniform(input_shape)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        compile_depth=CompileDepth.GENERATE_INITIAL_GRAPH,
    )
    
    return model.layers[0], [
        act1,
    ], compile_cfg


def get_bert_layer(training, recompute):
    input_shape = (1, 128, 128)
    model_config = {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 128,
        "initializer_range": 0.02,
        "intermediate_size": 512,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "bert",
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30522,
    }
    class TF_BertLayer(tf.keras.Model):
        def __init__(self, config):
            super().__init__()
            self.layer = TFBertLayer(config)

        def call(self, *x):
            return self.layer(*x)

    config = BertConfig(**model_config)
    model = TF_BertLayer(config=config)
    act1 = tf.random.uniform(input_shape)
    inputs = [act1] + [None]*6
    
    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
    )
    
    return model, inputs, compile_cfg


def get_albert_attention(training, recompute):
    
    input_shape = (1, 768, 768)
    model_config_v1 = {
        "_name_or_path": "albert-base-v1",
        "architectures": [
            "AlbertForMaskedLM"
        ],
        "attention_probs_dropout_prob": 0.1,
        "bos_token_id": 2,
        "classifier_dropout_prob": 0.1,
        "down_scale_factor": 1,
        "embedding_size": 128,
        "eos_token_id": 3,
        "gap_size": 0,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "inner_group_num": 1,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "albert",
        "net_structure_type": 0,
        "num_attention_heads": 12,
        "num_hidden_groups": 1,
        "num_hidden_layers": 12,
        "num_memory_blocks": 0,
        "pad_token_id": 0,
        "position_embedding_type": "absolute",
        "torchscript": True,
        "transformers_version": "4.12.2",
        "type_vocab_size": 2,
        "vocab_size": 30000
    }
    class TF_AlbertAttention(tf.keras.Model):
        def __init__(self, config):
            super().__init__()
            self.layer = TFAlbertAttention(config)

        def call(self, *x):
            return self.layer(*x)

    config = AlbertConfig(**model_config_v1)
    
    model = TF_AlbertAttention(config)
    act1 = tf.random.uniform(input_shape)
    inputs = [act1] + [None]*3
    
    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
    )
    
    return model, inputs, compile_cfg


def get_longformer_tf(training, recompute):
    model = TFLongformerModel.from_pretrained("allenai/longformer-base-4096")

    input_shape = (1, 8 * 512)
    act = tf.random.uniform(input_shape, 0, 9000, dtype=tf.int64)
    attention_mask = tf.ones((1, 8 * 512), dtype=tf.int64)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
    )
    
    return model, [
        act,
        attention_mask,
    ], compile_cfg


def get_wav2vec2_tf(training, recompute):
    model = TFWav2Vec2Model.from_pretrained("facebook/wav2vec2-base", from_pt=True)

    input_shape = (1, 512)
    act = tf.random.uniform(input_shape)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
    )
    
    return model, [
        act,
    ], compile_cfg


def get_inceptionv3_tf(training, recompute):
    input_shape = (1, 75, 75, 3)
    model = tf.keras.applications.InceptionV3(include_top=False, input_shape=input_shape[1:])

    act = tf.random.uniform(input_shape)
    return model, [act]


def get_resnet_tf(training, recompute):
    model = tf.keras.applications.resnet50.ResNet50(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
    )

    act = tf.random.uniform((1, 224, 224, 3))
    
    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER,
    )
    
    return model, [act], compile_cfg


def get_roberta_encoder_tf(training, recompute):
    input_shape = (1, 256, 256)
    roberta_model = TFRobertaModel.from_pretrained("arampacha/roberta-tiny", from_pt=True)
    class TF_RobertaEncoder(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.layer = roberta_model.layers[0].encoder

        def call(self, hidden_states):
            return self.layer(hidden_states, None, [None]*4, None, None, None, None, False, False, False)

    model = TF_RobertaEncoder()
    act = tf.random.uniform(input_shape)

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER,
    )
    
    return model, [act], compile_cfg


def get_detr(training, recompute):
    config = TrainingConfig()
    model = get_detr_model(config=config)
    act = tf.random.uniform((1, 256, 256, 3))

    compile_cfg = CompilerConfig(
        enable_training=training,
        enable_recompute=recompute,
        compile_depth=CompileDepth.GENERATE_INITIAL_GRAPH,
    )
    
    return model, [act], compile_cfg


tensorflow_model_name_to_pybuda_model = {
    "albert_attention_tf": get_albert_attention,
    "bert_encoder_tf": get_bert_layer,
    "detr": get_detr,
    "efficientnet_layer_tf": get_efficientnet_layer,
    "gpt2_block_tf": get_gpt2_block,
    "gptj_full_tf": get_gptj_full,
    "inceptionv3_tf": get_inceptionv3_tf,
    "mnist_tf": get_MNIST_model,
    "mobilenetv2_tf": get_mobilenetv2_model,
    "resnet_tf": get_resnet_model,
    "resnet_tf": get_resnet_tf,
    "roberta_encoder_tf": get_roberta_encoder_tf,
    "t5_small_block_tf": get_t5_small,
    "vgg16_tf": get_vgg_model,
    "wav2vec2_tf": get_wav2vec2_tf,
}

passing_tensorflow_model_name_to_pybuda_model_inference = [
    "detr",
    "efficientnet_layer_tf",
    "gptj_full_tf",
    "mnist_tf",
    "mobilenetv2_tf",
    "resnet_tf",
    "roberta_encoder_tf",
    "vgg16_tf",
]

passing_tensorflow_model_name_to_pybuda_model_training = [
    "efficientnet_layer_tf",
    "roberta_encoder_tf",
]
