import pybuda
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda.verify.config import TestKind
from pybuda._C.backend_api import BackendDevice
from transformers import Phi3Config, Phi3ForCausalLM, AutoTokenizer, Phi3ForTokenClassification, Phi3ForSequenceClassification
import os
import pytest

# Masked fill kernal produced invalid results in Silicon BackendType
# Issue link - https://yyz-gitlab.local.tenstorrent.com/tenstorrent/pybuda/-/issues/2712
# RMS block of phi3 produced different results on each run in GS when BBE is enabled
# https://yyz-gitlab.local.tenstorrent.com/tenstorrent/pybuda/-/issues/2838
# So Disabling the verification in BBE for Silicon BackendType

variants = ["microsoft/phi-3-mini-4k-instruct"]


@pytest.mark.parametrize("variant", variants)
def test_phi3_causal_lm(test_device, variant):

    # Configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    compiler_cfg.balancer_policy = "Ribbon"

    if test_device.arch == BackendDevice.Wormhole_B0:
        os.environ["PYBUDA_RIBBON2_CONSERVATIVE_OPTIMIZATION_ITERATIONS"] = "0"
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "15360"

    elif test_device.arch == BackendDevice.Grayskull:
        os.environ["PYBUDA_DRAM_PICK_CAPACITY"] = "1"
        os.environ["PYBUDA_DRAM_FLIP_FLOP"] = "1"

    # Phi3Config from pretrained variant, disable return_dict and caching.
    config = Phi3Config.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = Phi3Config(**config_dict)

    # Load tokenizer and model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(variant, return_tensors="pt", trust_remote_code=True)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model = Phi3ForCausalLM.from_pretrained(variant, trust_remote_code=True, config=config)
    model.eval()

    # input_prompt
    input_prompt = "Africa is an emerging economy because"

    # Tokenize input
    inputs = tokenizer(
        input_prompt,
        return_tensors="pt",
        max_length=256,
        pad_to_max_length=True,
        truncation=True,
    )

    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]

    tt_model = pybuda.PyTorchModule("pt_" + str(variant.split("/")[-1].replace("-", "_")), model)

    verify_module(
        tt_model,
        input_shapes=[(input_ids.shape, attn_mask.shape)],
        inputs=[(input_ids, attn_mask)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            enabled=False if test_device.devtype == pybuda.BackendType.Silicon else True,
        ),
    )


@pytest.mark.parametrize("variant", variants)
def test_phi3_token_classification(test_device, variant):

    # Configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    compiler_cfg.balancer_policy = "Ribbon"

    if test_device.arch == BackendDevice.Grayskull:
        os.environ["PYBUDA_DRAM_PICK_CAPACITY"] = "1"
        os.environ["PYBUDA_DRAM_FLIP_FLOP"] = "1"

    # Phi3Config from pretrained variant, disable return_dict and caching.
    config = Phi3Config.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = Phi3Config(**config_dict)

    # Load tokenizer and model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(variant, return_tensors="pt", trust_remote_code=True)

    model = Phi3ForTokenClassification.from_pretrained(variant, trust_remote_code=True, config=config)
    model.eval()

    # input_prompt
    input_prompt = "HuggingFace is a company based in Paris and New York"

    # Tokenize input
    inputs = tokenizer(input_prompt, return_tensors="pt")

    input_ids = inputs["input_ids"]

    tt_model = pybuda.PyTorchModule("pt_" + str(variant.split("/")[-1].replace("-", "_")), model)

    verify_module(
        tt_model,
        input_shapes=[(input_ids.shape,)],
        inputs=[(input_ids,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            enabled=False if test_device.devtype == pybuda.BackendType.Silicon else True,
        ),
    )


@pytest.mark.parametrize("variant", variants)
def test_phi3_sequence_classification(test_device, variant):

    # Configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    compiler_cfg.balancer_policy = "Ribbon"

    if test_device.arch == BackendDevice.Grayskull:
        os.environ["PYBUDA_DRAM_PICK_CAPACITY"] = "1"
        os.environ["PYBUDA_DRAM_FLIP_FLOP"] = "1"

    # Phi3Config from pretrained variant, disable return_dict and caching.
    config = Phi3Config.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config_dict["pad_token_id"] = None
    config = Phi3Config(**config_dict)

    # Load tokenizer and model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(variant, return_tensors="pt", trust_remote_code=True)
    model = Phi3ForSequenceClassification.from_pretrained(variant, trust_remote_code=True, config=config)
    model.eval()

    # input_prompt
    input_prompt = "the movie was great!"

    # Tokenize input
    inputs = tokenizer(input_prompt, return_tensors="pt")

    input_ids = inputs["input_ids"]

    tt_model = pybuda.PyTorchModule("pt_" + str(variant.split("/")[-1].replace("-", "_")), model)

    verify_module(
        tt_model,
        input_shapes=[(input_ids.shape,)],
        inputs=[(input_ids,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            enabled=False if test_device.devtype == pybuda.BackendType.Silicon else True,
        ),
    )
