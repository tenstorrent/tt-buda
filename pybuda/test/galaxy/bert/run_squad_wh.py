# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import re
import subprocess
import sys
import time
import threading
import queue
import os
import yaml

import torch
import transformers
from squad_preprocessing.helpers.data_processing import (RawResult,
                                                         get_answers,
                                                         torch_df_from_str)
from torch.utils.data import DataLoader
from transformers import BertForQuestionAnswering

from pybuda.config import _get_global_compiler_config
import pybuda
import pytest

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "squad_preprocessing"))
steady_state_samples_per_second = 0


ci_files = {
    "raw_file": "/proj_sw/large-model-cache/bert_squad_data/data/squad/dev-v1.1.json",
    "examples_file": "/proj_sw/large-model-cache/bert_squad_data/preprocessed_data/squad_tokenized/eval_examples.pt",
    "features_file": "/proj_sw/large-model-cache/bert_squad_data/preprocessed_data/squad_tokenized/eval_features.pt",
    "eval_script": "pybuda/test/galaxy/bert/squad_preprocessing/evaluate-v1.1.py",
    "results_file": "results.pt",
    "predictions_file": "predictions.json",
    "out_file": "results.json",
}

ground_truth = {
    "em": 86.91579943, # official GPU scores
    "f1": 93.15638642,    # official GPU scores
    "1_chip_number_ops_per_encoder": 28,
    "1_chip_24_encoder_perf": 155,  # fastest single chip perf observed
    "32_chip_24_encoder_perf": 3720, #fastest single chip perf observed
}

class BertEncoderLMHeadWrapper(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.encoder = model.bert.encoder
        self.lm_head = model.qa_outputs

    def forward(self, hidden_state, attention_mask):
        return self.lm_head(self.encoder(hidden_state, attention_mask).last_hidden_state)

def get_netlist_total_graphs(netlist):
    graphs = []

    for graph in netlist['graphs']:
        graphs.append(graph)

    return graphs

def get_netlist_total_ops(netlist):
    ops = []

    for graph in netlist['graphs']:
        for op in netlist['graphs'][graph]:
            if op != "target_device" and op != "input_count":
                ops.append(op)

    return ops

def encoder_output_buffering_single_chip():
    compiler_cfg = _get_global_compiler_config()
    config = pybuda.config

    # input_1 -> matmul_2, matmul_8, matmul_22
    config.insert_nop("input_1", ["matmul_2", "matmul_8", "matmul_22"], hoist_tms=False)
    config.insert_nop("_fused_op_8", ["matmul_55", "matmul_61", "matmul_75"], hoist_tms=False)
    config.insert_nop("_fused_op_17", ["matmul_108", "matmul_114", "matmul_128"], hoist_tms=False)
    config.insert_nop("_fused_op_26", ["matmul_161", "matmul_167", "matmul_181"], hoist_tms=False)
    config.insert_nop("_fused_op_35", ["matmul_214", "matmul_220", "matmul_234"], hoist_tms=False)
    config.insert_nop("_fused_op_44", ["matmul_267", "matmul_273", "matmul_287"], hoist_tms=False)
    config.insert_nop("_fused_op_53", ["matmul_320", "matmul_326", "matmul_340"], hoist_tms=False)
    config.insert_nop("_fused_op_62", ["matmul_373", "matmul_379", "matmul_393"], hoist_tms=False)
    config.insert_nop("_fused_op_71", ["matmul_426", "matmul_432", "matmul_446"], hoist_tms=False)
    config.insert_nop("_fused_op_80", ["matmul_479", "matmul_485", "matmul_499"], hoist_tms=False)
    config.insert_nop("_fused_op_89", ["matmul_532", "matmul_538", "matmul_552"], hoist_tms=False)
    config.insert_nop("_fused_op_98", ["matmul_585", "matmul_591", "matmul_605"], hoist_tms=False)
    config.insert_nop("_fused_op_107", ["matmul_638", "matmul_644", "matmul_658"], hoist_tms=False)
    config.insert_nop("_fused_op_116", ["matmul_691", "matmul_697", "matmul_711"], hoist_tms=False)
    config.insert_nop("_fused_op_125", ["matmul_744", "matmul_750", "matmul_764"], hoist_tms=False)
    config.insert_nop("_fused_op_134", ["matmul_797", "matmul_803", "matmul_817"], hoist_tms=False)
    config.insert_nop("_fused_op_143", ["matmul_850", "matmul_856", "matmul_870"], hoist_tms=False)
    config.insert_nop("_fused_op_152", ["matmul_903", "matmul_909", "matmul_923"], hoist_tms=False)
    config.insert_nop("_fused_op_161", ["matmul_956", "matmul_962", "matmul_976"], hoist_tms=False)
    config.insert_nop("_fused_op_170", ["matmul_1009", "matmul_1015", "matmul_1029"], hoist_tms=False)
    config.insert_nop("_fused_op_179", ["matmul_1062", "matmul_1068", "matmul_1082"], hoist_tms=False)
    config.insert_nop("_fused_op_188", ["matmul_1115", "matmul_1121", "matmul_1135"], hoist_tms=False)
    config.insert_nop("_fused_op_197", ["matmul_1168", "matmul_1174", "matmul_1188"], hoist_tms=False)
    config.insert_nop("_fused_op_206", ["matmul_1221", "matmul_1227", "matmul_1241"], hoist_tms=False)

    config.override_op_placement(op_name="buffer_0_input_1_matmul_2", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_8_matmul_55", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_17_matmul_108", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_26_matmul_161", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_35_matmul_214", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_44_matmul_267", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_53_matmul_320", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_62_matmul_373", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_71_matmul_426", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_80_matmul_479", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_89_matmul_532", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_98_matmul_585", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_107_matmul_638", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_116_matmul_691", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_125_matmul_744", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_134_matmul_797", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_143_matmul_850", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_152_matmul_903", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_161_matmul_956", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_170_matmul_1009", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_179_matmul_1062", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_188_matmul_1115", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_197_matmul_1168", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_206_matmul_1221", start=[1, 2])

    config.override_op_size("buffer_0_input_1_matmul_2", [1, 1])
    config.override_op_size("buffer_0__fused_op_8_matmul_55", [1, 1])
    config.override_op_size("buffer_0__fused_op_17_matmul_108", [1, 1])
    config.override_op_size("buffer_0__fused_op_26_matmul_161", [1, 1])
    config.override_op_size("buffer_0__fused_op_35_matmul_214", [1, 1])
    config.override_op_size("buffer_0__fused_op_44_matmul_267", [1, 1])
    config.override_op_size("buffer_0__fused_op_53_matmul_320", [1, 1])
    config.override_op_size("buffer_0__fused_op_62_matmul_373", [1, 1])
    config.override_op_size("buffer_0__fused_op_71_matmul_426", [1, 1])
    config.override_op_size("buffer_0__fused_op_80_matmul_479", [1, 1])
    config.override_op_size("buffer_0__fused_op_89_matmul_532", [1, 1])
    config.override_op_size("buffer_0__fused_op_98_matmul_585", [1, 1])
    config.override_op_size("buffer_0__fused_op_107_matmul_638", [1, 1])
    config.override_op_size("buffer_0__fused_op_116_matmul_691", [1, 1])
    config.override_op_size("buffer_0__fused_op_125_matmul_744", [1, 1])
    config.override_op_size("buffer_0__fused_op_134_matmul_797", [1, 1])
    config.override_op_size("buffer_0__fused_op_143_matmul_850", [1, 1])
    config.override_op_size("buffer_0__fused_op_152_matmul_903", [1, 1])
    config.override_op_size("buffer_0__fused_op_161_matmul_956", [1, 1])
    config.override_op_size("buffer_0__fused_op_170_matmul_1009", [1, 1])
    config.override_op_size("buffer_0__fused_op_179_matmul_1062", [1, 1])
    config.override_op_size("buffer_0__fused_op_188_matmul_1115", [1, 1])
    config.override_op_size("buffer_0__fused_op_197_matmul_1168", [1, 1])
    config.override_op_size("buffer_0__fused_op_206_matmul_1221", [1, 1])

def chip_breaks_single_chip():
    config = pybuda.config
    compiler_cfg = _get_global_compiler_config()

    config.set_chip_break("buffer_0__fused_op_8_matmul_55")
    config.set_chip_break("buffer_0__fused_op_8_matmul_55")
    config.set_chip_break("buffer_0__fused_op_17_matmul_108")
    config.set_chip_break("buffer_0__fused_op_26_matmul_161")
    config.set_chip_break("buffer_0__fused_op_35_matmul_214")
    config.set_chip_break("buffer_0__fused_op_44_matmul_267")
    config.set_chip_break("buffer_0__fused_op_53_matmul_320")
    config.set_chip_break("buffer_0__fused_op_62_matmul_373")
    config.set_chip_break("buffer_0__fused_op_71_matmul_426")
    config.set_chip_break("buffer_0__fused_op_80_matmul_479")
    config.set_chip_break("buffer_0__fused_op_89_matmul_532")
    config.set_chip_break("buffer_0__fused_op_98_matmul_585")
    config.set_chip_break("buffer_0__fused_op_107_matmul_638")
    config.set_chip_break("buffer_0__fused_op_116_matmul_691")
    config.set_chip_break("buffer_0__fused_op_125_matmul_744")
    config.set_chip_break("buffer_0__fused_op_134_matmul_797")
    config.set_chip_break("buffer_0__fused_op_143_matmul_850")
    config.set_chip_break("buffer_0__fused_op_152_matmul_903")
    config.set_chip_break("buffer_0__fused_op_161_matmul_956")
    config.set_chip_break("buffer_0__fused_op_170_matmul_1009")
    config.set_chip_break("buffer_0__fused_op_179_matmul_1062")
    config.set_chip_break("buffer_0__fused_op_188_matmul_1115")
    config.set_chip_break("buffer_0__fused_op_197_matmul_1168")
    config.set_chip_break("buffer_0__fused_op_206_matmul_1221")
    config.set_chip_break("matmul_1274_output_nop_0")

def encoder_output_buffering_galaxy():
    compiler_cfg = _get_global_compiler_config()
    config = pybuda.config

    # input_1 -> matmul_2, matmul_8, matmul_22, add_37
    config.insert_nop("input_1", ["matmul_2", "matmul_8", "matmul_22", "add_37"], hoist_tms=False)
    config.insert_nop("_fused_op_8", ["matmul_55", "matmul_61", "matmul_75", "add_90"], hoist_tms=False)
    config.insert_nop("_fused_op_17", ["matmul_108", "matmul_114", "matmul_128", "add_143"], hoist_tms=False)
    config.insert_nop("_fused_op_26", ["matmul_161", "matmul_167", "matmul_181", "add_196"], hoist_tms=False)
    config.insert_nop("_fused_op_35", ["matmul_214", "matmul_220", "matmul_234", "add_249"], hoist_tms=False)
    config.insert_nop("_fused_op_44", ["matmul_267", "matmul_273", "matmul_287", "add_302"], hoist_tms=False)
    config.insert_nop("_fused_op_53", ["matmul_320", "matmul_326", "matmul_340", "add_355"], hoist_tms=False)
    config.insert_nop("_fused_op_62", ["matmul_373", "matmul_379", "matmul_393", "add_408"], hoist_tms=False)
    config.insert_nop("_fused_op_71", ["matmul_426", "matmul_432", "matmul_446", "add_461"], hoist_tms=False)
    config.insert_nop("_fused_op_80", ["matmul_479", "matmul_485", "matmul_499", "add_514"], hoist_tms=False)
    config.insert_nop("_fused_op_89", ["matmul_532", "matmul_538", "matmul_552", "add_567"], hoist_tms=False)
    config.insert_nop("_fused_op_98", ["matmul_585", "matmul_591", "matmul_605", "add_620"], hoist_tms=False)
    config.insert_nop("_fused_op_107", ["matmul_638", "matmul_644", "matmul_658", "add_673"], hoist_tms=False)
    config.insert_nop("_fused_op_116", ["matmul_691", "matmul_697", "matmul_711", "add_726"], hoist_tms=False)
    config.insert_nop("_fused_op_125", ["matmul_744", "matmul_750", "matmul_764", "add_779"], hoist_tms=False)
    config.insert_nop("_fused_op_134", ["matmul_797", "matmul_803", "matmul_817", "add_832"], hoist_tms=False)
    config.insert_nop("_fused_op_143", ["matmul_850", "matmul_856", "matmul_870", "add_885"], hoist_tms=False)
    config.insert_nop("_fused_op_152", ["matmul_903", "matmul_909", "matmul_923", "add_938"], hoist_tms=False)
    config.insert_nop("_fused_op_161", ["matmul_956", "matmul_962", "matmul_976", "add_991"], hoist_tms=False)
    config.insert_nop("_fused_op_170", ["matmul_1009", "matmul_1015", "matmul_1029", "add_1044"], hoist_tms=False)
    config.insert_nop("_fused_op_179", ["matmul_1062", "matmul_1068", "matmul_1082", "add_1097"], hoist_tms=False)
    config.insert_nop("_fused_op_188", ["matmul_1115", "matmul_1121", "matmul_1135", "add_1150"], hoist_tms=False)
    config.insert_nop("_fused_op_197", ["matmul_1168", "matmul_1174", "matmul_1188", "add_1203"], hoist_tms=False)
    config.insert_nop("_fused_op_206", ["matmul_1221", "matmul_1227", "matmul_1241", "add_1256"], hoist_tms=False)

    config.override_op_placement(op_name="buffer_0_input_1_matmul_2", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_8_matmul_55", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_17_matmul_108", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_26_matmul_161", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_35_matmul_214", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_44_matmul_267", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_53_matmul_320", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_62_matmul_373", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_71_matmul_426", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_80_matmul_479", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_89_matmul_532", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_98_matmul_585", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_107_matmul_638", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_116_matmul_691", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_125_matmul_744", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_134_matmul_797", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_143_matmul_850", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_152_matmul_903", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_161_matmul_956", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_170_matmul_1009", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_179_matmul_1062", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_188_matmul_1115", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_197_matmul_1168", start=[1, 2])
    config.override_op_placement(op_name="buffer_0__fused_op_206_matmul_1221", start=[1, 2])

    config.override_op_size("buffer_0_input_1_matmul_2", [2, 1])
    config.override_op_size("buffer_0__fused_op_8_matmul_55", [2, 1])
    config.override_op_size("buffer_0__fused_op_17_matmul_108", [2, 1])
    config.override_op_size("buffer_0__fused_op_26_matmul_161", [2, 1])
    config.override_op_size("buffer_0__fused_op_35_matmul_214", [2, 1])
    config.override_op_size("buffer_0__fused_op_44_matmul_267", [2, 1])
    config.override_op_size("buffer_0__fused_op_53_matmul_320", [2, 1])
    config.override_op_size("buffer_0__fused_op_62_matmul_373", [2, 1])
    config.override_op_size("buffer_0__fused_op_71_matmul_426", [2, 1])
    config.override_op_size("buffer_0__fused_op_80_matmul_479", [2, 1])
    config.override_op_size("buffer_0__fused_op_89_matmul_532", [2, 1])
    config.override_op_size("buffer_0__fused_op_98_matmul_585", [2, 1])
    config.override_op_size("buffer_0__fused_op_107_matmul_638", [2, 1])
    config.override_op_size("buffer_0__fused_op_116_matmul_691", [2, 1])
    config.override_op_size("buffer_0__fused_op_125_matmul_744", [2, 1])
    config.override_op_size("buffer_0__fused_op_134_matmul_797", [2, 1])
    config.override_op_size("buffer_0__fused_op_143_matmul_850", [2, 1])
    config.override_op_size("buffer_0__fused_op_152_matmul_903", [2, 1])
    config.override_op_size("buffer_0__fused_op_161_matmul_956", [2, 1])
    config.override_op_size("buffer_0__fused_op_170_matmul_1009", [2, 1])
    config.override_op_size("buffer_0__fused_op_179_matmul_1062", [2, 1])
    config.override_op_size("buffer_0__fused_op_188_matmul_1115", [2, 1])
    config.override_op_size("buffer_0__fused_op_197_matmul_1168", [2, 1])
    config.override_op_size("buffer_0__fused_op_206_matmul_1221", [2, 1])

    config.override_op_placement("matmul_1274_output_nop_0", start=[6, 6])

def chip_breaks_galaxy():
    compiler_cfg = _get_global_compiler_config()
    config = pybuda.config

    config.set_chip_break("buffer_0__fused_op_8_matmul_55")
    config.set_chip_break("buffer_0__fused_op_8_matmul_55")
    config.set_chip_break("buffer_0__fused_op_17_matmul_108")
    config.set_chip_break("buffer_0__fused_op_26_matmul_161")
    config.set_chip_break("buffer_0__fused_op_35_matmul_214")
    config.set_chip_break("buffer_0__fused_op_44_matmul_267")
    config.set_chip_break("buffer_0__fused_op_53_matmul_320")
    config.set_chip_break("buffer_0__fused_op_62_matmul_373")
    config.set_chip_break("buffer_0__fused_op_71_matmul_426")
    config.set_chip_break("buffer_0__fused_op_80_matmul_479")
    config.set_chip_break("buffer_0__fused_op_89_matmul_532")
    config.set_chip_break("buffer_0__fused_op_98_matmul_585")
    config.set_chip_break("buffer_0__fused_op_107_matmul_638")
    config.set_chip_break("buffer_0__fused_op_116_matmul_691")
    config.set_chip_break("buffer_0__fused_op_125_matmul_744")
    config.set_chip_break("buffer_0__fused_op_134_matmul_797")
    config.set_chip_break("buffer_0__fused_op_143_matmul_850")
    config.set_chip_break("buffer_0__fused_op_152_matmul_903")
    config.set_chip_break("buffer_0__fused_op_161_matmul_956")
    config.set_chip_break("buffer_0__fused_op_170_matmul_1009")
    config.set_chip_break("buffer_0__fused_op_179_matmul_1062")
    config.set_chip_break("buffer_0__fused_op_188_matmul_1115")
    config.set_chip_break("buffer_0__fused_op_197_matmul_1168")
    config.set_chip_break("buffer_0__fused_op_206_matmul_1221")
    config.set_chip_break("matmul_1274_output_nop_0")

def attention_mask_buffering_galaxy():
    compiler_cfg = _get_global_compiler_config()
    config = pybuda.config

    config.insert_nop("attention_mask", ["_fused_op_0", "_fused_op_9", "_fused_op_18", "_fused_op_27", "_fused_op_36", "_fused_op_45", "_fused_op_54", "_fused_op_63", "_fused_op_72", "_fused_op_81", "_fused_op_90", "_fused_op_99", "_fused_op_108", "_fused_op_117", "_fused_op_126", "_fused_op_135", "_fused_op_144", "_fused_op_153", "_fused_op_162", "_fused_op_171", "_fused_op_180", "_fused_op_189", "_fused_op_198", "_fused_op_207"], hoist_tms=False)
    config.insert_nop("buffer_0_attention_mask__fused_op_0", ["_fused_op_9", "_fused_op_18", "_fused_op_27", "_fused_op_36", "_fused_op_45", "_fused_op_54", "_fused_op_63", "_fused_op_72", "_fused_op_81", "_fused_op_90", "_fused_op_99", "_fused_op_108", "_fused_op_117", "_fused_op_126", "_fused_op_135", "_fused_op_144", "_fused_op_153", "_fused_op_162", "_fused_op_171", "_fused_op_180", "_fused_op_189", "_fused_op_198", "_fused_op_207"], hoist_tms=False)
    config.insert_nop("buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9", ["_fused_op_18", "_fused_op_27", "_fused_op_36", "_fused_op_45", "_fused_op_54", "_fused_op_63", "_fused_op_72", "_fused_op_81", "_fused_op_90", "_fused_op_99", "_fused_op_108", "_fused_op_117", "_fused_op_126", "_fused_op_135", "_fused_op_144", "_fused_op_153", "_fused_op_162", "_fused_op_171", "_fused_op_180", "_fused_op_189", "_fused_op_198", "_fused_op_207"], hoist_tms=False)
    config.insert_nop("buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18", ["_fused_op_27", "_fused_op_36", "_fused_op_45", "_fused_op_54", "_fused_op_63", "_fused_op_72", "_fused_op_81", "_fused_op_90", "_fused_op_99", "_fused_op_108", "_fused_op_117", "_fused_op_126", "_fused_op_135", "_fused_op_144", "_fused_op_153", "_fused_op_162", "_fused_op_171", "_fused_op_180", "_fused_op_189", "_fused_op_198", "_fused_op_207"], hoist_tms=False)
    config.insert_nop("buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27", ["_fused_op_36", "_fused_op_45", "_fused_op_54", "_fused_op_63", "_fused_op_72", "_fused_op_81", "_fused_op_90", "_fused_op_99", "_fused_op_108", "_fused_op_117", "_fused_op_126", "_fused_op_135", "_fused_op_144", "_fused_op_153", "_fused_op_162", "_fused_op_171", "_fused_op_180", "_fused_op_189", "_fused_op_198", "_fused_op_207"], hoist_tms=False)
    config.insert_nop("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36", ["_fused_op_45", "_fused_op_54", "_fused_op_63", "_fused_op_72", "_fused_op_81", "_fused_op_90", "_fused_op_99", "_fused_op_108", "_fused_op_117", "_fused_op_126", "_fused_op_135", "_fused_op_144", "_fused_op_153", "_fused_op_162", "_fused_op_171", "_fused_op_180", "_fused_op_189", "_fused_op_198", "_fused_op_207"], hoist_tms=False)
    config.insert_nop("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45", ["_fused_op_54", "_fused_op_63", "_fused_op_72", "_fused_op_81", "_fused_op_90", "_fused_op_99", "_fused_op_108", "_fused_op_117", "_fused_op_126", "_fused_op_135", "_fused_op_144", "_fused_op_153", "_fused_op_162", "_fused_op_171", "_fused_op_180", "_fused_op_189", "_fused_op_198", "_fused_op_207"], hoist_tms=False)
    config.insert_nop("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54", ["_fused_op_63", "_fused_op_72", "_fused_op_81", "_fused_op_90", "_fused_op_99", "_fused_op_108", "_fused_op_117", "_fused_op_126", "_fused_op_135", "_fused_op_144", "_fused_op_153", "_fused_op_162", "_fused_op_171", "_fused_op_180", "_fused_op_189", "_fused_op_198", "_fused_op_207"], hoist_tms=False)
    config.insert_nop("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63", ["_fused_op_72", "_fused_op_81", "_fused_op_90", "_fused_op_99", "_fused_op_108", "_fused_op_117", "_fused_op_126", "_fused_op_135", "_fused_op_144", "_fused_op_153", "_fused_op_162", "_fused_op_171", "_fused_op_180", "_fused_op_189", "_fused_op_198", "_fused_op_207"], hoist_tms=False)
    config.insert_nop("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72", ["_fused_op_81", "_fused_op_90", "_fused_op_99", "_fused_op_108", "_fused_op_117", "_fused_op_126", "_fused_op_135", "_fused_op_144", "_fused_op_153", "_fused_op_162", "_fused_op_171", "_fused_op_180", "_fused_op_189", "_fused_op_198", "_fused_op_207"], hoist_tms=False)
    config.insert_nop("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81", ["_fused_op_90", "_fused_op_99", "_fused_op_108", "_fused_op_117", "_fused_op_126", "_fused_op_135", "_fused_op_144", "_fused_op_153", "_fused_op_162", "_fused_op_171", "_fused_op_180", "_fused_op_189", "_fused_op_198", "_fused_op_207"], hoist_tms=False)
    config.insert_nop("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90", ["_fused_op_99", "_fused_op_108", "_fused_op_117", "_fused_op_126", "_fused_op_135", "_fused_op_144", "_fused_op_153", "_fused_op_162", "_fused_op_171", "_fused_op_180", "_fused_op_189", "_fused_op_198", "_fused_op_207"], hoist_tms=False)
    config.insert_nop("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99", ["_fused_op_108", "_fused_op_117", "_fused_op_126", "_fused_op_135", "_fused_op_144", "_fused_op_153", "_fused_op_162", "_fused_op_171", "_fused_op_180", "_fused_op_189", "_fused_op_198", "_fused_op_207"], hoist_tms=False)
    config.insert_nop("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108", ["_fused_op_117", "_fused_op_126", "_fused_op_135", "_fused_op_144", "_fused_op_153", "_fused_op_162", "_fused_op_171", "_fused_op_180", "_fused_op_189", "_fused_op_198", "_fused_op_207"], hoist_tms=False)
    config.insert_nop("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117", ["_fused_op_126", "_fused_op_135", "_fused_op_144", "_fused_op_153", "_fused_op_162", "_fused_op_171", "_fused_op_180", "_fused_op_189", "_fused_op_198", "_fused_op_207"], hoist_tms=False)
    config.insert_nop("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126", ["_fused_op_135", "_fused_op_144", "_fused_op_153", "_fused_op_162", "_fused_op_171", "_fused_op_180", "_fused_op_189", "_fused_op_198", "_fused_op_207"], hoist_tms=False)
    config.insert_nop("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135", ["_fused_op_144", "_fused_op_153", "_fused_op_162", "_fused_op_171", "_fused_op_180", "_fused_op_189", "_fused_op_198", "_fused_op_207"], hoist_tms=False)
    config.insert_nop("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144", ["_fused_op_153", "_fused_op_162", "_fused_op_171", "_fused_op_180", "_fused_op_189", "_fused_op_198", "_fused_op_207"], hoist_tms=False)
    config.insert_nop("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153", ["_fused_op_162", "_fused_op_171", "_fused_op_180", "_fused_op_189", "_fused_op_198", "_fused_op_207"], hoist_tms=False)
    config.insert_nop("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162", ["_fused_op_171", "_fused_op_180", "_fused_op_189", "_fused_op_198", "_fused_op_207"], hoist_tms=False)
    config.insert_nop("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162__fused_op_171", ["_fused_op_180", "_fused_op_189", "_fused_op_198", "_fused_op_207"], hoist_tms=False)
    config.insert_nop("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162__fused_op_171__fused_op_180", ["_fused_op_189", "_fused_op_198", "_fused_op_207"], hoist_tms=False)
    config.insert_nop("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162__fused_op_171__fused_op_180__fused_op_189", ["_fused_op_198", "_fused_op_207"], hoist_tms=False)
    config.insert_nop("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162__fused_op_171__fused_op_180__fused_op_189__fused_op_198", ["_fused_op_207"], hoist_tms=False)

    config.add_schedule_constraint(["matmul_55", "buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9"])
    config.add_schedule_constraint(["matmul_108", "buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18"])
    config.add_schedule_constraint(["matmul_161", "buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27"])
    config.add_schedule_constraint(["matmul_214", "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36"])
    config.add_schedule_constraint(["matmul_267", "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45"])
    config.add_schedule_constraint(["matmul_320", "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54"])
    config.add_schedule_constraint(["matmul_373", "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63"])
    config.add_schedule_constraint(["matmul_426", "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72"])
    config.add_schedule_constraint(["matmul_479", "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81"])
    config.add_schedule_constraint(["matmul_532", "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90"])
    config.add_schedule_constraint(["matmul_585", "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99"])
    config.add_schedule_constraint(["matmul_638", "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108"])
    config.add_schedule_constraint(["matmul_691", "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117"])
    config.add_schedule_constraint(["matmul_744", "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126"])
    config.add_schedule_constraint(["matmul_797", "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135"])
    config.add_schedule_constraint(["matmul_850", "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144"])
    config.add_schedule_constraint(["matmul_903", "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153"])
    config.add_schedule_constraint(["matmul_956", "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162"])
    config.add_schedule_constraint(["matmul_1009", "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162__fused_op_171"])
    config.add_schedule_constraint(["matmul_1062", "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162__fused_op_171__fused_op_180"])
    config.add_schedule_constraint(["matmul_1115", "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162__fused_op_171__fused_op_180__fused_op_189"])
    config.add_schedule_constraint(["matmul_1168", "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162__fused_op_171__fused_op_180__fused_op_189__fused_op_198"])
    config.add_schedule_constraint(["matmul_1221", "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162__fused_op_171__fused_op_180__fused_op_189__fused_op_198__fused_op_207"])
    
    config.override_op_size("buffer_0_attention_mask__fused_op_0", [1, 1])
    config.override_op_size("buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9", [1, 1])
    config.override_op_size("buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18", [1, 1])
    config.override_op_size("buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27", [1, 1])
    config.override_op_size("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36", [1, 1])
    config.override_op_size("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45", [1, 1])
    config.override_op_size("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54", [1, 1])
    config.override_op_size("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63", [1, 1])
    config.override_op_size("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72", [1, 1])
    config.override_op_size("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81", [1, 1])
    config.override_op_size("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90", [1, 1])
    config.override_op_size("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99", [1, 1])
    config.override_op_size("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108", [1, 1])
    config.override_op_size("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117", [1, 1])
    config.override_op_size("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126", [1, 1])
    config.override_op_size("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135", [1, 1])
    config.override_op_size("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144", [1, 1])
    config.override_op_size("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153", [1, 1])
    config.override_op_size("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162", [1, 1])
    config.override_op_size("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162__fused_op_171", [1, 1])
    config.override_op_size("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162__fused_op_171__fused_op_180", [1, 1])
    config.override_op_size("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162__fused_op_171__fused_op_180__fused_op_189", [1, 1])
    config.override_op_size("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162__fused_op_171__fused_op_180__fused_op_189__fused_op_198", [1, 1])
    config.override_op_size("buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162__fused_op_171__fused_op_180__fused_op_189__fused_op_198__fused_op_207", [1, 1])
    
    config.override_op_placement(op_name="buffer_0_attention_mask__fused_op_0", start=[3, 5])
    config.override_op_placement(op_name="buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9", start=[3, 5])
    config.override_op_placement(op_name="buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18", start=[3, 5])
    config.override_op_placement(op_name="buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27", start=[3, 5])
    config.override_op_placement(op_name="buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36", start=[3, 5])
    config.override_op_placement(op_name="buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45", start=[3, 5])
    config.override_op_placement(op_name="buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54", start=[3, 5])
    config.override_op_placement(op_name="buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63", start=[3, 5])
    config.override_op_placement(op_name="buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72", start=[3, 5])
    config.override_op_placement(op_name="buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81", start=[3, 5])
    config.override_op_placement(op_name="buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90", start=[3, 5])
    config.override_op_placement(op_name="buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99", start=[3, 5])
    config.override_op_placement(op_name="buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108", start=[3, 5])
    config.override_op_placement(op_name="buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117", start=[3, 5])
    config.override_op_placement(op_name="buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126", start=[3, 5])
    config.override_op_placement(op_name="buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135", start=[3, 5])
    config.override_op_placement(op_name="buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144", start=[3, 5])
    config.override_op_placement(op_name="buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153", start=[3, 5])
    config.override_op_placement(op_name="buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162", start=[3, 5])
    config.override_op_placement(op_name="buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162__fused_op_171", start=[3, 5])
    config.override_op_placement(op_name="buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162__fused_op_171__fused_op_180", start=[3, 5])
    config.override_op_placement(op_name="buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162__fused_op_171__fused_op_180__fused_op_189", start=[3, 5])
    config.override_op_placement(op_name="buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162__fused_op_171__fused_op_180__fused_op_189__fused_op_198", start=[3, 5])
    config.override_op_placement(op_name="buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162__fused_op_171__fused_op_180__fused_op_189__fused_op_198__fused_op_207", start=[3, 5])

def intermediate_dram_queues_galaxy():
    buffer_list = [
    "buffer_0_attention_mask__fused_op_0",
    "buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9",
    "buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18",
    "buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27",
    "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36",
    "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45",
    "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54",
    "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63",
    "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72",
    "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81",
    "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90",
    "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99",
    "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108",
    "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117",
    "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126",
    "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135",
    "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144",
    "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153",
    "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162",
    "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162__fused_op_171",
    "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162__fused_op_171__fused_op_180",
    "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162__fused_op_171__fused_op_180__fused_op_189",
    "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162__fused_op_171__fused_op_180__fused_op_189__fused_op_198",
    "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162__fused_op_171__fused_op_180__fused_op_189__fused_op_198__fused_op_207"
    ]

    pybuda.config._get_global_compiler_config().insert_queues = []

    for i in range(len(buffer_list)-1):
        pybuda.config._get_global_compiler_config().insert_queues.append((buffer_list[i], buffer_list[i+1], 0))

    pybuda.config._get_global_compiler_config().insert_queues.append(("buffer_0_input_1_matmul_2", "add_37", 1))
    pybuda.config._get_global_compiler_config().insert_queues.append(("buffer_0__fused_op_8_matmul_55", "add_90", 1))
    pybuda.config._get_global_compiler_config().insert_queues.append(("buffer_0__fused_op_17_matmul_108", "add_143", 1))
    pybuda.config._get_global_compiler_config().insert_queues.append(("buffer_0__fused_op_26_matmul_161", "add_196", 1))
    pybuda.config._get_global_compiler_config().insert_queues.append(("buffer_0__fused_op_35_matmul_214", "add_249", 1))
    pybuda.config._get_global_compiler_config().insert_queues.append(("buffer_0__fused_op_44_matmul_267", "add_302", 1))
    pybuda.config._get_global_compiler_config().insert_queues.append(("buffer_0__fused_op_53_matmul_320", "add_355", 1))
    pybuda.config._get_global_compiler_config().insert_queues.append(("buffer_0__fused_op_62_matmul_373", "add_408", 1))
    pybuda.config._get_global_compiler_config().insert_queues.append(("buffer_0__fused_op_71_matmul_426", "add_461", 1))
    pybuda.config._get_global_compiler_config().insert_queues.append(("buffer_0__fused_op_80_matmul_479", "add_514", 1))
    pybuda.config._get_global_compiler_config().insert_queues.append(("buffer_0__fused_op_89_matmul_532", "add_567", 1))
    pybuda.config._get_global_compiler_config().insert_queues.append(("buffer_0__fused_op_98_matmul_585", "add_620", 1))
    pybuda.config._get_global_compiler_config().insert_queues.append(("buffer_0__fused_op_107_matmul_638", "add_673", 1))
    pybuda.config._get_global_compiler_config().insert_queues.append(("buffer_0__fused_op_116_matmul_691", "add_726", 1))
    pybuda.config._get_global_compiler_config().insert_queues.append(("buffer_0__fused_op_125_matmul_744", "add_779", 1))
    pybuda.config._get_global_compiler_config().insert_queues.append(("buffer_0__fused_op_134_matmul_797", "add_832", 1))
    pybuda.config._get_global_compiler_config().insert_queues.append(("buffer_0__fused_op_143_matmul_850", "add_885", 1))
    pybuda.config._get_global_compiler_config().insert_queues.append(("buffer_0__fused_op_152_matmul_903", "add_938", 1))
    pybuda.config._get_global_compiler_config().insert_queues.append(("buffer_0__fused_op_161_matmul_956", "add_991", 1))
    pybuda.config._get_global_compiler_config().insert_queues.append(("buffer_0__fused_op_170_matmul_1009", "add_1044", 1))
    pybuda.config._get_global_compiler_config().insert_queues.append(("buffer_0__fused_op_179_matmul_1062", "add_1097", 1))
    pybuda.config._get_global_compiler_config().insert_queues.append(("buffer_0__fused_op_188_matmul_1115", "add_1150", 1))
    pybuda.config._get_global_compiler_config().insert_queues.append(("buffer_0__fused_op_197_matmul_1168", "add_1203", 1))
    pybuda.config._get_global_compiler_config().insert_queues.append(("buffer_0__fused_op_206_matmul_1221", "add_1256", 1))

def df_overrides():
    compiler_cfg = _get_global_compiler_config()

    compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
            op_type="matmul",
            name_regex_match="matmul_2",
            math_fidelity=pybuda.MathFidelity.LoFi,
            output_df=pybuda._C.DataFormat.Bfp8_b,
            accumulate_df=pybuda._C.DataFormat.Float16_b,
            intermediate_df=pybuda._C.DataFormat.Bfp8_b,
            input_df= {0: [pybuda._C.DataFormat.Bfp8_b, True], 1: [pybuda._C.DataFormat.Bfp8_b, True], 2: [pybuda._C.DataFormat.Bfp8_b, True]}
        ))
    compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
        op_type="matmul",
        name_regex_match="matmul_8",
        math_fidelity=pybuda.MathFidelity.LoFi,
        output_df=pybuda._C.DataFormat.Bfp8_b,
        accumulate_df=pybuda._C.DataFormat.Float16_b,
        intermediate_df=pybuda._C.DataFormat.Bfp8_b,
        input_df= {0: [pybuda._C.DataFormat.Bfp8_b, True], 1: [pybuda._C.DataFormat.Bfp8_b, True], 2: [pybuda._C.DataFormat.Bfp8_b, True]}
    ))
    compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
            op_type="matmul",
            name_regex_match="matmul_22",
            math_fidelity=pybuda.MathFidelity.LoFi,
            output_df=pybuda._C.DataFormat.Bfp8_b,
            accumulate_df=pybuda._C.DataFormat.Float16_b,
            intermediate_df=pybuda._C.DataFormat.Bfp8_b,
            input_df= {0: [pybuda._C.DataFormat.Bfp8_b, True], 1: [pybuda._C.DataFormat.Bfp8_b, True], 2: [pybuda._C.DataFormat.Bfp8_b, True]}
        ))
    compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
            op_type="add",
            name_regex_match="add_37",
            math_fidelity=pybuda.MathFidelity.HiFi3,
            output_df=pybuda._C.DataFormat.Float16_b,
            accumulate_df=pybuda._C.DataFormat.Float16_b,
            intermediate_df=pybuda._C.DataFormat.Float16_b,
            input_df= {0: [pybuda._C.DataFormat.Bfp8_b, True], 1: [pybuda._C.DataFormat.Bfp8_b, True]}
        ))

    for i in range(24):
        # START
        compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
            op_type="matmul",
            name_regex_match=f"matmul_{55+(53*i)}",
            math_fidelity=pybuda.MathFidelity.LoFi,
            output_df=pybuda._C.DataFormat.Bfp8_b,
            accumulate_df=pybuda._C.DataFormat.Float16_b,
            intermediate_df=pybuda._C.DataFormat.Bfp8_b,
            input_df= {0: [pybuda._C.DataFormat.Float16_b, True], 1: [pybuda._C.DataFormat.Bfp8_b, True], 2: [pybuda._C.DataFormat.Bfp8_b, True]}
        ))
        compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
            op_type="matmul",
            name_regex_match=f"matmul_{61+(53*i)}",
            math_fidelity=pybuda.MathFidelity.LoFi,
            output_df=pybuda._C.DataFormat.Bfp8_b,
            accumulate_df=pybuda._C.DataFormat.Float16_b,
            intermediate_df=pybuda._C.DataFormat.Bfp8_b,
            input_df= {0: [pybuda._C.DataFormat.Bfp8_b, True], 1: [pybuda._C.DataFormat.Bfp8_b, True], 2: [pybuda._C.DataFormat.Bfp8_b, True]}
        ))
        compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
            op_type="matmul",
            name_regex_match=f"matmul_{14+(53*i)}",
            math_fidelity=pybuda.MathFidelity.LoFi,
            output_df=pybuda._C.DataFormat.Bfp8_b,
            accumulate_df=pybuda._C.DataFormat.Float16_b,
            intermediate_df=pybuda._C.DataFormat.Bfp8_b,
            input_df= {0: [pybuda._C.DataFormat.Bfp8_b, True], 1: [pybuda._C.DataFormat.Bfp8_b, True]}
        ))
        compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
            op_type="matmul",
            name_regex_match=f"matmul_{75+(53*i)}",
            math_fidelity=pybuda.MathFidelity.LoFi,
            output_df=pybuda._C.DataFormat.Bfp8_b,
            accumulate_df=pybuda._C.DataFormat.Float16_b,
            intermediate_df=pybuda._C.DataFormat.Bfp8_b,
            input_df= {0: [pybuda._C.DataFormat.Bfp8_b, True], 1: [pybuda._C.DataFormat.Bfp8_b, True], 2: [pybuda._C.DataFormat.Bfp8_b, True]}
        ))
        compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
            op_type="matmul",
            name_regex_match=f"matmul_{29+(53*i)}",
            math_fidelity=pybuda.MathFidelity.LoFi,
            output_df=pybuda._C.DataFormat.Bfp8_b,
            accumulate_df=pybuda._C.DataFormat.Float16_b,
            intermediate_df=pybuda._C.DataFormat.Bfp8_b,
            input_df= {0: [pybuda._C.DataFormat.Bfp8_b, True], 1: [pybuda._C.DataFormat.Bfp8_b, True]}
        ))
        compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
            op_type="matmul",
            name_regex_match=f"matmul_{33+(53*i)}",
            math_fidelity=pybuda.MathFidelity.LoFi,
            output_df=pybuda._C.DataFormat.Bfp8_b,
            accumulate_df=pybuda._C.DataFormat.Float16_b,
            intermediate_df=pybuda._C.DataFormat.Bfp8_b,
            input_df= {0: [pybuda._C.DataFormat.Bfp8_b, True], 1: [pybuda._C.DataFormat.Bfp8_b, True], 2: [pybuda._C.DataFormat.Bfp8_b, True]}
        ))
        compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
            op_type="matmul",
            name_regex_match=f"matmul_{41+(53*i)}",
            math_fidelity=pybuda.MathFidelity.LoFi,
            output_df=pybuda._C.DataFormat.Bfp8_b,
            accumulate_df=pybuda._C.DataFormat.Float16_b,
            intermediate_df=pybuda._C.DataFormat.Bfp8_b,
            input_df= {0: [pybuda._C.DataFormat.Bfp8_b, True], 1: [pybuda._C.DataFormat.Bfp8_b, True], 2: [pybuda._C.DataFormat.Bfp8_b, True]}
        ))
        compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
            op_type="matmul",
            name_regex_match=f"matmul_{47+(53*i)}",
            math_fidelity=pybuda.MathFidelity.LoFi,
            output_df=pybuda._C.DataFormat.Bfp8_b,
            accumulate_df=pybuda._C.DataFormat.Float16_b,
            intermediate_df=pybuda._C.DataFormat.Bfp8_b,
            input_df= {0: [pybuda._C.DataFormat.Bfp8_b, True], 1: [pybuda._C.DataFormat.Bfp8_b, True], 2: [pybuda._C.DataFormat.Bfp8_b, True]}
        ))
        compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
            op_type="add",
            name_regex_match=f"add_{90+(53*i)}",
            math_fidelity=pybuda.MathFidelity.HiFi3,
            output_df=pybuda._C.DataFormat.Float16_b,
            accumulate_df=pybuda._C.DataFormat.Float16_b,
            intermediate_df=pybuda._C.DataFormat.Float16_b,
            input_df= {0: [pybuda._C.DataFormat.Bfp8_b, True], 1: [pybuda._C.DataFormat.Float16_b, True]}
        ))
        compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
            op_type="add",
            name_regex_match=f"add_{51+(53*i)}",
            math_fidelity=pybuda.MathFidelity.HiFi3,
            output_df=pybuda._C.DataFormat.Float16_b,
            accumulate_df=pybuda._C.DataFormat.Float16_b,
            intermediate_df=pybuda._C.DataFormat.Float16_b,
            input_df= {0: [pybuda._C.DataFormat.Bfp8_b, True], 1: [pybuda._C.DataFormat.Bfp8_b, True]}
        ))
        compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
            op_type="fused_op",
            name_regex_match=f"_fused_op_{0+(9*i)}",
            math_fidelity=pybuda.MathFidelity.HiFi3,
            output_df=pybuda._C.DataFormat.Bfp8_b,
            accumulate_df=pybuda._C.DataFormat.Float16_b,
            intermediate_df=pybuda._C.DataFormat.Bfp8_b,
            input_df= {0: [pybuda._C.DataFormat.Bfp8_b, True], 1: [pybuda._C.DataFormat.Float16_b, True], 2: [pybuda._C.DataFormat.Float16_b, True]}
        ))
        compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
            op_type="fused_op",
            name_regex_match=f"_fused_op_{1+(9*i)}",
            math_fidelity=pybuda.MathFidelity.HiFi3,
            output_df=pybuda._C.DataFormat.Float16_b,
            accumulate_df=pybuda._C.DataFormat.Float16_b,
            intermediate_df=pybuda._C.DataFormat.Float16_b,
            input_df= {0: [pybuda._C.DataFormat.Bfp8_b, True], 1: [pybuda._C.DataFormat.Bfp8_b, True]}
        ))
        compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
            op_type="fused_op",
            name_regex_match=f"_fused_op_{2+(9*i)}",
            math_fidelity=pybuda.MathFidelity.HiFi3,
            output_df=pybuda._C.DataFormat.Bfp8_b,
            accumulate_df=pybuda._C.DataFormat.Float16_b,
            intermediate_df=pybuda._C.DataFormat.Bfp8_b,
            input_df= {0: [pybuda._C.DataFormat.Bfp8_b, True], 1: [pybuda._C.DataFormat.Float16_b, True]}
        ))
        compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
            op_type="fused_op",
            name_regex_match=f"_fused_op_{3+(9*i)}",
            math_fidelity=pybuda.MathFidelity.HiFi3,
            output_df=pybuda._C.DataFormat.Float16_b,
            accumulate_df=pybuda._C.DataFormat.Float16_b,
            intermediate_df=pybuda._C.DataFormat.Float16_b,
            input_df= {0: [pybuda._C.DataFormat.Float16_b, True], 1: [pybuda._C.DataFormat.Float16_b, True], 2: [pybuda._C.DataFormat.Float16_b, True]}
        ))
        compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
            op_type="fused_op",
            name_regex_match=f"_fused_op_{4+(9*i)}",
            math_fidelity=pybuda.MathFidelity.HiFi3,
            output_df=pybuda._C.DataFormat.Float16_b,
            accumulate_df=pybuda._C.DataFormat.Float16_b,
            intermediate_df=pybuda._C.DataFormat.Float16_b,
            input_df= {0: [pybuda._C.DataFormat.Bfp8_b, True], 1: [pybuda._C.DataFormat.Bfp8_b, True], 2: [pybuda._C.DataFormat.Bfp8_b, True]}
        ))
        compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
            op_type="fused_op",
            name_regex_match=f"_fused_op_{5+(9*i)}",
            math_fidelity=pybuda.MathFidelity.HiFi3,
            output_df=pybuda._C.DataFormat.Bfp8_b,
            accumulate_df=pybuda._C.DataFormat.Float16_b,
            intermediate_df=pybuda._C.DataFormat.Float16_b,
            input_df= {0: [pybuda._C.DataFormat.Bfp8_b, True], 1: [pybuda._C.DataFormat.Float16_b, True], 2: [pybuda._C.DataFormat.Float16_b, True], 3: [pybuda._C.DataFormat.Float16_b, True]}
        ))
        compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
            op_type="fused_op",
            name_regex_match=f"_fused_op_{6+(9*i)}",
            math_fidelity=pybuda.MathFidelity.HiFi3,
            output_df=pybuda._C.DataFormat.Bfp8_b,
            accumulate_df=pybuda._C.DataFormat.Float16_b,
            intermediate_df=pybuda._C.DataFormat.Float16_b,
            input_df= {0: [pybuda._C.DataFormat.Float16_b, True], 1: [pybuda._C.DataFormat.Float16_b, True], 2: [pybuda._C.DataFormat.Float16_b, True]}
        ))
        compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
            op_type="fused_op",
            name_regex_match=f"_fused_op_{7+(9*i)}",
            math_fidelity=pybuda.MathFidelity.HiFi3,
            output_df=pybuda._C.DataFormat.Float16_b,
            accumulate_df=pybuda._C.DataFormat.Float16_b,
            intermediate_df=pybuda._C.DataFormat.Float16_b,
            input_df= {0: [pybuda._C.DataFormat.Bfp8_b, True], 1: [pybuda._C.DataFormat.Bfp8_b, True], 2: [pybuda._C.DataFormat.Bfp8_b, True]}
        ))
        compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
            op_type="fused_op",
            name_regex_match=f"_fused_op_{8+(9*i)}",
            math_fidelity=pybuda.MathFidelity.HiFi3,
            output_df=pybuda._C.DataFormat.Bfp8_b,
            accumulate_df=pybuda._C.DataFormat.Float16_b,
            intermediate_df=pybuda._C.DataFormat.Float16_b,
            input_df= {0: [pybuda._C.DataFormat.Bfp8_b, True], 1: [pybuda._C.DataFormat.Float16_b, True], 2: [pybuda._C.DataFormat.Float16_b, True], 3: [pybuda._C.DataFormat.Float16_b, True]}
        ))
        compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
            op_type="matmul",
            name_regex_match=f"softmax_{18+(53*i)}.dc.reduce_sum.1.lc1",
            math_fidelity=pybuda.MathFidelity.LoFi,
            output_df=pybuda._C.DataFormat.Bfp8_b,
            accumulate_df=pybuda._C.DataFormat.Float16_b,
            intermediate_df=pybuda._C.DataFormat.Bfp8_b,
            input_df= {0: [pybuda._C.DataFormat.Bfp8_b, True], 1: [pybuda._C.DataFormat.Bfp8_b, True]}
        ))
        compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
            op_type="matmul",
            name_regex_match=f"layernorm_{38+(53*i)}.dc.reduce_sum.0.lc1",
            math_fidelity=pybuda.MathFidelity.HiFi3,
            output_df=pybuda._C.DataFormat.Float16_b,
            accumulate_df=pybuda._C.DataFormat.Float16_b,
            intermediate_df=pybuda._C.DataFormat.Float16_b,
            input_df= {0: [pybuda._C.DataFormat.Float16_b, True], 1: [pybuda._C.DataFormat.Float16_b, True]}
        ))
        compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
            op_type="matmul",
            name_regex_match=f"layernorm_{38+(53*i)}.dc.multiply.4",
            math_fidelity=pybuda.MathFidelity.HiFi3,
            output_df=pybuda._C.DataFormat.Float16_b,
            accumulate_df=pybuda._C.DataFormat.Float16_b,
            intermediate_df=pybuda._C.DataFormat.Float16_b,
            input_df= {0: [pybuda._C.DataFormat.Bfp8_b, True], 1: [pybuda._C.DataFormat.Bfp8_b, True]}
        ))
        compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
            op_type="matmul",
            name_regex_match=f"layernorm_{38+(53*i)}.dc.reduce_sum.5.lc1",
            math_fidelity=pybuda.MathFidelity.HiFi3,
            output_df=pybuda._C.DataFormat.Bfp8_b,
            accumulate_df=pybuda._C.DataFormat.Float16_b,
            intermediate_df=pybuda._C.DataFormat.Float16_b,
            input_df= {0: [pybuda._C.DataFormat.Float16_b, True], 1: [pybuda._C.DataFormat.Float16_b, True]}
        ))
        compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
            op_type="matmul",
            name_regex_match=f"layernorm_{52+(53*i)}.dc.reduce_sum.0.lc1",
            math_fidelity=pybuda.MathFidelity.HiFi3,
            output_df=pybuda._C.DataFormat.Float16_b,
            accumulate_df=pybuda._C.DataFormat.Float16_b,
            intermediate_df=pybuda._C.DataFormat.Float16_b,
            input_df= {0: [pybuda._C.DataFormat.Float16_b, True], 1: [pybuda._C.DataFormat.Float16_b, True]}
        ))
        compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
            op_type="matmul",
            name_regex_match=f"layernorm_{52+(53*i)}.dc.multiply.4",
            math_fidelity=pybuda.MathFidelity.HiFi3,
            output_df=pybuda._C.DataFormat.Float16_b,
            accumulate_df=pybuda._C.DataFormat.Float16_b,
            intermediate_df=pybuda._C.DataFormat.Float16_b,
            input_df= {0: [pybuda._C.DataFormat.Bfp8_b, True], 1: [pybuda._C.DataFormat.Bfp8_b, True]}
        ))
        compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
            op_type="matmul",
            name_regex_match=f"layernorm_{52+(53*i)}.dc.reduce_sum.5.lc1",
            math_fidelity=pybuda.MathFidelity.HiFi3,
            output_df=pybuda._C.DataFormat.Bfp8_b,
            accumulate_df=pybuda._C.DataFormat.Float16_b,
            intermediate_df=pybuda._C.DataFormat.Float16_b,
            input_df= {0: [pybuda._C.DataFormat.Float16_b, True], 1: [pybuda._C.DataFormat.Float16_b, True]}
        ))

    # Set data format for LM head
    compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
            op_type="matmul",
            name_regex_match="matmul_1274",
            math_fidelity=pybuda.MathFidelity.LoFi,
            output_df=pybuda._C.DataFormat.Bfp8_b,
            accumulate_df=pybuda._C.DataFormat.Float16_b,
            intermediate_df=pybuda._C.DataFormat.Bfp8_b,
            input_df= {0: [pybuda._C.DataFormat.Bfp8_b, True], 1: [pybuda._C.DataFormat.Bfp8_b, True], 2: [pybuda._C.DataFormat.Bfp8_b, True]}
        ))

    # Set data format for final output
    compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
                op_type="nop",
                name_regex_match="matmul_1274_output_nop_0",
                output_df=pybuda._C.DataFormat.Float16_b,
                accumulate_df=pybuda._C.DataFormat.Float16_b,
                intermediate_df=pybuda._C.DataFormat.Float16_b,
                input_df= {0: [pybuda._C.DataFormat.Bfp8_b, True]}
            ))

    # Set data format for additional buffers
    for i in range(24):
        compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
                op_type="nop",
                name_regex_match=f"buffer_0__fused_op_{0+(9*i)}__fused_op_{2+(9*i)}",
                output_df=pybuda._C.DataFormat.Bfp8_b,
                accumulate_df=pybuda._C.DataFormat.Float16_b,
                intermediate_df=pybuda._C.DataFormat.Float16_b,
                input_df= {0: [pybuda._C.DataFormat.Bfp8_b, True]}
            ))

    for i in range(24):
        compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
                op_type="nop",
                name_regex_match=f"buffer_0__fused_op_{5+(9*i)}_add_{51+(53*i)}",
                output_df=pybuda._C.DataFormat.Bfp8_b,
                accumulate_df=pybuda._C.DataFormat.Bfp8_b,
                intermediate_df=pybuda._C.DataFormat.Bfp8_b,
                input_df= {0: [pybuda._C.DataFormat.Bfp8_b, True]}
            ))

    # Mixed precision settings below
    # Set output of each encoder data format
    encoder_output_list = [
        "buffer_0__fused_op_8_matmul_55",
        "buffer_0__fused_op_17_matmul_108",
        "buffer_0__fused_op_26_matmul_161",
        "buffer_0__fused_op_35_matmul_214",
        "buffer_0__fused_op_44_matmul_267",
        "buffer_0__fused_op_53_matmul_320",
        "buffer_0__fused_op_62_matmul_373",
        "buffer_0__fused_op_71_matmul_426",
        "buffer_0__fused_op_80_matmul_479",
        "buffer_0__fused_op_89_matmul_532",
        "buffer_0__fused_op_98_matmul_585",
        "buffer_0__fused_op_107_matmul_638",
        "buffer_0__fused_op_116_matmul_691",
        "buffer_0__fused_op_125_matmul_744",
        "buffer_0__fused_op_134_matmul_797",
        "buffer_0__fused_op_143_matmul_850",
        "buffer_0__fused_op_152_matmul_903",
        "buffer_0__fused_op_161_matmul_956",
        "buffer_0__fused_op_170_matmul_1009",
        "buffer_0__fused_op_179_matmul_1062",
        "buffer_0__fused_op_188_matmul_1115",
        "buffer_0__fused_op_197_matmul_1168",
        "buffer_0__fused_op_206_matmul_1221"
    ]

    for buffer in encoder_output_list:
        compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
                op_type="nop",
                name_regex_match=buffer,
                output_df=pybuda._C.DataFormat.Float16_b,
                accumulate_df=pybuda._C.DataFormat.Float16_b,
                intermediate_df=pybuda._C.DataFormat.Float16_b,
                input_df= {0: [pybuda._C.DataFormat.Float16_b, True]}
            ))

    # Set data format for attention mask buffer
    attention_buffer_list = [
        "buffer_0_attention_mask__fused_op_0",
        "buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9",
        "buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18",
        "buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27",
        "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36",
        "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45",
        "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54",
        "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63",
        "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72",
        "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81",
        "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90",
        "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99",
        "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108",
        "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117",
        "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126",
        "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135",
        "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144",
        "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153",
        "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162",
        "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162__fused_op_171",
        "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162__fused_op_171__fused_op_180",
        "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162__fused_op_171__fused_op_180__fused_op_189",
        "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162__fused_op_171__fused_op_180__fused_op_189__fused_op_198",
        "buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_buffer_0_attention_mask__fused_op_0__fused_op_9__fused_op_18__fused_op_27__fused_op_36__fused_op_45__fused_op_54__fused_op_63__fused_op_72__fused_op_81__fused_op_90__fused_op_99__fused_op_108__fused_op_117__fused_op_126__fused_op_135__fused_op_144__fused_op_153__fused_op_162__fused_op_171__fused_op_180__fused_op_189__fused_op_198__fused_op_207"
    ]

    for buffer in attention_buffer_list:
        compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
                op_type="nop",
                name_regex_match=buffer,
                output_df=pybuda._C.DataFormat.Float16_b,
                accumulate_df=pybuda._C.DataFormat.Float16_b,
                intermediate_df=pybuda._C.DataFormat.Float16_b,
                input_df= {0: [pybuda._C.DataFormat.Float16_b, True]}
            ))

    # Set data format for input buffer
    input_buffer_list = [
        "buffer_0_input_1_matmul_2",
    ]

    for buffer in input_buffer_list:
        compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
                op_type="nop",
                name_regex_match=buffer,
                output_df=pybuda._C.DataFormat.Bfp8_b,
                accumulate_df=pybuda._C.DataFormat.Bfp8_b,
                intermediate_df=pybuda._C.DataFormat.Bfp8_b,
                input_df= {0: [pybuda._C.DataFormat.Bfp8_b, True]}
            ))

def op_overrides(parameters):
    config = pybuda.config
    compiler_cfg = _get_global_compiler_config()

    for i in range(parameters["num_encoders"]):
        # matmul_2
        config.override_op_size(f"matmul_{2+53*i}", [3, 1])
        config.override_op_placement(op_name=f"matmul_{2+53*i}", start=[0, 0])
        # matmul_8
        config.override_op_size(f"matmul_{8+53*i}", [3, 1])
        config.override_op_placement(op_name=f"matmul_{8+53*i}", start=[0, 1])
        # matmul_14
        config.override_op_size(f"matmul_{14+53*i}", [1, 1])
        config.override_op_placement(op_name=f"matmul_{14+53*i}", start=[0, 2])
        # matmul_22
        config.override_op_size(f"matmul_{22+53*i}", [2, 2])
        config.override_op_placement(op_name=f"matmul_{22+53*i}", start=[4, 5])
        # matmul_29
        config.override_op_size(f"matmul_{29+53*i}", [1, 1])
        config.override_op_placement(op_name=f"matmul_{29+53*i}", start=[2, 7])
        # matmul_33
        config.override_op_size(f"matmul_{33+53*i}", [2, 2])
        config.override_op_placement(op_name=f"matmul_{33+53*i}", start=[3, 0])
        # matmul_41
        config.override_op_size(f"matmul_{41+53*i}", [3, 4])
        config.override_op_placement(op_name=f"matmul_{41+53*i}", start=[6, 0])
        # matmul_47
        config.override_op_size(f"matmul_{47+53*i}", [2, 4])
        config.override_op_placement(op_name=f"matmul_{47+53*i}", start=[6, 4])
        
        #if parameters["num_chips"] == 1:
        #    config.set_chip_break(f"matmul_{55+53*i}")

        # fused_op_0
        config.override_op_size(f"_fused_op_{0+9*i}", [3, 3])
        config.override_op_placement(op_name=f"_fused_op_{0+9*i}", start=[0, 3], transpose_op=True)
        # fused_op_1
        config.override_op_size(f"_fused_op_{1+9*i}", [2, 1])
        config.override_op_placement(op_name=f"_fused_op_{1+9*i}", start=[2, 6])
        # fused_op_2
        config.override_op_size(f"_fused_op_{2+9*i}", [2, 1])
        config.override_op_placement(op_name=f"_fused_op_{2+9*i}", start=[0, 7])
        # fused_op_3
        config.override_op_size(f"_fused_op_{3+9*i}", [2, 1])
        config.override_op_placement(op_name=f"_fused_op_{3+9*i}", start=[4, 3], transpose_op=True)
        # fused_op_4
        config.override_op_size(f"_fused_op_{4+9*i}", [1, 1])
        config.override_op_placement(op_name=f"_fused_op_{4+9*i}", start=[5, 7], transpose_op=True)
        # fused_op_5
        config.override_op_size(f"_fused_op_{5+9*i}", [2, 1])
        config.override_op_placement(op_name=f"_fused_op_{5+9*i}", start=[5, 3], transpose_op=True)
        # fused_op_6
        config.override_op_size(f"_fused_op_{6+9*i}", [2, 1])
        config.override_op_placement(op_name=f"_fused_op_{6+9*i}", start=[8, 5])
        # fused_op_7
        config.override_op_size(f"_fused_op_{7+9*i}", [1, 1])
        config.override_op_placement(op_name=f"_fused_op_{7+9*i}", start=[9, 2], transpose_op=True)
        # fused_op_8
        config.override_op_size(f"_fused_op_{8+9*i}", [2, 1])
        config.override_op_placement(op_name=f"_fused_op_{8+9*i}", start=[9, 0], transpose_op=True)

        # layernorms
        config.override_op_size(f"layernorm_{38+53*i}.dc.reduce_sum.0.lc1", [1, 1])
        config.override_op_placement(op_name=f"layernorm_{38+53*i}.dc.reduce_sum.0.lc1", start=[9, 7])
        config.override_op_size(f"layernorm_{38+53*i}.dc.reduce_sum.5.lc1", [1, 1])
        config.override_op_placement(op_name=f"layernorm_{38+53*i}.dc.reduce_sum.5.lc1", start=[5, 2])
        config.override_op_size(f"layernorm_{52+53*i}.dc.reduce_sum.0.lc1", [1, 1])
        config.override_op_placement(op_name=f"layernorm_{52+53*i}.dc.reduce_sum.0.lc1", start=[9, 6])
        config.override_op_size(f"layernorm_{52+53*i}.dc.reduce_sum.5.lc1", [1, 1])
        config.override_op_placement(op_name=f"layernorm_{52+53*i}.dc.reduce_sum.5.lc1", start=[9, 3])

        config.override_op_size(f"layernorm_{38+53*i}.dc.multiply.4", [2, 1])
        config.override_op_placement(op_name=f"layernorm_{38+53*i}.dc.multiply.4", start=[5, 0], transpose_op=True)
        config.override_op_size(f"layernorm_{52+53*i}.dc.multiply.4", [2, 1])
        config.override_op_placement(op_name=f"layernorm_{52+53*i}.dc.multiply.4", start=[8, 4])

        # add ops
        config.override_op_size(f"add_{37+53*i}", [2, 1])
        config.override_op_placement(op_name=f"add_{37+53*i}", start=[3, 2])
        config.override_op_size(f"add_{51+53*i}", [2, 1])
        config.override_op_placement(op_name=f"add_{51+53*i}", start=[3, 7])

        # softmax reduce_sum
        config.override_op_size(f"softmax_{18+53*i}.dc.reduce_sum.1.lc1", [2, 1])
        config.override_op_placement(op_name=f"softmax_{18+53*i}.dc.reduce_sum.1.lc1", start=[0, 6])
        
        # buffer insertion:
        # attention_mask -> 
        # fused_op_0 -> fused_op_2 buffer insertion
        config.insert_nop(f"_fused_op_{0+9*i}", [f"_fused_op_{2+9*i}"], hoist_tms=False)
        config.override_op_size(f"buffer_0__fused_op_{0+9*i}__fused_op_{2+9*i}", [2, 1])
        config.override_op_placement(op_name=f"buffer_0__fused_op_{0+9*i}__fused_op_{2+9*i}", start=[3, 3], transpose_op=True)

        # fused_op_5 -> add_51 buffer insertion
        config.insert_nop(f"_fused_op_{5+9*i}", [f"add_{51+53*i}"], hoist_tms=False)
        config.override_op_size(f"buffer_0__fused_op_{5+9*i}_add_{51+53*i}", [1, 2])
        config.override_op_placement(op_name=f"buffer_0__fused_op_{5+9*i}_add_{51+53*i}", start=[8, 6])

def apply_overrides(parameters):
    pybuda.set_configuration_options(
            math_fidelity=pybuda.MathFidelity.HiFi3,
            backend_opt_level=4,
            enable_auto_fusing=True,
            enable_auto_transposing_placement=False,
            accumulate_df=pybuda._C.DataFormat.Float16_b,
            #performance_trace=pybuda.PerfTraceLevel.LIGHT
            )

    if parameters["num_chips"] >= 32:
        df_overrides()
        
    op_overrides(parameters)

    # Add in additional buffering
    if parameters["num_chips"] == 1:
        pass
        #encoder_output_buffering_single_chip()
    elif parameters["num_chips"] >= 32:
        encoder_output_buffering_galaxy()
        attention_mask_buffering_galaxy()
        intermediate_dram_queues_galaxy()

    # Add in chip breaks for galaxy
    if parameters["num_chips"] >= 32:
        chip_breaks_galaxy()

def main(parameters):
    # Apply environment variables
    os.environ['PYBUDA_EXP_APPROX'] = '1'
    os.environ['PYBUDA_FUSE_OPS'] = '1'
    os.environ['PYBUDA_NLP_MANUAL_TARGET'] = '185000'
    os.environ['PYBUDA_DISABLE_DRAM0'] = '1'
    os.environ['PYBUDA_FORCE_INTERMED_TO_OUTPUT_DF'] = '1'
    os.environ['PYBUDA_EXTRA_L1_MARGIN'] = '131072'
    os.environ['PYBUDA_DISABLE_FORK_JOIN_NOPS'] = '1'
    os.environ['ENABLE_ETH_SERIALIZATON'] = '1'
    os.environ['TT_BACKEND_PUSH_TIMEOUT'] = '500'
    os.environ['TT_BACKEND_TIMEOUT'] = '500'
    os.environ['TT_BACKEND_GET_TIMEOUT'] = '500'
    os.environ['TT_BACKEND_POP_TIMEOUT'] = '500'
    os.environ['PYBUDA_MIN_MATMUL_BUFFER_ALLOW_IN1'] = '1'
    os.environ['PYBUDA_FUSE_STOP_ON_RECIPROCAL'] = '1'
    os.environ['PYBUDA_FUSE_MATMUL_GELU'] = '1'
    os.environ['PYBUDA_DISABLE_STABLE_SOFTMAX'] = '1'
    os.environ['PYBUDA_DISABLE_DYNAMIC_DRAM'] = '1'

    input_sleep_timer = 0
    if parameters["num_chips"] == 1:
        input_sleep_timer = 1

    if parameters["num_chips"] >= 32:
        os.environ['PYBUDA_MICROBATCH_LOOPING'] = '1'

    if parameters["num_chips"] > 1 and parameters["num_chips"] % 2 != 0:
        os.environ['PYBUDA_NEBULA_GALAXY_PLACER'] = '1'

    # Update parameters object with cached file locations
    parameters["raw_file"] = ci_files['raw_file']
    parameters["examples_file"] = ci_files['examples_file']
    parameters["features_file"] = ci_files['features_file']
    parameters["results_file"] = ci_files['results_file']
    parameters["eval_script"] = ci_files['eval_script']
    parameters["predictions_file"] = ci_files['predictions_file']
    parameters["out_file"] = ci_files['out_file']
    
    # Values to collect
    exact_match = None
    f1_score = None

    # Define num encoders
    num_encoders = parameters["num_encoders"]

    # Load Bert model from HuggingFace
    model = BertForQuestionAnswering.from_pretrained(parameters["hf_model"]).to(dtype=torch_df_from_str(parameters["data_type"]))
    model.bert.encoder.layer = model.bert.encoder.layer[:num_encoders]
    model.eval()

    # Load data
    try:
        eval_features = torch.load(parameters["features_file"])
    except FileNotFoundError:
        print("Features file not found. Run SQuAD preprocessing.")
    try:
        eval_examples = torch.load(parameters["examples_file"])
    except FileNotFoundError:
        print("Examples file not found. Run SQuAD preprocessing.")

    # Prepare data
    embeddings = []
    unique_ids = []
    attention_masks = []
    for feature in eval_features:
        embeddings.append(feature.embeddings)
        unique_ids.append(feature.unique_id)
        attention_masks.append(feature.input_mask.reshape(1,1,-1))

    if parameters["quiet"]:
        os.environ['LOGURU_LEVEL'] = 'ERROR'
        os.environ['LOGGER_LEVEL'] = 'Error'
        from loguru import logger
        logger.remove()
        logger.add(sys.stderr, level="ERROR")

    # Create DataLoader
    batch_size = parameters["batch_size"]
    generator = DataLoader(
        list(zip(embeddings, attention_masks)), batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0
    )
    num_batches = len(generator)

    # Segment model
    model.eval()
    model = BertEncoderLMHeadWrapper(model)

    if parameters["num_chips"] == 65:
        galaxy_chips = [0, 1, 16, 17, 14, 13, 12, 11, 18, 56, 57, 58, 55, 19, 20, 54, 53, 21, 24, 25, 26, 39, 38, 42, 47, 48, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 22, 23, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 40, 41, 43, 44, 45, 46, 49, 50, 51, 52, 59, 60, 61, 62, 63, 64]
    elif parameters["num_chips"] == 64:
        galaxy_chips = [0, 15, 16, 13, 12, 11, 10, 17, 55, 56, 57, 54, 18, 19, 53, 52, 20, 23, 24, 39, 2, 40, 38, 37, 41, 42, 43, 44, 45, 46, 47, 1, 3, 4, 5, 6, 7, 8, 9, 14, 21, 22, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 48, 49, 50, 51, 58, 59, 60, 61, 62, 63]
    elif parameters["num_chips"] == 33:
        galaxy_chips = [0, 1, 31, 32, 29, 28, 27, 7, 6, 30, 4, 5, 8, 9, 18, 3, 19, 17, 10, 11, 16, 20, 21, 15, 12, 13, 14, 22, 23, 24, 25, 26, 2]
    elif parameters["num_chips"] == 32:
        galaxy_chips = [0, 30, 31, 28, 27, 26, 6, 5, 29, 3, 4, 7, 8, 17, 2, 18, 16, 9, 10, 15, 19, 20, 14, 11, 12, 13, 21, 22, 23, 24, 25, 1]
    elif parameters["num_chips"] == 2:
        galaxy_chips = [0, 1]
    else:
        galaxy_chips = [0]

    tt0 = pybuda.TTDevice('tt0',
                          module=pybuda.PyTorchModule("bert_squad", model),
                          chip_ids=galaxy_chips,
                          fp32_fallback=pybuda.DataFormat.Float16,
                          devtype=pybuda.BackendType.Silicon,
                          arch=pybuda.BackendDevice.Wormhole_B0)

    compiler_cfg = _get_global_compiler_config()

    pybuda.config.set_configuration_options(default_df_override=pybuda.DataFormat.Float16_b)

    # Apply overrides
    apply_overrides(parameters)

    output_queue = pybuda.initialize_pipeline(training=False,
                                              sample_inputs=(embeddings[0].expand(batch_size, *embeddings[0].size()[1:]).to(dtype=torch.float32),
                                                             attention_masks[0].expand(batch_size, *attention_masks[0].size()[1:]).to(dtype=torch.float32)),
                                              microbatch_count=batch_size)

    if parameters["quiet"]:
        os.system('clear')
    print('Compilation complete.')
    print()

    while True:
        if parameters["wait_for_user"]:
            cmd = input("Press Enter to start benchmark... (q to quit)\n")
            if cmd in ['q', 'quit', 'exit']:
                break

        # Prepare store objects
        results = []
        start_logits = []
        end_logits = []

        start_str = f" Starting benchmark at {time.asctime(time.localtime(time.time()))} "
        print()
        print('*' * len(start_str))
        print(start_str)
        print('*' * len(start_str))
        pybuda.sync()

        # Run inference
        # -----------------------------------------------------------------------
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ UPDATE FOR GALAXY ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # -----------------------------------------------------------------------

        num_batches = len(generator)
        batch_starts = []
        batch_ends = []

        def push_inputs_thread():
            with torch.inference_mode():
                # Warmup loop
                for batch in generator:
                    #print("Pushing input")
                    tt0.push_to_inputs(batch)
                    #print("Pushed input")
                    batch_starts.append(time.time())
                    time.sleep(input_sleep_timer)
                
                # Benchmark loop
                for batch in generator:
                    for _ in range(parameters["loops"]):
                        #print("Pushing input1")
                        tt0.push_to_inputs(batch)
                        #print("Pushed input1")
                        batch_starts.append(time.time())
                        time.sleep(input_sleep_timer)

        def pop_outputs_thread():
            global steady_state_samples_per_second
            print()
            for loop in range(parameters["loops"] + 1): # add a warmup loop
                loop_start = time.time()
                parameters_loops = parameters["loops"]
                loop_name = "Warmup" if loop == 0 else f"Loop {str(loop).rjust(len(str(parameters_loops)))}"
                for i in range(num_batches):
                    while True:
                        try:
                            #print("Reading output", flush=True)
                            output = output_queue.get(timeout=5)
                            #print("Done reading output", flush=True)
                            output = output[0].value()
                            batch_ends.append(time.time())

                            idx_0 = len(batch_ends) - (i+1)
                            idx_i = len(batch_ends)

                            ends = batch_ends[idx_0:idx_i]
                            starts = batch_starts[idx_0:idx_i]

                            assert len(ends) == len(starts)

                            latency_sum = sum(ends[i] - starts[i] for i in range(len(ends)))
                            mean_latency_ms = (1000 * latency_sum) / (i + 1)

                            loop_current = time.time() - loop_start
                            inputs_per_second = batch_size * (i + 1) / loop_current
                            steady_state_samples_per_second = inputs_per_second
                            # Update a rudimentary progress bar with the current rate/s and then '=' for each input for the current loop
                            pre = '\r' if parameters["quiet"] else ''
                            end = ''   if parameters["quiet"] else '\n'
                            latency_str = f"{mean_latency_ms:3.0f} ms " if parameters["pybuda_latency"] else " "
                            print(f"{pre}{loop_name} @ {inputs_per_second:4.0f} seq/s, {latency_str}|{'=' * (i + 1)}{' ' * (num_batches - i - 1)}|", end=end, flush=True)

                            if loop == 0: # only record outputs for one loop to calculate EM + F1
                                start_logits.extend(output[:, :, 0].detach().cpu().tolist())
                                end_logits.extend(output[:, :, 1].detach().cpu().tolist())
                            break                        
                        except queue.Empty as _:
                            if pybuda.error_raised():
                                print(" * Aborting output thread due to error")
                                return
                if parameters["quiet"]:
                    print()
            print()
        
        output_thread = threading.Thread(target=pop_outputs_thread)
        output_thread.start()
        pybuda.sync()

        input_thread = threading.Thread(target=push_inputs_thread)
        input_thread.start()
        time.sleep(2)

        start_time = time.time()

        #for _ in range(parameters["loops"] + 1): # add a warmup loop
        #    for _ in range(num_batches):
        #        pybuda.run_forward(input_count=1)
        pybuda.run_forward(input_count=(num_batches * (parameters["loops"] + 1)))

        input_thread.join()
        output_thread.join()

        pybuda.sync()

        end_time = time.time()

        end_str = f" Ending benchmark at {time.asctime(time.localtime(time.time()))} "
        print('*' * len(end_str))
        print(end_str)

        # Data postprocessing
        try:
            for idx, unique_id in enumerate(unique_ids):
                results.append(RawResult(unique_id=unique_id, start_logits=start_logits[idx], end_logits=end_logits[idx]))
        except IndexError:
            pass
        torch.save(results, parameters["results_file"])

        # Get SQuADv1.1 answers
        answers, _ = get_answers(
            examples=eval_examples,
            features=eval_features,
            results=results,
            n_best_size=parameters["n_best_size"],
            max_answer_length=parameters["max_answer_length"],
            do_lower_case=True
            )
        with open(parameters["predictions_file"], "w", encoding="utf-8") as file:
            file.write(json.dumps(answers, indent=4) + "\n")

        # Evaluation metrics
        total_time = end_time - start_time
        total_samples = batch_size * len(generator) * (parameters["loops"] + 1) # include warmup loop - TODO: time without warmup or remove overall timings
        stderr = subprocess.DEVNULL if parameters["quiet"] else None # suppress stderr if quiet
        eval_out = subprocess.check_output([sys.executable, parameters["eval_script"], parameters["raw_file"], parameters["predictions_file"]], stderr=stderr)
        eval_out = eval_out.decode()
        eval_out = re.findall(r"\d+\.\d+", eval_out)
        eval_results = {"exact_match": float(eval_out[0]), "f1": float(eval_out[1])}

        # Print and save results
        print(f" Total time for {total_samples} inputs: {total_time:.4f}")
        print(f" Steady state samples/s: {(steady_state_samples_per_second):.1f}")
        print(f" Batch size: {batch_size}")
        print(f" Evaluation: EM={eval_results['exact_match']:.2f}, F1={eval_results['f1']:.2f}")
        print('*' * len(end_str))
        print()

        exact_match = eval_results['exact_match']
        f1_score = eval_results['f1']

        all_results = {
            "total_time": total_time,
            "total_samples": total_samples,
            "steady_state_samples_per_sec": steady_state_samples_per_second,
            "batch_size": batch_size,
            "evaluation_score": eval_results
        }
        with open(parameters["out_file"], "w") as file:
            file.write(json.dumps(all_results))

        if not parameters["wait_for_user"]:
            break

    return (exact_match, f1_score, steady_state_samples_per_second)

def verify_results(parameters, em, f1, samples_per_second):
    # Check if results are correct
    if parameters["num_encoders"] == 24:
        # Verify em/f1
        assert em > (ground_truth["em"] * 0.98), "Test Failure: low EM"
        assert f1 > (ground_truth["f1"] * 0.98), "Test Failure: low F1"
    if parameters["num_chips"] == 1 and parameters["num_encoders"] == 24:
        # Verify perf
        assert samples_per_second > (ground_truth["1_chip_24_encoder_perf"] * 0.95), "Test Failure: low 1 chip 24 encoder performance"
    if parameters["num_chips"] == 32 and parameters["num_encoders"] == 24:
        # Verify perf
        assert samples_per_second > (ground_truth["32_chip_24_encoder_perf"] * 0.95), "Test Failure: low 32 chip 24 encoder performance"

# Pytest to run bert demo model
@pytest.mark.parametrize("data_type", ["Fp64", "Fp32", "Fp16", "Fp16_b"])
@pytest.mark.parametrize("seq_max_length", [384])
@pytest.mark.parametrize("batch_size", ["batch1", "batch64", "batch128", "batch256"])
@pytest.mark.parametrize("n_best_size", ["n_best_size_1", "n_best_size_5", "n_best_size_10"])
@pytest.mark.parametrize("max_answer_length", ["max_answer_length_10", "max_answer_length_30", "max_answer_length_50"])
@pytest.mark.parametrize("loops", ["loops1", "loops2", "loops5", "loops10"])
@pytest.mark.parametrize("num_chips", ["chip1", "chip2", "chip32", "chip33", "chip64", "chip65"])
@pytest.mark.parametrize("encoders", ["encoder1", "encoder2", "encoder3", "encoder4", "encoder12", "encoder24"])
def test_bert_squad(data_type, seq_max_length, batch_size, n_best_size, max_answer_length, loops, num_chips, encoders):
    parameters = {}

    parameters["hf_model"] = "bert-large-uncased-whole-word-masking-finetuned-squad"
    parameters["data_type"] = data_type
    parameters["seq_max_length"] = seq_max_length
    parameters["batch_size"] = int(batch_size.replace("batch", ""))
    parameters["n_best_size"] = int(n_best_size.replace("n_best_size_", ""))
    parameters["max_answer_length"] = int(max_answer_length.replace("max_answer_length_", ""))
    parameters["loops"] = int(loops.replace("loops", "")) - 1   # Reduce loop count by one since a warmup loop is added by default
    parameters["wait_for_user"] = False
    parameters["quiet"] = False
    parameters["pybuda_latency"] = False
    parameters["num_chips"] = int(num_chips.replace("chip", ""))
    parameters["num_encoders"] = int(encoders.replace("encoder", ""))

    # Run test case
    em, f1, samples_per_second = main(parameters)

    # Verify results
    verify_results(parameters, em, f1, samples_per_second)

if __name__ == "__main__":
    # Setup Arguments
    parser = argparse.ArgumentParser(description="SQuAD Precompute Arguments")
    parser.add_argument(
        "--hf_model",
        help="HuggingFace model.",
        default="bert-large-uncased-whole-word-masking-finetuned-squad"
        )
    parser.add_argument(
        "--data_type",
        help="Data type.",
        choices=["Fp64", "Fp32", "Fp16", "Fp16_b"],
        default="Fp32"
        )
    parser.add_argument(
        "--seq_max_length",
        help="Sequence max length.",
        default=384,
        type=int
        )
    parser.add_argument(
        "--batch_size",
        help="Batch size.",
        default=256, type=int
        )
    parser.add_argument(
        "--n_best_size",
        help="Number of best predictions to return.",
        default=5, type=int
        )
    parser.add_argument(
        "--max_answer_length",
        help="Max length of answer.",
        default=30, type=int
        )
    parser.add_argument(
        "--loops",
        help="Number of loops through the dev dataset.",
        default=5, type=int
        )
    parser.add_argument(
        "--wait_for_user",
        help="Wait for user input and loop benchmark after compilation.",
        action="store_true"
        )
    parser.add_argument(
        "--quiet",
        help="Suppress debug output.",
        action="store_true"
        )
    parser.add_argument(
        "--pybuda_latency",
        help="Show latency including time through pybuda queues (higher than hw latency).",
        action="store_true"
        )
    parser.add_argument(
        "--num_chips",
        help="Run on single chip or galaxy",
        default=1, type=int
        )
    parser.add_argument(
        "--num_encoders",
        help="Number of encoders to specify",
        default=24, type=int
        )

    args = parser.parse_args()
    parameters = vars(args)

    # Run test case
    em, f1, samples_per_second = main(parameters)

    # Verify results
    verify_results(parameters, em, f1, samples_per_second)
