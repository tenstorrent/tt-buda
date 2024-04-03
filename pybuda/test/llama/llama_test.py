# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
'''
Llama-7B-odkv test
'''
import os
import pytest
import generate_eval
import tt_eval


@pytest.mark.parametrize("device", ["silicon"])
@pytest.mark.parametrize("arch", ["greyskull", "wormhole_b0"])
@pytest.mark.parametrize("chips_to_use", ["chip1", "chip2", "chip32"])
def test_llama(device, arch, chips_to_use):
    '''
    see large-lm/investigations/Llama-7B-odkv/README.md
    '''
    data_folder = 'pybuda/test/llama/eval_data/pt_gt_128'
    if chips_to_use == "chip1":
        num_chips = 1
    elif chips_to_use == "chip2":
        num_chips = 2
    elif chips_to_use == "chip32":
        num_chips = 32

    # Step 1. Generate PyTorch ground truth
    # python generate_eval.py --input eval_data/episode_iv.txt --output eval_data/pt_gt_128 --context-length 128 --num-samples 100
    args_for_generate = {
            'model': 'decapoda-research/llama-7b-hf',
            'input': 'pybuda/test/llama/eval_data/episode_iv.txt',
            'output': data_folder,
            'context_length': 128,
            'num_samples': 100,
            'device': 'cpu',
            'num_layers': 32
    }
    generate_eval.generate_eval(args_for_generate)

    # Step 2. Run TT model
    # python tt_eval.py -d silicon --arch wormhole_b0 --num-chips 1 --context-length 128 --input eval_data/pt_gt_128/ --opt-level 1 --amp-config amp_configs/w6.json
    args_for_eval = {
            'model': 'decapoda-research/llama-7b-hf',
            'device': device,
            'arch': arch,
            'num_chips': num_chips,
            'fuse': False,
            'perf': None,
            'log_level': 'ERROR',
            'context_length': 128,
            'input': data_folder,
            'num_layers': 32,
            'opt_level': 1,
            'verify': False,
            'amp_config': 'pybuda/test/llama/amp_configs/w6.json',
            'input_count': None,
            'nlp_target_cycles': -1
    }
    tt_eval.eval(args_for_eval)


if __name__ == '__main__':
    test_llama('silicon', 'wormhole_b0', 'chip1')

