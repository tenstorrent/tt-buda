# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import yaml

# from easydict import EasyDict as edict


def update_config(config_file):
    with open(config_file) as f:
        config = dict(yaml.load(f, Loader=yaml.FullLoader))
        return config
