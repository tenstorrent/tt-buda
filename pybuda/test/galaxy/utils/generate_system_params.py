# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import argparse
import pybuda
from pybuda.config import _get_global_compiler_config
from pybuda.ttdevice import get_device_config
from pybuda._C.backend_api import BackendDevice, BackendType


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate and store backend db to yaml.')
    parser.add_argument("--save-params-file", help="Path to dump backend runtime params yaml.", default="runtime_params.yaml")
    args = parser.parse_args()

    compiler_cfg = _get_global_compiler_config()
    pybuda.set_configuration_options(
        backend_runtime_params_path = args.save_params_file,
        store_backend_db_to_yaml = True,
    )

    device_cfg = get_device_config(
        BackendDevice.Wormhole_B0,
        [0], # chip ids
        compiler_cfg.backend_cluster_descriptor_path,
        compiler_cfg.backend_runtime_params_path,
        compiler_cfg.store_backend_db_to_yaml,
        backend_type = BackendType.Silicon,
      )

