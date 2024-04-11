# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pwd
import tempfile
import filelock
import shutil
from pathlib import Path

from pybuda.utils import (
    get_pybuda_git_hash,
    resolve_output_build_directory,
    write_buda_envs_configs,
    clear_test_output_build_directory,
    create_test_output_build_directory,
    get_current_pytest,
)
from loguru import logger


def enabled():
    return "PYBUDA_CI_DIR" in os.environ


def capture_tensors():
    return enabled() and os.environ.get("PYBUDA_CI_CAPTURE_TENSORS", "0") != "0"


def get_netlist_dir():
    if not enabled():
        return resolve_output_build_directory()

    base_dir = os.environ.get("PYBUDA_CI_DIR")
    netlist_dir = os.path.join(
        base_dir,
        (
            get_current_pytest()
            .replace(" ", "_")
            .replace("/", "_")
            .replace(".", "_")
            .replace("::", "_")
            .replace("[", "_")
            .replace("]", "")
            .replace("(", "_")
            .replace(")", "_")
            .replace(",", "_")
        ),
    )
    os.makedirs(netlist_dir, exist_ok=True)
    return netlist_dir


def comment_test_info(net) -> str:
    try:
        git_hash = get_pybuda_git_hash()
        net.append_comment(f"git checkout {git_hash}")
    except:
        pass

    current_test = get_current_pytest()
    if current_test:
        net.append_comment(f"pytest {current_test}")
    else:
        import sys
        net.append_comment(f"{' '.join(sys.argv)}")


def create_symlink(target_path: str, symlink_path: str, *, remove_existing: bool = False):
    # Path objects for target and symlink
    target, symlink = Path(target_path), Path(symlink_path)

    # Create a lock file in a standard temporary directory under user's name
    lock_file_path = os.path.join(tempfile.gettempdir(), pwd.getpwuid(os.getuid()).pw_name, f"{symlink.name}.lock")
    lock = filelock.FileLock(lock_file_path)

    with lock:
        try:
            # Remove the existing symlink, file, or directory if it exists
            if symlink.is_symlink() or symlink.is_file():
                symlink.unlink()
            elif symlink.is_dir() and remove_existing:
                shutil.rmtree(symlink)
            elif symlink.is_dir():
                raise ValueError(f"Directory exists at {symlink_path}, cannot create symlink")
            
            symlink.parent.mkdir(parents=True, exist_ok=True) # ensure parent directory exists
            symlink.symlink_to(target)
            logger.info(f'Symlink created from {symlink_path} to {target_path}')
        
        except Exception as e:
            logger.warning(f'Failed to create symlink: {e}')

def initialize_output_build_directory(backend_output_directory: str):
    clear_test_output_build_directory(backend_output_directory)
    create_test_output_build_directory(backend_output_directory)

    logger.info(
        f"Pybuda output build directory for compiled artifacts: {backend_output_directory}"
    )
    if not enabled():
        create_symlink(
            os.path.abspath(backend_output_directory),
            os.path.join(os.getcwd(), "tt_build/test_out"),
            remove_existing=True,
        )


def write_netlist(net, netlist_filename: str) -> str:
    with open(netlist_filename, "w") as f:
        f.write(net.dump_to_yaml())


def write_netlist_and_buda_envs_config(
    net, graph_name: str, default_directory: str
) -> str:
    comment_test_info(net)
    netlist_dir = default_directory if not enabled() else get_netlist_dir()
    netlist_name = graph_name + "_netlist.yaml"
    netlist_filename = os.path.join(netlist_dir, netlist_name)

    write_buda_envs_configs(netlist_dir)
    write_netlist(net, netlist_filename)

    if not enabled():
        create_symlink(netlist_filename, os.path.join(os.getcwd(), netlist_name))
    return netlist_filename
