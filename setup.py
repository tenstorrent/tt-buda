# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import re
import sys
import sysconfig
import platform
import subprocess

__requires__ = ['pip >= 24.0']

from distutils.version import LooseVersion
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

# BudaBackend files to be copied over
bbe_files = {
    "lib": {
        "path": "build/lib" ,
        "files": ["libtt.so", "libdevice.so"],
    },
    "bin": {
        "path": "build/bin" ,
        "files": ["net2pipe", "pipegen2", "blobgen2", "op_model", "dpnra"],
    },
    "device_descriptors": {
        "path": "device",
        "files": [
            "grayskull_120_arch.yaml",
            "grayskull_10x12.yaml",
            "wormhole_8x10.yaml",
            "wormhole_80_arch.yaml",
            "wormhole_b0_8x10.yaml",
            "wormhole_b0_8x10_harvested.yaml",
            "wormhole_80_harvested.yaml",
            "wormhole_b0_80_arch.yaml",
            "wormhole_b0_80_harvested.yaml",
            "wormhole_b0_1x1.yaml",
            "grayskull_10x12.yaml",
            "wormhole_b0_4x6.yaml",
            "blackhole_1x1.yaml",
            "blackhole_8x10.yaml",
            "blackhole_80_arch.yaml",
            "blackhole_10x14.yaml",
            "blackhole_10x14_no_eth.yaml",

        ]
    },
    "params": {
        "path": "perf_lib/op_model/params",
        "files": "*"
    },
    "device_silicon_wormhole_bin": {
        "path": "umd/device/bin/silicon/x86",
        "files": [
            "create-ethernet-map"
        ]
    },
    "misc": {
        "path": "infra",
        "files": [
            "common.mk"
        ]
    },
    "firmware": {
        "path": "src/firmware/riscv",
        "files": "*"
    },
    "firmware_brisc_hex": {
        "path": "build/src/firmware/riscv/targets/brisc/out",
        "files": [
            "brisc.hex",
            "brisc.elf"
        ]
    },
    "firmware_ncrisc_hex": {
        "path": "build/src/firmware/riscv/targets/ncrisc/out",
        "files": [
            "ncrisc.hex",
            "ncrisc.elf"
        ]
    },
    "kernels": {
        "path": "src/ckernels", # TODO clean up, maybe we don't need *everything* here?
        "files": "*" 
    },
    "third_party_grayskull": {
        "path": "third_party/tt_llk_grayskull",
        "files": "*"
    },
    "third_party_wormhole_b0": {
        "path": "third_party/tt_llk_wormhole_b0",
        "files": "*"
    },
    "third_party_blackhole": {
        "path": "third_party/tt_llk_blackhole",
        "files": "*"
    },
    "kernel_gen": {
        "path": "build/src/ckernels/gen/out",
        "files": "*",
    },
    "hlk": {
        "path": "hlks",
        "files": "*",
    },
    "perf_lib": {
        "path": "perf_lib",
        "files": [
            "scratch_api.h",
            "__init__.py",
            "data_movement_perf_sweep.py",
            "fork_join.py",
            "logger_utils.py",
            "op_perf_test.py",
            "ops.py",
            "overlay_decouple.py",
            "perf_analysis.py",
            "perf_analysis_base.py",
            "perf_analyzer_api.py",
            "perf_analyzer_summary.py",
            "perf_comparison.py",
            "perf_graph.py",
            "perf_report.py",
            "perf_sweep.py",
            "perf_test_base.py",
            "perf_to_vcd.py",
            "postprocess_api.py",
            "run_perf_test.py",
            "sweep_params.py",
            "vcdparse.py",
        ]
    },
    # TODO: cleanup, this is deprecated.
    "overlay": {
        "path": "tb/llk_tb/overlay",
        "files": "*" # TODO, clean-up, don't need everything
    },
    # TODO: cleanup, see if this should be on some other section.
    "blobgen2_cpp_overlay": {
        "path": "src/blobgen2",
        "files": [
            "blob_init.hex.static"
        ]
    },
    "versim_lib": { # TODO, remove
        "path": "common_lib",
        "files": "*",
    },
    "sfpi": {
        "path": "third_party/sfpi",
        "files": "*" 
    },
}

if "BACKEND_ARCH_NAME" in os.environ and os.environ["BACKEND_ARCH_NAME"] == "wormhole_b0" or os.environ["BACKEND_ARCH_NAME"] == "blackhole":
    bbe_files["firmware_erisc_hex"] = {
        # "path": "build/src/firmware/riscv/targets/erisc_app/out",
        "path": "erisc_hex",
        "files": [
            "erisc_app.hex",
            # "erisc_app.elf",
            "erisc_app.iram.hex",
            "erisc_app.l1.hex",
            "split_iram_l1"
        ]
    }


pybuda_files = {
    "test" : {
        "path": "pybuda/test",
        "files": [
            "conftest.py",
            "__init__.py",
            "utils.py",
            "test_user.py"
        ]
    }
}

class TTExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])

class MyBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            fullname = self.get_ext_fullname(ext.name)
            filename = self.get_ext_filename(fullname)
            build_lib = self.build_lib
            if not os.path.exists(build_lib):
                continue # editable install?

            # Build using our make flow, and then copy the file over

            # Pass the required variables for building Wormhole or Grayskull
            if "BACKEND_ARCH_NAME" not in os.environ:
                print("Please provide environment variable `BACKEND_ARCH_NAME` to the build process.")
                sys.exit(1)

            additional_env_variables = {
                "BACKEND_ARCH_NAME": os.environ.get("BACKEND_ARCH_NAME"),
            }
            env = os.environ.copy()
            env.update(additional_env_variables)
            nproc = os.cpu_count()
            subprocess.check_call(["make", f"-j{nproc}", "pybuda", r'DEVICE_VERSIM_INSTALL_ROOT=\$$ORIGIN/../..'], env=env)
            subprocess.check_call([f"cd third_party/budabackend && make -j{nproc} netlist_analyzer"], env=env, shell=True)

            src = "build/lib/libpybuda_csrc.so"
            self.copy_file(src, os.path.join(build_lib, filename))

            self._copy_budabackend(build_lib + "/budabackend")
            self._copy_pybuda(build_lib)

    def _copy_pybuda(self, target_path):

        for t, d in pybuda_files.items():
            path = target_path + "/" + d["path"]
            os.makedirs(path, exist_ok=True)

            src_path = d["path"]
            if d["files"] == "*":
                self.copy_tree(src_path, path)
            else:
                for f in d["files"]:
                    self.copy_file(src_path + "/" + f, path + "/" + f)

    def _copy_budabackend(self, target_path):

        src_root = "third_party/budabackend"

        for t, d in bbe_files.items():
            path = target_path + "/" + d["path"]
            os.makedirs(path, exist_ok=True)

            src_path = src_root + "/" + d["path"]
            if d["files"] == "*":
                self.copy_tree(src_path, path)
            else:
                for f in d["files"]:
                    self.copy_file(src_path + "/" + f, path + "/" + f)

with open("README.md", "r") as f:
    long_description = f.read()

# Read the requirements from the core list that is shared between
# dev and test.
with open("python_env/core_requirements.txt", "r") as f:
    requirements = f.read().splitlines()

# Add specific requirements for distribution
# due to how we use the requirements file we can not use include requirements files
with open("python_env/dist_requirements.txt", "r") as f:
    requirements += [r for r in f.read().splitlines() if not r.startswith("-r")]

# pybuda._C
pybuda_c = TTExtension("pybuda._C")

# budabackend
#budabackend = CMakeExtension("budabackend", "pybuda/csrc")

ext_modules = [pybuda_c]

packages = [p for p in find_packages("pybuda") if not p.startswith("test")]

short_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
date = subprocess.check_output(['git', 'show', '-s', '--format=%cd', "--date=format:%y%m%d", 'HEAD']).decode('ascii').strip()

arch_codes = {"wormhole_b0": "wh_b0", "grayskull": "gs", "blackhole": "bh"}
arch_code = arch_codes[os.environ["BACKEND_ARCH_NAME"]]

version = "0.1." + date + "+dev." + arch_code + "." + short_hash

setup(
    name='pybuda',
    version=version,
    author='Tenstorrent',
    url="http://www.tenstorrent.com",
    author_email='info@tenstorrent.com',
    description='AI/ML framework for Tenstorrent devices',
    python_requires='>=3.8',
    packages=packages,
    package_data={"pybuda": ["tti/runtime_param_yamls/*.yaml"]},
    package_dir={"pybuda": "pybuda/pybuda"},
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass=dict(build_ext=MyBuild),
    zip_safe=False,
    install_requires=requirements,
    license="TBD",
    keywords="pybuda machine learning tenstorrent",
    # PyPI
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ]
)
