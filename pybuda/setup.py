# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from setuptools import find_packages, setup


setup(
    name="pybuda",
    version="0.1",
    description="Tenstorrent Python Buda framework",
    packages=["pybuda"],
    package_dir={"pybuda": "pybuda"},
)
