# TT-BUDA Installation

## Overview

The TT-BUDA software stack can compile models from several different frameworks and execute them in many different ways on Tenstorrent hardware.

This user guide is intended to help you setup your system with the appropriate device drivers, firmware, system dependencies, and compiler / runtime software.

**Note on terminology:**

While TT-BUDA is the official Tenstorrent AI/ML compiler stack, PyBUDA is the Python interface for TT-BUDA. TT-BUDA is the core technology; however, PyBUDA allows users to access and utilize TT-BUDA's features directly from Python. This includes directly importing model architectures and weights from PyTorch, TensorFlow, ONNX, and TFLite.

### OS Compatibility

Presently, Tenstorrent software is only supported on the **Ubuntu 20.04 LTS (Focal Fossa)** operating system.

## Release Versions

To access the PyBUDA software and associated files, please navigate to the [Releases](https://github.com/tenstorrent/tt-buda/releases) section of this repository.

Once you have identified the release version you would like to install, you can download the individual files by clicking on their name.

## Table Of Contents

1. [Installation Instructions](#installation-instructions)
   1. [Setup HugePages](#setup-hugepages)
   2. [PCI Driver Installation](#pci-driver-installation)
   3. [Device Firmware Update](#device-firmware-update)
   4. [Backend Compiler Dependencies](#backend-compiler-dependencies)
   5. [TT-SMI](#tt-smi)
   6. [Tenstorrent Software Package](#tenstorrent-software-package)
   7. [Python Environment Installation](#python-environment-installation)
   8. [Docker Container Installation](#docker-container-installation)
   9. [Smoke Test](#smoke-test)

## Installation Instructions

PyBUDA can be installed using two methods: Docker or Python virtualenv.

If you would like to run PyBUDA in a Docker container, then follow the instructions for [PCI Driver Installation](#pci-driver-installation) and [Device Firmware Update](#device-firmware-update) and followed by [Docker Container Installation](#docker-container-installation).

If you would like to run PyBUDA in a Python virtualenv, then follow the instructions for the [Setup HugePages](#setup-hugepages), [PCI Driver Installation](#pci-driver-installation), [Device Firmware Update](#device-firmware-update), and [Backend Compiler Dependencies](#backend-compiler-dependencies), followed by the [Tenstorrent Software Package](#tenstorrent-software-package).

### Setup HugePages

```bash
NUM_DEVICES=$(lspci -d 1e52: | wc -l)
sudo sed -i "s/^GRUB_CMDLINE_LINUX_DEFAULT=.*$/GRUB_CMDLINE_LINUX_DEFAULT=\"hugepagesz=1G hugepages=${NUM_DEVICES} nr_hugepages=${NUM_DEVICES} iommu=pt\"/g" /etc/default/grub
sudo update-grub
sudo sed -i "/\s\/dev\/hugepages-1G\s/d" /etc/fstab; echo "hugetlbfs /dev/hugepages-1G hugetlbfs pagesize=1G,rw,mode=777 0 0" | sudo tee -a /etc/fstab
sudo reboot
```

### PCI Driver Installation

Please navigate to [tt-kmd](https://github.com/tenstorrent/tt-kmd).

### Device Firmware Update

Please navigate to [tt-flash](https://github.com/tenstorrent/tt-flash) and [tt-firmware-gs](https://github.com/tenstorrent/tt-firmware-gs).

### Backend Compiler Dependencies

Instructions to install the Tenstorrent backend compiler dependencies on a fresh install of Ubuntu Server 20.04.

You may need to append each `apt-get` command with `sudo` if you do not have root permissions.

```bash
apt-get update -y && apt-get upgrade -y --no-install-recommends
apt-get install -y software-properties-common
apt-get install -y python3.8-venv libboost-all-dev libgoogle-glog-dev libgl1-mesa-glx ruby
apt-get install -y build-essential clang-6.0 libhdf5-serial-dev libzmq3-dev
```

### TT-SMI

Please navigate to [tt-smi](https://github.com/tenstorrent/tt-smi).

### Tenstorrent Software Package

The directory contains instructions to install the PyBUDA and TVM software stack. This package consists of the following files:

```bash
pybuda-<version>.whl   <- Latest PyBUDA Release
tvm-<version>.whl      <- Latest TVM Release
```

### Python Environment Installation

It is strongly recommended to use virtual environments for each project utilizing PyBUDA and
Python dependencies. Creating a new virtual environment with PyBUDA and libraries is very easy.

#### Step 1. Navigate to [Releases](https://github.com/tenstorrent/tt-buda/releases)

#### Step 2. Under the latest release, download the `pybuda` and `tvm` wheel files

#### Step 3. Create your Python environment in desired directory

```bash
python3 -m venv env
```

#### Step 4. Activate environment

```bash
source env/bin/activate
```

#### Step 5. Pip install PyBuda and TVM

```bash
pip install pybuda-<version>.whl tvm-<version>.whl
```

### Docker Container Installation

Alternatively, PyBUDA and its dependencies are provided as Docker images which can run in separate containers.

#### Step 1. Setup a personal access token (classic)

Create a personal access token from: [GitHub Tokens](https://github.com/settings/tokens).
Give the token the permissions to read packages from the container registry `read:packages`.

#### Step 2. Login to Docker Registry

```bash
GITHUB_TOKEN=<your token>
echo $GITHUB_TOKEN | sudo docker login ghcr.io -u <your github username> --password-stdin
```

#### Step 3. Pull the image

```bash
sudo docker pull ghcr.io/tenstorrent/tt-buda/<TAG>
```

#### Step 4. Run the container

```bash
sudo docker run --rm -ti --shm-size=4g --device /dev/tenstorrent -v /dev/hugepages-1G:/dev/hugepages-1G -v `pwd`/:/home/ ghcr.io/tenstorrent/tt-buda/<TAG> bash
```

#### Step 5. Change root directory

```bash
cd home/
```

### Smoke Test

With your Python environment with PyBUDA install activated, run the following Python script:

```python
import pybuda
import torch


# Sample PyTorch module
class PyTorchTestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights1 = torch.nn.Parameter(torch.rand(32, 32), requires_grad=True)
        self.weights2 = torch.nn.Parameter(torch.rand(32, 32), requires_grad=True)

    def forward(self, act1, act2):
        m1 = torch.matmul(act1, self.weights1)
        m2 = torch.matmul(act2, self.weights2)
        return m1 + m2, m1


def test_module_direct_pytorch():
    input1 = torch.rand(4, 32, 32)
    input2 = torch.rand(4, 32, 32)

    # Run single inference pass on a PyTorch module, using a wrapper to convert to PyBUDA first
    output = pybuda.PyTorchModule("direct_pt", PyTorchTestModule()).run(input1, input2)
    print(output)


if __name__ == "__main__":
    test_module_direct_pytorch()
```
