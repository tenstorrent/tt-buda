# TT-Buda

## Introduction

The TT-Buda software stack can compile AI/ML models from several different frameworks such as PyTorch and Tensorflow, and execute them in many different ways on Tenstorrent hardware.

**Note on terminology:**

TT-Buda is the official Tenstorrent AI/ML compiler stack and PyBuda is the Python interface for TT-Buda. PyBuda allows users to access and utilize TT-Buda's features directly from Python. This includes directly importing model architectures and weights from PyTorch, TensorFlow, ONNX, and TFLite.

## Model Demos

Model demos are now part of a separate repo:

https://github.com/tenstorrent/tt-buda-demos

## Docs

See: [Docs](https://docs.tenstorrent.com/tenstorrent/v/tt-buda)

## Build

https://docs.tenstorrent.com/tenstorrent/v/tt-buda/installation

## Env setup

Set `LD_LIBRARY_PATH` to the location of `third_party/budabackend/build/lib` - preferrably the absolute path to allow scripts to find them from anywhere.

## Silicon

See README.silicon.md for details on how to run on silicon.

