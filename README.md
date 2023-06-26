# dgSPARSE Library

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
[![Latest Release](https://img.shields.io/github/v/release/dgsparse/dgsparse-library)](https://github.com/dgSPARSE/dgSPARSE-Library/releases/)

## Introdution

The dgSPARSE Library (Deep Graph Sparse Library) is a high performance library for sparse kernel acceleration on GPUs based on CUDA.

## File Structure

```
.
├── include:
│   └── dgsparse.h: The header file of the dgSPARSE Library.
├── lib:
│   └── dgsparse.so: The dynamic link file of the dgSPARSE Library.
└── src: Some source codes and references of implementations in the dgSPARSE Library.
    └── ge-spmm: GE-SpMM implementation.
```

## Installation

First, setup the the following environment variables:

```bash
export CUDA_HOME=/usr/local/cuda # your cuda path
export LD_LIBRARY_PATH=$CUDA_HOME/lib64 # your cuda lib path
```

Then, install with pip.

```bash
pip install dgsparse
```

> for developers, you could also install from source code with `python setup.py install`.

## Run Examples

Check your environment.

```bash
export CUDA_HOME=/usr/local/cuda # your cuda path
export LD_LIBRARY_PATH=$CUDA_HOME/lib64
```

Then build dgsparse through `make exp`.
You could run our kernels in the example folder.

## Documentation

Please refer to [dgSPARSE Library Documentation](https://dgsparse.github.io/dgSPARSE-doc/) for more details.
