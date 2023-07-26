import importlib
import os.path as osp

import torch

__version__ = "0.1"

for library in ["_spmm"]:
    cuda_spec = importlib.machinery.PathFinder().find_spec(
        f"{library}_cuda", [osp.dirname(__file__)]
    )
    # cpu_spec = importlib.machinery.PathFinder().find_spec(
    #     f'{library}_cpu', [osp.dirname(__file__)])
    spec = cuda_spec  # or cpu_spec
    if spec is not None:
        torch.ops.load_library(spec.origin)
    else:
        raise ImportError(
            f"Could not find module '{library}_cuda' in " f"{osp.dirname(__file__)}"
        )
import dgsparse._C

cuda_version = dgsparse._C.cuda_version()
if torch.version.cuda is not None and cuda_version != -1:  # pragma: no cover
    if cuda_version < 10000:
        major, minor = int(str(cuda_version)[0]), int(str(cuda_version)[2])
    else:
        major, minor = int(str(cuda_version)[0:2]), int(str(cuda_version)[3])
    t_major, t_minor = [int(x) for x in torch.version.cuda.split(".")]

    if t_major != major:
        raise RuntimeError(
            f"Detected that PyTorch and dgsparse were compiled with "
            f"different CUDA versions. PyTorch has CUDA version "
            f"{t_major}.{t_minor} and dgsparse has CUDA version "
            f"{major}.{minor}. Please reinstall the dgsparse that "
            f"matches your PyTorch install."
        )

from .spmm import spmm_sum, spmm_max, spmm_min, spmm_mean  # noqa
from .storage import Storage
from .tensor import SparseTensor

# from .tensor import SparseTensor

__all__ = ["spmm_sum", "spmm_max", "spmm_min", "spmm_mean", "Storage", "SparseTensor"]
