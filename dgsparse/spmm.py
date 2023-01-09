import torch

# torch.ops.load_library("_spmm_cuda.so")

# torch.ops.dgsparse.SpMM

def spmm_sum(sparse: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
    return torch.ops.dgsparse.spmm_sum(sparse.crow_indices(), sparse.col_indices(), sparse.values(), dense, True)