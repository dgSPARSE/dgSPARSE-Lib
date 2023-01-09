import torch
from dgsparse.tensor import SparseTensor

# torch.ops.load_library("_spmm_cuda.so")

# torch.ops.dgsparse.SpMM

def spmm_sum(sparse: SparseTensor, dense: torch.Tensor) -> torch.Tensor:
    has_value = sparse.has_value
    rowptr = sparse.rowptr
    col = sparse.col
    values = sparse.values
    return torch.ops.dgsparse.spmm_sum(rowptr, col, values, dense, has_value)