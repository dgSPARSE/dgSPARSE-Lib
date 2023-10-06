import torch
from dgsparse.tensor import SparseTensor
from typing import Tuple


def csr2csc(sparse: SparseTensor) -> Tuple[torch.Tensor]:
    rowptr = sparse.storage._rowptr
    col = sparse.storage._col
    values = sparse.storage._values
    return torch.ops.dgsparse_spmm.csr2csc(rowptr, col, values)
