import torch
from typing import Any, Dict, List, Optional, Tuple, Union

from dgsparse.storage import Storage


@torch.jit.script
class SparseTensor(object):
    storage: Storage

    def __init__(
        self,
        row: Optional[torch.Tensor] = None,
        rowptr: Optional[torch.Tensor] = None,
        col: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        has_value: bool = False,
        is_sorted: bool = False,
    ):
        self.storage = Storage(row=row, rowptr=rowptr, col=col, values=values)
        self.has_value = has_value

    @classmethod
    def from_torch_sparse_csr_tensor(self, mat: torch.Tensor, has_value: bool = True):
        if has_value:
            values = mat.values()
        else:
            values = None
        return SparseTensor(
            row=None,
            rowptr=mat.crow_indices(),
            col=mat.col_indices(),
            values=values,
            has_value=has_value,
            is_sorted=True,
        )
