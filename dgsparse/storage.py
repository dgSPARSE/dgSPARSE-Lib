import warnings
from typing import Optional, List, Tuple

import torch


@torch.jit.script
class Storage(object):
    _row: Optional[torch.Tensor]
    _rowptr: Optional[torch.Tensor]
    _col: Optional[torch.Tensor]
    _values: Optional[torch.Tensor]

    def __init__(
        self,
        row: Optional[torch.Tensor] = None,
        rowptr: Optional[torch.Tensor] = None,
        col: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
    ):

        assert row is not None or rowptr is not None
        assert col is not None
        assert col.dtype == torch.int
        assert col.dim() == 1
        col = col.contiguous()

        M: int = 0
        if rowptr is not None:
            M = rowptr.numel() - 1
        elif row is not None and row.numel() > 0:
            M = int(row.max()) + 1

        N: int = 0
        if col.numel() > 0:
            N = int(col.max()) + 1

        self.sparse_sizes = (M, N)

        if row is not None:
            assert row.dtype == torch.int
            assert row.device == col.device
            assert row.dim() == 1
            assert row.numel() == col.numel()
            row = row.contiguous()

        if rowptr is not None:
            assert rowptr.dtype == torch.int
            assert rowptr.device == col.device
            assert rowptr.dim() == 1
            assert rowptr.numel() - 1 == self.sparse_sizes[0]
            rowptr = rowptr.contiguous()

        if values is not None:
            assert values.device == col.device
            assert values.size(0) == col.size(0)
            values = values.contiguous()

        self._row = row
        self._rowptr = rowptr
        self._col = col
        self._values = values

    @classmethod
    def empty(self):
        row = torch.tensor([], dtype=torch.int)
        col = torch.tensor([], dtype=torch.int)
        return Storage(row=row, rowptr=None, col=col, values=None)
