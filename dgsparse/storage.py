import warnings
from typing import Optional, List, Tuple

import torch


@torch.jit.script
class Storage(object):
    _row: Optional[torch.Tensor]
    _rowptr: Optional[torch.Tensor]
    _col: Optional[torch.Tensor]
    _values: Optional[torch.Tensor]
    _colptr: torch.Tensor
    _csr2csc: torch.Tensor
    _csc2csr: torch.Tensor

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
        self.nnz = col.size(0)

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
            assert values.size(0) == self.nnz
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

    # def colptr(self) -> torch.Tensor:
    #     colptr = self._colptr
    #     if colptr is not None:
    #         return colptr
    #     rows, cols = self.sparse_sizes
    #     device = self._col.device
    #     idx = torch.range(0, 100, device=device)
    #     colptr, row, csr2csc = torch.ops.dgsparse.csr2csc(rows, cols, self._rowptr, self._col, idx)
    #     if self._row == None:
    #         self._row = row
    #     if self._csr2csc == None:
    #         self._csr2csc = csr2csc
    #     self._colptr = colptr

    # def csr2csc(self):
    #     if self._csr2csc is not None:
    #         return self._csr2csc
    #     rows, cols = self.sparse_sizes
    #     device = self._col.device
    #     idx = torch.range(0, 100, device=device)
    #     colptr, row, csr2csc = torch.ops.dgsparse.csr2csc(rows, cols, self._rowptr, self._col, idx)
    #     if self._row == None:
    #         self._row = row
    #     if self._colptr == None:
    #         self._colptr = colptr
    #     self._csr2csc = csr2csc
    #     return csr2csc
