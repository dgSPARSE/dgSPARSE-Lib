import warnings
from typing import Optional, List, Tuple
from torch_scatter import scatter_add

import torch


class Storage(object):
    _row: Optional[torch.Tensor]
    _rowptr: Optional[torch.Tensor]
    _col: Optional[torch.Tensor]
    _values: Optional[torch.Tensor]
    _colptr: torch.Tensor
    _csr2csc: torch.Tensor
    _csc2csr: torch.Tensor
    _colcount: Optional[torch.Tensor]

    def __init__(
        self,
        row: Optional[torch.Tensor] = None,
        rowptr: Optional[torch.Tensor] = None,
        col: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        colptr: Optional[torch.Tensor] = None,
        csr2csc: Optional[torch.Tensor] = None,
        csc2csr: Optional[torch.Tensor] = None,
        colcount: Optional[torch.Tensor] = None,
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
        else:
            values = torch.ones((self.nnz), dtype=torch.float, device=col.device)
            values = values.contiguous()

        if colptr is not None:
            assert colptr.dtype == torch.long
            assert colptr.device == col.device
            assert colptr.dim() == 1
            assert colptr.numel() - 1 == self.sparse_sizes[1]
            colptr = colptr.contiguous()

        if csr2csc is not None:
            assert csr2csc.dtype == torch.long
            assert csr2csc.device == col.device
            assert csr2csc.dim() == 1
            assert csr2csc.numel() == col.size(0)
            csr2csc = csr2csc.contiguous()

        if colcount is not None:
            assert colcount.dtype == torch.long
            assert colcount.device == col.device
            assert colcount.dim() == 1
            assert colcount.numel() == self.sparse_sizes[1]
            colcount = colcount.contiguous()

        self._row = row
        self._rowptr = rowptr
        self._col = col
        self._values = values
        self._colptr = colptr
        self._csr2csc = csr2csc
        self._colcount = colcount

    @classmethod
    def empty(self):
        row = torch.tensor([], dtype=torch.int)
        col = torch.tensor([], dtype=torch.int)
        return Storage(row=row, rowptr=None, col=col, values=None, colptr=None, csc2csr=None, csr2csc=None, colcount=None)
    
    # def row(self) -> torch.Tensor:
    #     row = self._row
    #     if row is not None:
    #         return row

    #     rowptr = self._rowptr
    #     if rowptr is not None:
    #         print(rowptr)
    #         print("将要执行ptr2ind")
    #         row = torch.ops.dgsparse_convert.ptr2ind(rowptr, self._col.numel())
    #         print(rowptr)
    #         self._row = row
    #         return row

    #     raise ValueError
    
    # def rowptr(self) -> torch.Tensor:
    #     rowptr = self._rowptr
    #     if rowptr is not None:
    #         return rowptr

    #     row = self._row
    #     if row is not None:
    #         rowptr = torch.ops.dgsparse_convert.ind2ptr(row, self.sparse_sizes[0])
    #         self._rowptr = rowptr
    #         return rowptr

    #     raise ValueError
    
    # def colptr(self) -> torch.Tensor:
    #     colptr = self._colptr
    #     if colptr is not None:
    #         return colptr

    #     csr2csc = self._csr2csc
    #     if csr2csc is not None:
    #         colptr = torch.ops.dgsparse_convert.ind2ptr(self._col[csr2csc],
    #                                                 self.sparse_sizes[1])
    #     else:
    #         colptr = self._col.new_zeros(self.sparse_sizes[1] + 1)
    #         torch.cumsum(self.colcount(), dim=0, out=colptr[1:])
    #     self._colptr = colptr
    #     return colptr
    
    # def csr2csc(self) -> torch.Tensor:
    #     csr2csc = self._csr2csc
    #     if csr2csc is not None:
    #         return csr2csc

    #     idx = self.sparse_sizes[0] * self._col + self.row()
    #     csr2csc = idx.argsort()
    #     self._csr2csc = csr2csc
    #     return csr2csc
    
    # def colcount(self) -> torch.Tensor:
    #     colcount = self._colcount
    #     if colcount is not None:
    #         return colcount

    #     colptr = self._colptr
    #     if colptr is not None:
    #         colcount = colptr[1:] - colptr[:-1]
    #     else:
    #         colcount = scatter_add(torch.ones_like(self._col), self._col,
    #                                dim_size=self.sparse_sizes[1])
    #     self._colcount = colcount
    #     return colcount
    
    # def col(self) -> torch.Tensor:
    #     return self._col
    
    # def value(self) -> Optional[torch.Tensor]:
    #     return self._values

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

    def csr2csc(self):
        if self._csr2csc is not None:
            return self._csr2csc
        rows, cols = self.sparse_sizes
        device = self._col.device
        # idx = torch.range(0, 100, device=device)
        idx = self._values
        colptr, row, csr2csc = torch.ops.dgsparse_spmm.csr2csc(rows, cols, self._rowptr, self._col, idx)
        if self._row is None:
            self._row = row
        if self._colptr is None:
            self._colptr = colptr
        self._csr2csc = csr2csc
        return csr2csc
