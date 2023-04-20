import torch
from torch import nn
import os
from scipy.io import mmread
from dgsparse import spmm_sum
from dgsparse import SparseTensor


class Csr2Csc:
    def __init__(self, path, device):
        sparsecsr = mmread(path).astype("float32").tocsr()
        self.device = device
        self.sparsecsr = sparsecsr
        self.rowptr = torch.from_numpy(sparsecsr.indptr).to(device).int()
        self.colind = torch.from_numpy(sparsecsr.indices).to(device).int()
        self.weight = torch.from_numpy(sparsecsr.data).to(device).float()

        nodes = self.rowptr.size(0) - 1
        in_dim = 32
        self.input_feature = torch.rand(
            (nodes, in_dim), requires_grad=True, device=device
        )
        shape = sparsecsr.shape
        self.rows = shape[0]
        self.cols = shape[1]

    def check(self):
        colptr, row, weight_transpose = torch.ops.dgsparse.csr2csc(
            self.rows, self.cols, self.rowptr, self.colind, self.weight
        )
        tran_s = self.sparsecsr.tocsc()
        colptr_check = torch.from_numpy(tran_s.indptr).to(self.device).int()
        row_check = torch.from_numpy(tran_s.indices).to(self.device).int()
        data_check = torch.from_numpy(tran_s.data).to(self.device).float()

        assert torch.allclose(colptr, colptr_check) == True
        assert torch.allclose(row, row_check) == True
        assert torch.allclose(weight_transpose, data_check) == True


def test_csr2csc():
    # cuda:0
    # gc = Csr2Csc("../example/data/p2p-Gnutella31.mtx", 0)
    # gc.check()
    # cpu check
    gc = Csr2Csc("../example/data/p2p-Gnutella31.mtx", "cpu")
    gc.check()
