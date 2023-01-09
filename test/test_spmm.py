import torch
from torch import nn
import os
from scipy.io import mmread
from dgsparse import spmm_sum


class SpMM():
    def __init__(self, path, in_dim, device) -> None:
        sparsecsr = mmread(path).astype('float32').tocsr()
        rowptr = torch.from_numpy(sparsecsr.indptr).to(device).int()
        colind = torch.from_numpy(sparsecsr.indices).to(device).int()
        weight = torch.from_numpy(sparsecsr.data).to(device).float()
        self.csr = torch.sparse_csr_tensor(rowptr, colind, weight, dtype=torch.float)
        nodes = rowptr.size(0) - 1
        self.input_feature = torch.rand((nodes, in_dim)).to(device)

    def calculate(self):
        out_check = torch.sparse.mm(self.csr, self.input_feature)
        out = spmm_sum(self.csr, self.input_feature)
        assert(torch.allclose(out, out_check), True)


def test_spmm():
    gc = SpMM("../example/data/p2p-Gnutella31.mtx", 32, 0)
    # gc.calculate()
