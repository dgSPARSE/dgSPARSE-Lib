import torch
from torch import nn
import os
from scipy.io import mmread
from dgsparse import spmm_sum, spmm_max, spmm_min, spmm_mean
from dgsparse import SparseTensor
import torch_sparse


class SpMMSum:
    def __init__(self, path, in_dim, device, algorithm) -> None:
        sparsecsr = mmread(path).astype("float32").tocsr()
        rowptr = torch.from_numpy(sparsecsr.indptr).to(device).int()
        colind = torch.from_numpy(sparsecsr.indices).to(device).int()
        weight = torch.from_numpy(sparsecsr.data).to(device).float()
        self.rowptr = rowptr
        self.colind = colind
        sparsecoo = sparsecsr.tocoo()

        # prepare for pytorch_sparse
        row = torch.from_numpy(sparsecoo.row).type(torch.int64)
        col = torch.from_numpy(sparsecoo.col).type(torch.int64)
        print(rowptr)
        print(colind)
        # print(weight)
        index = torch.cat((row.unsqueeze(0), col.unsqueeze(0)), 0)
        self.index = index.to(device)
        self.m = sparsecoo.shape[0]
        self.n = sparsecoo.shape[1]
        self.weight = weight.requires_grad_()

        # prepare for torch and dgsparse
        self.tcsr = torch.sparse_csr_tensor(
            rowptr, colind, weight, dtype=torch.float, requires_grad=True
        )

        self.dcsr = SparseTensor.from_torch_sparse_csr_tensor(
            self.tcsr.clone().detach(), True, requires_grad=True
        )
        nodes = rowptr.size(0) - 1
        self.nodes = nodes
        self.in_dim = in_dim
        self.device = device
        self.algorithm = algorithm
        self.input_feature = torch.rand(
            (nodes, in_dim), requires_grad=True, device=device
        )
        # don't use copy to device, or may can not implement backward_check

    def forward_check(self):
        out_check = torch.sparse.mm(self.tcsr, self.input_feature)
        out = spmm_sum(self.dcsr, self.input_feature, self.algorithm)
        # print(out-out_check)
        print(out)
        print(out_check)
        assert torch.allclose(out, out_check) == True

    def backward_check(self):
        out_check = torch.sparse.mm(self.tcsr, self.input_feature)
        out_check.sum().backward()
        dX_check = self.input_feature.grad
        dA_check = self.tcsr.grad
        out = spmm_sum(self.dcsr, self.input_feature, self.algorithm)
        out.retain_grad()
        out.sum().backward()
        dX = self.input_feature.grad
        dA_nnz = self.dcsr.storage._values.grad

        pyg_check = torch_sparse.spmm(
            self.index, self.weight, self.m, self.n, self.input_feature
        )
        pyg_check.sum().backward()

        assert torch.allclose(dX, dX_check) == True
        assert torch.allclose(dA_nnz, dA_check.values()) == True


class SpMMMax:
    def __init__(self, path, in_dim, device, algorithm) -> None:
        sparsecsr = mmread(path).astype("float32").tocsr()
        rowptr = torch.from_numpy(sparsecsr.indptr).to(device).int()
        colind = torch.from_numpy(sparsecsr.indices).to(device).int()
        weight = torch.from_numpy(sparsecsr.data).to(device).float()
        self.rowptr = rowptr
        self.colind = colind
        sparsecoo = sparsecsr.tocoo()

        # prepare for pytorch_sparse
        row = torch.from_numpy(sparsecoo.row).type(torch.int64)
        col = torch.from_numpy(sparsecoo.col).type(torch.int64)
        print(rowptr)
        print(colind)
        # print(weight)
        index = torch.cat((row.unsqueeze(0), col.unsqueeze(0)), 0)
        self.index = index.to(device)
        self.m = sparsecoo.shape[0]
        self.n = sparsecoo.shape[1]
        self.weight = weight

        # prepare for torch and dgsparse
        self.tcsr = torch.sparse_csr_tensor(
            rowptr, colind, weight, dtype=torch.float, requires_grad=True
        )
        self.dcsr = SparseTensor.from_torch_sparse_csr_tensor(
            self.tcsr.detach(), True, requires_grad=True
        )
        nodes = rowptr.size(0) - 1
        self.nodes = nodes
        self.in_dim = in_dim
        self.device = device
        self.algorithm = algorithm
        self.input_feature = torch.rand(
            (nodes, in_dim), requires_grad=True, device=device
        )
        # don't use copy to device, or may can not implement backward_check

    def forward_check(self):
        out_check = torch.sparse.mm(
            self.tcsr.cpu(), self.input_feature.cpu(), "max"
        ).cuda()
        out = spmm_max(self.dcsr, self.input_feature, self.algorithm)
        print(out - out_check)
        assert torch.allclose(out, out_check) == True

    def backward_check(self):
        out_check = torch.sparse.mm(
            self.tcsr.cpu(), self.input_feature.cpu(), "max"
        ).cuda()
        out_check.sum().backward()
        dX_check = self.input_feature.grad
        dA_check = self.tcsr.grad
        out = spmm_max(self.dcsr, self.input_feature, self.algorithm)
        out.sum().backward()
        dX = self.input_feature.grad
        dA_nnz = self.dcsr.storage._values.grad
        dA = torch.sparse_csr_tensor(
            self.rowptr, self.colind, dA_nnz, dtype=torch.float
        )
        print(dA)
        print(dA_check)
        print(dA.values() - dA_check.values())
        assert torch.allclose(dX, dX_check) == True
        assert torch.allclose(dA.values(), dA_check.values()) == True


class SpMMMin:
    def __init__(self, path, in_dim, device, algorithm) -> None:
        sparsecsr = mmread(path).astype("float32").tocsr()
        rowptr = torch.from_numpy(sparsecsr.indptr).to(device).int()
        colind = torch.from_numpy(sparsecsr.indices).to(device).int()
        weight = torch.from_numpy(sparsecsr.data).to(device).float()
        self.rowptr = rowptr
        self.colind = colind
        sparsecoo = sparsecsr.tocoo()

        # prepare for pytorch_sparse
        row = torch.from_numpy(sparsecoo.row).type(torch.int64)
        col = torch.from_numpy(sparsecoo.col).type(torch.int64)
        print(rowptr)
        print(colind)
        # print(weight)
        index = torch.cat((row.unsqueeze(0), col.unsqueeze(0)), 0)
        self.index = index.to(device)
        self.m = sparsecoo.shape[0]
        self.n = sparsecoo.shape[1]
        self.weight = weight

        # prepare for torch and dgsparse
        self.tcsr = torch.sparse_csr_tensor(
            rowptr, colind, weight, dtype=torch.float, requires_grad=True
        )
        self.dcsr = SparseTensor.from_torch_sparse_csr_tensor(
            self.tcsr.detach(), True, requires_grad=True
        )
        nodes = rowptr.size(0) - 1
        self.nodes = nodes
        self.in_dim = in_dim
        self.device = device
        self.algorithm = algorithm
        self.input_feature = torch.rand(
            (nodes, in_dim), requires_grad=True, device=device
        )
        # don't use copy to device, or may can not implement backward_check

    def forward_check(self):
        out_check = torch.sparse.mm(
            self.tcsr.cpu(), self.input_feature.cpu(), "min"
        ).cuda()
        out = spmm_min(self.dcsr, self.input_feature, self.algorithm)
        print(out - out_check)
        assert torch.allclose(out, out_check) == True

    def backward_check(self):
        out_check = torch.sparse.mm(
            self.tcsr.cpu(), self.input_feature.cpu(), "min"
        ).cuda()
        out_check.sum().backward()
        dX_check = self.input_feature.grad
        dA_check = self.tcsr.grad
        out = spmm_min(self.dcsr, self.input_feature, self.algorithm)
        out.sum().backward()
        dX = self.input_feature.grad
        dA_nnz = self.dcsr.storage._values.grad
        dA = torch.sparse_csr_tensor(
            self.rowptr, self.colind, dA_nnz, dtype=torch.float
        )
        assert torch.allclose(dX, dX_check) == True
        assert torch.allclose(dA.values(), dA_check.values()) == True
    
class SpMMMean:
    def __init__(self, path, in_dim, device, algorithm) -> None:
        sparsecsr = mmread(path).astype("float32").tocsr()
        rowptr = torch.from_numpy(sparsecsr.indptr).to(device).int()
        colind = torch.from_numpy(sparsecsr.indices).to(device).int()
        weight = torch.from_numpy(sparsecsr.data).to(device).float()
        self.rowptr = rowptr
        self.colind = colind
        sparsecoo = sparsecsr.tocoo()

        # prepare for pytorch_sparse
        row = torch.from_numpy(sparsecoo.row).type(torch.int64)
        col = torch.from_numpy(sparsecoo.col).type(torch.int64)
        print(rowptr)
        print(colind)
        # print(weight)
        index = torch.cat((row.unsqueeze(0), col.unsqueeze(0)), 0)
        self.index = index.to(device)
        self.m = sparsecoo.shape[0]
        self.n = sparsecoo.shape[1]
        self.weight = weight

        # prepare for torch and dgsparse
        self.tcsr = torch.sparse_csr_tensor(
            rowptr, colind, weight, dtype=torch.float, requires_grad=True
        )
        self.dcsr = SparseTensor.from_torch_sparse_csr_tensor(
            self.tcsr.detach(), True, requires_grad=True
        )
        nodes = rowptr.size(0) - 1
        self.nodes = nodes
        self.in_dim = in_dim
        self.device = device
        self.algorithm = algorithm
        self.input_feature = torch.rand(
            (nodes, in_dim), requires_grad=True, device=device
        )
        # don't use copy to device, or may can not implement backward_check

    def forward_check(self):
        out_check = torch.sparse.mm(
            self.tcsr.cpu(), self.input_feature.cpu(), "mean"
        ).cuda()
        out = spmm_mean(self.dcsr, self.input_feature, self.algorithm)
        print(out - out_check)
        assert torch.allclose(out, out_check) == True

    def backward_check(self):
        out_check = torch.sparse.mm(
            self.tcsr.cpu(), self.input_feature.cpu(), "mean"
        ).cuda()
        out_check.sum().backward()
        dX_check = self.input_feature.grad
        dA_check = self.tcsr.grad
        out = spmm_mean(self.dcsr, self.input_feature, self.algorithm)
        out.sum().backward()
        dX = self.input_feature.grad
        dA_nnz = self.dcsr.storage._values.grad
        dA = torch.sparse_csr_tensor(
            self.rowptr, self.colind, dA_nnz, dtype=torch.float
        )
        print(dA)
        print(dA_check)
        # print(dX)
        # print(dX_check)
        assert torch.allclose(dX, dX_check) == True
        assert torch.allclose(dA.values(), dA_check.values()) == True


def test_spmm():
    iteration = 10
    for i in range(iteration):
        gc = SpMMMean("../example/data/p2p-Gnutella31.mtx", 32, 0, 0)
        gc.forward_check()
        print(f"{i} Forward Pass")
        gc.backward_check()
        print(f"{i} Backward Pass")

test_spmm()