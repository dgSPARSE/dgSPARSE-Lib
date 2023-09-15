import torch
from torch import nn
import os
from scipy.io import mmread
from dgsparse import spmm_sum, spmm_max, spmm_min, spmm_mean
from dgsparse import SparseTensor
import torch_sparse
import pytest

class SpMMSum:
    def __init__(self, data, in_dim, device, algorithm) -> None:
        # prepare for torch and dgsparse
        self.tcsr = data.tcsr

        self.dcsr = SparseTensor.from_torch_sparse_csr_tensor(
            self.tcsr.clone().detach(), True, requires_grad=True
        )

        self.in_dim = in_dim
        self.device = device
        self.algorithm = algorithm
        self.input_feature = torch.rand(
            (data.num_nodes, in_dim), requires_grad=True, device=device
        )

    def forward_check(self):
        out_check = torch.sparse.mm(self.tcsr, self.input_feature)
        out = spmm_sum(self.dcsr, self.input_feature, self.algorithm)
        assert torch.allclose(out, out_check) == True

    def backward_check(self):
        out_check = torch.sparse.mm(self.tcsr, self.input_feature)
        out_check.sum().backward()
        dX_check = self.input_feature.grad
        dA_check = self.tcsr.grad
        out = spmm_sum(self.dcsr, self.input_feature, self.algorithm)
        out.sum().backward()
        dX = self.input_feature.grad
        dA_nnz = self.dcsr.storage._values.grad

        assert torch.allclose(dX, dX_check) == True
        assert torch.allclose(dA_nnz, dA_check.values()) == True

class SpMMMax:
    def __init__(self, data, in_dim, device, algorithm) -> None:
        # prepare for torch and dgsparse
        self.tcsr = data.tcsr

        self.dcsr = SparseTensor.from_torch_sparse_csr_tensor(
            self.tcsr.clone().detach(), True, requires_grad=True
        )

        self.in_dim = in_dim
        self.device = device
        self.algorithm = algorithm
        self.input_feature = torch.rand(
            (data.num_nodes, in_dim), requires_grad=True, device=device
        )

    def forward_check(self):
        out_check = torch.sparse.mm(self.tcsr.cpu(), self.input_feature.cpu(), "max").to(self.device)
        out = spmm_max(self.dcsr, self.input_feature, self.algorithm)
        assert torch.allclose(out, out_check) == True

    def backward_check(self):
        out_check = torch.sparse.mm(self.tcsr.cpu(), self.input_feature.cpu(), "max").to(self.device)
        out_check.sum().backward()
        dX_check = self.input_feature.grad
        dA_check = self.tcsr.grad
        out = spmm_max(self.dcsr, self.input_feature, self.algorithm)
        out.sum().backward()
        dX = self.input_feature.grad
        dA_nnz = self.dcsr.storage._values.grad

        assert torch.allclose(dX, dX_check) == True
        assert torch.allclose(dA_nnz, dA_check.values()) == True

class SpMMMin:
    def __init__(self, data, in_dim, device, algorithm) -> None:
        # prepare for torch and dgsparse
        self.tcsr = data.tcsr

        self.dcsr = SparseTensor.from_torch_sparse_csr_tensor(
            self.tcsr.clone().detach(), True, requires_grad=True
        )

        self.in_dim = in_dim
        self.device = device
        self.algorithm = algorithm
        self.input_feature = torch.rand(
            (data.num_nodes, in_dim), requires_grad=True, device=device
        )

    def forward_check(self):
        out_check = torch.sparse.mm(self.tcsr.cpu(), self.input_feature.cpu(), "min").to(self.device)
        out = spmm_min(self.dcsr, self.input_feature, self.algorithm)
        assert torch.allclose(out, out_check) == True

    def backward_check(self):
        out_check = torch.sparse.mm(self.tcsr.cpu(), self.input_feature.cpu(), "min").to(self.device)
        out_check.sum().backward()
        dX_check = self.input_feature.grad
        dA_check = self.tcsr.grad
        out = spmm_min(self.dcsr, self.input_feature, self.algorithm)
        out.sum().backward()
        dX = self.input_feature.grad
        dA_nnz = self.dcsr.storage._values.grad

        assert torch.allclose(dX, dX_check) == True
        assert torch.allclose(dA_nnz, dA_check.values()) == True


class SpMMMean:
    def __init__(self, data, in_dim, device, algorithm) -> None:
        # prepare for torch and dgsparse
        self.tcsr = data.tcsr

        self.dcsr = SparseTensor.from_torch_sparse_csr_tensor(
            self.tcsr.clone().detach(), True, requires_grad=True
        )

        self.in_dim = in_dim
        self.device = device
        self.algorithm = algorithm
        self.input_feature = torch.rand(
            (data.num_nodes, in_dim), requires_grad=True, device=device
        )

    def forward_check(self):
        out_check = torch.sparse.mm(self.tcsr.cpu(), self.input_feature.cpu(), "mean").to(self.device)
        out = spmm_mean(self.dcsr, self.input_feature, self.algorithm)
        assert torch.allclose(out, out_check) == True

    def backward_check(self):
        out_check = torch.sparse.mm(self.tcsr.cpu(), self.input_feature.cpu(), "mean").to(self.device)
        out_check.sum().backward()
        dX_check = self.input_feature.grad
        dA_check = self.tcsr.grad
        out = spmm_mean(self.dcsr, self.input_feature, self.algorithm)
        out.sum().backward()
        dX = self.input_feature.grad
        dA_nnz = self.dcsr.storage._values.grad

        assert torch.allclose(dX, dX_check) == True
        assert torch.allclose(dA_nnz, dA_check.values()) == True


from utils import GraphDataset

datasets = ["cora", "citeseer", "pubmed", "ppi"]
features = [32, 64, 128]
@pytest.mark.parametrize('dataset', datasets)
@pytest.mark.parametrize('feat', features)
def test_spmm_sum(dataset, feat):
    data = GraphDataset(dataset, 0)
    gc = SpMMSum(data, feat, 0, 0)
    gc.forward_check()
    gc.backward_check()


datasets = ["cora", "citeseer", "pubmed", "ppi"]
features = [32, 64, 128]
@pytest.mark.parametrize('dataset', datasets)
@pytest.mark.parametrize('feat', features)
def test_spmm_max(dataset, feat):
    data = GraphDataset(dataset, 0)
    gc = SpMMMax(data, feat, 0, 0)
    gc.forward_check()
    gc.backward_check()

datasets = ["cora", "citeseer", "pubmed", "ppi"]
features = [32, 64, 128]
@pytest.mark.parametrize('dataset', datasets)
@pytest.mark.parametrize('feat', features)
def test_spmm_min(dataset, feat):
    data = GraphDataset(dataset, 0)
    gc = SpMMMin(data, feat, 0, 0)
    gc.forward_check()
    gc.backward_check()

datasets = ["cora", "citeseer", "pubmed", "ppi"]
features = [32, 64, 128]
@pytest.mark.parametrize('dataset', datasets)
@pytest.mark.parametrize('feat', features)
def test_spmm_mean(dataset, feat):
    data = GraphDataset(dataset, 0)
    gc = SpMMMean(data, feat, 0, 0)
    gc.forward_check()
    gc.backward_check()

