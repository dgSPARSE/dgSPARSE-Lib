import torch
from dgsparse import spmm_sum, spmm_max, spmm_min, spmm_mean
from dgsparse import SparseTensor
# import pytest
# import torch_sparse
from torch_sparse import matmul
import time
# import tqdm
import dgl.sparse as dglsp
from utils import GraphDataset


class SpMMSum:
    def __init__(self, data, device, algorithm) -> None:
        # prepare for torch and dgsparse
        self.tcsr = data.tcsr
        self.dcsr = SparseTensor.from_torch_sparse_csr_tensor(
            self.tcsr.clone().detach(), True, requires_grad=True)
        self.adj_t = data.adj_t
        # self.dgl_A = data.dgl_A
        self.dgl_graph = data.dgl_graph

        self.device = device
        self.algorithm = algorithm
        self.input_feature = data.features
        # self.input_feature =
        # torch.randn((self.dcsr.storage.sparse_sizes[1], 1)).to(device)
        # print(self.input_feature.size())

    def forward_check(self):
        # warm up
        for _ in range(10):
            matmul(self.adj_t, self.input_feature, reduce='sum')
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            matmul(self.adj_t, self.input_feature, reduce='sum')
            torch.cuda.synchronize()
        end = time.time()
        torch_sparse_time = end - start

        # warm up
        for _ in range(10):
            dglsp.spmm(self.dgl_A, self.input_feature)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            dglsp.spmm(self.dgl_A, self.input_feature)
            torch.cuda.synchronize()
        end = time.time()
        dgl_time = end - start

        # warm up
        for _ in range(10):
            spmm_sum(self.dcsr, self.input_feature, self.algorithm)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            spmm_sum(self.dcsr, self.input_feature, self.algorithm)
            torch.cuda.synchronize()
        end = time.time()
        dgsparse_time = end - start

        return torch_sparse_time, dgl_time, dgsparse_time

    def backward_check(self):
        pass
        # #warm up
        # for i in range(10):
        #     out_check = matmul(self.adj_t, self.input_feature, reduce="sum")

        # torch.cuda.synchronize()
        # start = time.time()
        # for i in range(100):
        #     out_check = torch_sparse.spmm_sum(self.adj_t, self.input_feature)
        #     print(out_check.sum())
        #     out_check.sum().backward()
        # torch.cuda.synchronize()
        # end = time.time()
        # torch_sparse_time = end - start

        # #warm up
        # for i in range(10):
        #     out = spmm_sum(self.dcsr, self.input_feature, self.algorithm)

        # torch.cuda.synchronize()
        # start = time.time()
        # for i in range(100):
        #     out = spmm_sum(self.dcsr, self.input_feature, self.algorithm)
        #     out.sum().backward()
        # torch.cuda.synchronize()
        # end = time.time()
        # dgsparse_time = end - start
        # return 0, dgsparse_time


class SpMMMax:
    def __init__(self, data, device, algorithm) -> None:
        # prepare for torch and dgsparse
        self.tcsr = data.tcsr

        self.dcsr = SparseTensor.from_torch_sparse_csr_tensor(
            self.tcsr.clone().detach(), True, requires_grad=True)
        self.adj_t = data.adj_t
        # self.dgl_A = data.dgl_A
        self.dgl_graph = data.dgl_graph

        self.device = device
        self.algorithm = algorithm
        self.input_feature = data.features

    def forward_check(self):
        # warm up
        for _ in range(10):
            matmul(self.adj_t, self.input_feature, reduce='max')
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            matmul(self.adj_t, self.input_feature, reduce='max')
        torch.cuda.synchronize()
        end = time.time()
        torch_sparse_time = end - start

        # warm up
        for _ in range(10):
            dglsp.spmm(self.dgl_A, self.input_feature)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            dglsp.spmm(self.dgl_A, self.input_feature)
        torch.cuda.synchronize()
        end = time.time()
        dgl_time = end - start

        # warm up
        for _ in range(10):
            spmm_max(self.dcsr, self.input_feature, self.algorithm)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            spmm_max(self.dcsr, self.input_feature, self.algorithm)
        torch.cuda.synchronize()
        end = time.time()
        dgsparse_time = end - start

        return torch_sparse_time, dgl_time, dgsparse_time

    def backward_check(self):
        pass


class SpMMMin:
    def __init__(self, data, device, algorithm) -> None:
        # prepare for torch and dgsparse
        self.tcsr = data.tcsr

        self.dcsr = SparseTensor.from_torch_sparse_csr_tensor(
            self.tcsr.clone().detach(), True, requires_grad=True)
        self.adj_t = data.adj_t
        # self.dgl_A = data.dgl_A
        self.dgl_graph = data.dgl_graph

        self.device = device
        self.algorithm = algorithm
        self.input_feature = data.features

    def forward_check(self):
        # warm up
        for _ in range(10):
            matmul(self.adj_t, self.input_feature, reduce='min')
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            matmul(self.adj_t, self.input_feature, reduce='min')
        torch.cuda.synchronize()
        end = time.time()
        torch_sparse_time = end - start

        # warm up
        for _ in range(10):
            dglsp.spmm(self.dgl_A, self.input_feature)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            dglsp.spmm(self.dgl_A, self.input_feature)
        torch.cuda.synchronize()
        end = time.time()
        dgl_time = end - start

        # warm up
        for _ in range(10):
            spmm_min(self.dcsr, self.input_feature, self.algorithm)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            spmm_min(self.dcsr, self.input_feature, self.algorithm)
        torch.cuda.synchronize()
        end = time.time()
        dgsparse_time = end - start

        return torch_sparse_time, dgl_time, dgsparse_time

    def backward_check(self):
        pass


class SpMMMean:
    def __init__(self, data, device, algorithm) -> None:
        # prepare for torch and dgsparse
        self.tcsr = data.tcsr

        self.dcsr = SparseTensor.from_torch_sparse_csr_tensor(
            self.tcsr.clone().detach(), True, requires_grad=True)
        self.adj_t = data.adj_t
        # self.dgl_A = data.dgl_A
        self.dgl_graph = data.dgl_graph

        self.device = device
        self.algorithm = algorithm
        self.input_feature = data.features

    def forward_check(self):
        # warm up
        for _ in range(10):
            matmul(self.adj_t, self.input_feature, reduce='mean')
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            matmul(self.adj_t, self.input_feature, reduce='mean')
        torch.cuda.synchronize()
        end = time.time()
        torch_sparse_time = end - start

        # warm up
        for _ in range(10):
            dglsp.spmm(self.dgl_A, self.input_feature)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            dglsp.spmm(self.dgl_A, self.input_feature)
        torch.cuda.synchronize()
        end = time.time()
        dgl_time = end - start

        # warm up
        for _ in range(10):
            spmm_mean(self.dcsr, self.input_feature, self.algorithm)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            spmm_mean(self.dcsr, self.input_feature, self.algorithm)
        torch.cuda.synchronize()
        end = time.time()
        dgsparse_time = end - start

        return torch_sparse_time, dgl_time, dgsparse_time

    def backward_check(self):
        pass


def check_time(gc, direction='forward'):
    print(f'{direction} time:')
    torch_sparse_time_list = []
    dgl_time_list = []
    dgsparse_time_list = []
    if direction == 'forward':
        torch_sparse_time, dgl_time, dgsparse_time = gc.forward_check()
    elif direction == 'backward':
        torch_sparse_time, dgsparse_time = gc.backward_check()
    else:
        raise ValueError
    torch_sparse_time_list.append(torch_sparse_time)
    dgl_time_list.append(dgl_time)
    dgsparse_time_list.append(dgsparse_time)
    print(f'torch_sparse forward time is: {torch_sparse_time_list}')
    print(f'dgl forward time is: {dgl_time_list}')
    print(f'dgsparse forward time is: {dgsparse_time_list}')


def test_spmm_time(dataset, device, reduce='sum'):
    print()
    print(f'start testing {dataset} dataset, reduce is: {reduce}')
    data = GraphDataset(dataset, device)
    if reduce == 'sum':
        gc = SpMMSum(data, device, 0)
    elif reduce == 'max':
        gc = SpMMMax(data, device, 0)
    elif reduce == 'min':
        gc = SpMMMin(data, device, 0)
    elif reduce == 'mean':
        gc = SpMMMean(data, device, 0)
    else:
        raise ValueError
    check_time(gc, direction='forward')
    # check_time(gc, direction="backward")


if __name__ == '__main__':
    # datasets = ["cora", "citeseer", "pubmed", "ppi"]
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    datasets = ['cora', 'citeseer', 'pubmed', 'ppi', 'reddit']
    for dataset in datasets:
        test_spmm_time(dataset, device, reduce='sum')
        test_spmm_time(dataset, device, reduce='max')
        test_spmm_time(dataset, device, reduce='min')
        test_spmm_time(dataset, device, reduce='mean')
