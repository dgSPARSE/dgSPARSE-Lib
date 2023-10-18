import torch
from dgsparse import spmm_sum, spmm_max, spmm_min, spmm_mean
from dgsparse import SparseTensor
# import pytest
# import torch_sparse
import time
import dgl
# import tqdm
from utils import GraphDataset


class SpMMSum:

    def __init__(self, data, in_dim, device, algorithm) -> None:
        # prepare for torch and dgsparse
        self.tcsr = data.tcsr
        self.dcsr = SparseTensor.from_torch_sparse_csr_tensor(
            self.tcsr.clone().detach(), True, requires_grad=True)
        self.adj_t = data.adj_t
        # self.dgl_A = data.dgl_A
        self.dgl_graph = data.dgl_graph

        self.device = device
        self.algorithm = algorithm
        # self.input_feature = data.features
        self.input_feature = torch.rand((data.num_nodes, in_dim),
                                        device=device,
                                        requires_grad=True)
        # self.input_feature =
        # torch.randn((self.dcsr.storage.sparse_sizes[1], 1)).to(device)
        # print(self.input_feature.size())

    def forward_check(self):
        # warm up
        for _ in range(10):
            # out = matmul(self.adj_t, self.input_feature, reduce="sum")
            self.adj_t.spmm(self.input_feature, reduce='sum')
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            # out = matmul(self.adj_t, self.input_feature, reduce="sum")
            self.adj_t.spmm(self.input_feature, reduce='sum')
        torch.cuda.synchronize()
        end = time.time()
        torch_sparse_time = end - start

        # warm up
        for _ in range(10):
            dgl.ops.copy_u_sum(self.dgl_graph, self.input_feature)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            dgl.ops.copy_u_sum(self.dgl_graph, self.input_feature)
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
        #warm up
        for _ in range(10):
            out = self.adj_t.spmm(self.input_feature, reduce='sum')
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            #     out_check = torch_sparse.spmm_sum(self.adj_t, self.input_feature)
            out = self.adj_t.spmm(self.input_feature, reduce='sum')
            out.sum().backward()
        torch.cuda.synchronize()
        end = time.time()
        torch_sparse_time = end - start

        # #warm up
        for _ in range(10):
            out = dgl.ops.copy_u_sum(self.dgl_graph, self.input_feature)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            out = dgl.ops.copy_u_sum(self.dgl_graph, self.input_feature)
            out.sum().backward()
        torch.cuda.synchronize()
        end = time.time()
        dgl_time = end - start

        # warm up
        for _ in range(10):
            out = spmm_sum(self.dcsr, self.input_feature, self.algorithm)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            out = spmm_sum(self.dcsr, self.input_feature, self.algorithm)
            out.sum().backward()
        torch.cuda.synchronize()
        end = time.time()
        dgsparse_time = end - start

        return torch_sparse_time, dgl_time, dgsparse_time


class SpMMMax:

    def __init__(self, data, in_dim, device, algorithm) -> None:
        # prepare for torch and dgsparse
        self.tcsr = data.tcsr

        self.dcsr = SparseTensor.from_torch_sparse_csr_tensor(
            self.tcsr.clone().detach(), True, requires_grad=True)
        self.adj_t = data.adj_t
        # self.dgl_A = data.dgl_A
        self.dgl_graph = data.dgl_graph

        self.device = device
        self.algorithm = algorithm
        # self.input_feature = data.features
        self.input_feature = torch.rand((data.num_nodes, in_dim),
                                        device=device,
                                        requires_grad=True)

    def forward_check(self):
        # warm up
        for _ in range(10):
            # out_check = matmul(self.adj_t, self.input_feature, reduce="max")
            # out = torch_sparse.spmm_max(self.adj_t, self.input_feature)
            self.adj_t.spmm(self.input_feature, reduce='max')
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            # out_check = matmul(self.adj_t, self.input_feature, reduce="max")
            # out = torch_sparse.spmm_max(self.adj_t, self.input_feature)
            self.adj_t.spmm(self.input_feature, reduce='max')
        torch.cuda.synchronize()
        end = time.time()
        torch_sparse_time = end - start

        # warm up
        for _ in range(10):
            dgl.ops.copy_u_max(self.dgl_graph, self.input_feature)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            dgl.ops.copy_u_max(self.dgl_graph, self.input_feature)
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
        #warm up
        for _ in range(10):
            out = self.adj_t.spmm(self.input_feature, reduce='max')
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            #     out_check = torch_sparse.spmm_sum(self.adj_t, self.input_feature)
            out = self.adj_t.spmm(self.input_feature, reduce='max')
            out.sum().backward()
        torch.cuda.synchronize()
        end = time.time()
        torch_sparse_time = end - start

        # #warm up
        for _ in range(10):
            out = dgl.ops.copy_u_max(self.dgl_graph, self.input_feature)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            out = dgl.ops.copy_u_max(self.dgl_graph, self.input_feature)
            out.sum().backward()
        torch.cuda.synchronize()
        end = time.time()
        dgl_time = end - start

        # warm up
        for _ in range(10):
            out = spmm_max(self.dcsr, self.input_feature, self.algorithm)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            out = spmm_max(self.dcsr, self.input_feature, self.algorithm)
            out.sum().backward()
        torch.cuda.synchronize()
        end = time.time()
        dgsparse_time = end - start

        return torch_sparse_time, dgl_time, dgsparse_time


class SpMMMin:

    def __init__(self, data, in_dim, device, algorithm) -> None:
        # prepare for torch and dgsparse
        self.tcsr = data.tcsr

        self.dcsr = SparseTensor.from_torch_sparse_csr_tensor(
            self.tcsr.clone().detach(), True, requires_grad=True)
        self.adj_t = data.adj_t
        # self.dgl_A = data.dgl_A
        self.dgl_graph = data.dgl_graph

        self.device = device
        self.algorithm = algorithm
        # self.input_feature = data.features
        self.input_feature = torch.rand((data.num_nodes, in_dim),
                                        device=device,
                                        requires_grad=True)

    def forward_check(self):
        # warm up
        for _ in range(10):
            # out_check = matmul(self.adj_t, self.input_feature, reduce="min")
            # out = torch_sparse.spmm_min(self.adj_t, self.input_feature)
            self.adj_t.spmm(self.input_feature, reduce='min')
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            # out_check = matmul(self.adj_t, self.input_feature, reduce="min")
            # out = torch_sparse.spmm_min(self.adj_t, self.input_feature)
            self.adj_t.spmm(self.input_feature, reduce='min')
        torch.cuda.synchronize()
        end = time.time()
        torch_sparse_time = end - start

        # warm up
        for _ in range(10):
            dgl.ops.copy_u_min(self.dgl_graph, self.input_feature)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            dgl.ops.copy_u_min(self.dgl_graph, self.input_feature)
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
        #warm up
        for _ in range(10):
            out = self.adj_t.spmm(self.input_feature, reduce='min')
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            #     out_check = torch_sparse.spmm_sum(self.adj_t, self.input_feature)
            out = self.adj_t.spmm(self.input_feature, reduce='min')
            out.sum().backward()
        torch.cuda.synchronize()
        end = time.time()
        torch_sparse_time = end - start

        # #warm up
        for _ in range(10):
            out = dgl.ops.copy_u_min(self.dgl_graph, self.input_feature)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            out = dgl.ops.copy_u_min(self.dgl_graph, self.input_feature)
            out.sum().backward()
        torch.cuda.synchronize()
        end = time.time()
        dgl_time = end - start

        # warm up
        for _ in range(10):
            out = spmm_min(self.dcsr, self.input_feature, self.algorithm)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            out = spmm_min(self.dcsr, self.input_feature, self.algorithm)
            out.sum().backward()
        torch.cuda.synchronize()
        end = time.time()
        dgsparse_time = end - start

        return torch_sparse_time, dgl_time, dgsparse_time


class SpMMMean:

    def __init__(self, data, in_dim, device, algorithm) -> None:
        # prepare for torch and dgsparse
        self.tcsr = data.tcsr

        self.dcsr = SparseTensor.from_torch_sparse_csr_tensor(
            self.tcsr.clone().detach(), True, requires_grad=True)
        self.adj_t = data.adj_t
        # self.dgl_A = data.dgl_A
        self.dgl_graph = data.dgl_graph

        self.device = device
        self.algorithm = algorithm
        # self.input_feature = data.features
        self.input_feature = torch.rand((data.num_nodes, in_dim),
                                        device=device,
                                        requires_grad=True)

    def forward_check(self):
        # warm up
        for _ in range(10):
            # out_check = matmul(self.adj_t, self.input_feature, reduce="mean")
            # out = torch_sparse.spmm_mean(self.adj_t, self.input_feature)
            self.adj_t.spmm(self.input_feature, reduce='mean')
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            # out_check = matmul(self.adj_t, self.input_feature, reduce="mean")
            # out = torch_sparse.spmm_mean(self.adj_t, self.input_feature)
            self.adj_t.spmm(self.input_feature, reduce='mean')
        torch.cuda.synchronize()
        end = time.time()
        torch_sparse_time = end - start

        # warm up
        for _ in range(10):
            dgl.ops.copy_u_mean(self.dgl_graph, self.input_feature)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            dgl.ops.copy_u_mean(self.dgl_graph, self.input_feature)
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
        #warm up
        for _ in range(10):
            out = self.adj_t.spmm(self.input_feature, reduce='mean')
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            #     out_check = torch_sparse.spmm_sum(self.adj_t, self.input_feature)
            out = self.adj_t.spmm(self.input_feature, reduce='mean')
            out.sum().backward()
        torch.cuda.synchronize()
        end = time.time()
        torch_sparse_time = end - start

        # #warm up
        for _ in range(10):
            out = dgl.ops.copy_u_mean(self.dgl_graph, self.input_feature)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            out = dgl.ops.copy_u_mean(self.dgl_graph, self.input_feature)
            out.sum().backward()
        torch.cuda.synchronize()
        end = time.time()
        dgl_time = end - start

        # warm up
        for _ in range(10):
            out = spmm_mean(self.dcsr, self.input_feature, self.algorithm)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            out = spmm_mean(self.dcsr, self.input_feature, self.algorithm)
            out.sum().backward()
        torch.cuda.synchronize()
        end = time.time()
        dgsparse_time = end - start

        return torch_sparse_time, dgl_time, dgsparse_time


def check_time(gc, stage='forward'):
    print(f'{stage} time:')
    torch_sparse_time_list = []
    dgl_time_list = []
    dgsparse_time_list = []
    if stage == 'forward':
        torch_sparse_time, dgl_time, dgsparse_time = gc.forward_check()
    elif stage == 'backward':
        torch_sparse_time, dgl_time, dgsparse_time = gc.backward_check()
    else:
        raise ValueError
    torch_sparse_time_list.append(torch_sparse_time)
    dgl_time_list.append(dgl_time)
    dgsparse_time_list.append(dgsparse_time)
    print(f'torch_sparse {stage} time is: {torch_sparse_time_list}')
    print(f'dgl {stage} time is: {dgl_time_list}')
    print(f'dgsparse {stage} time is: {dgsparse_time_list}')


def test_spmm_time(dataset, in_dim, device, reduce='sum'):
    print()
    print(f'start testing {dataset} dataset, \
        reduce is: {reduce}, in_dim is: {in_dim}')
    data = GraphDataset(dataset, device)
    if reduce == 'sum':
        gc = SpMMSum(data, in_dim, device, 0)
    elif reduce == 'max':
        gc = SpMMMax(data, in_dim, device, 0)
    elif reduce == 'min':
        gc = SpMMMin(data, in_dim, device, 0)
    elif reduce == 'mean':
        gc = SpMMMean(data, in_dim, device, 0)
    else:
        raise ValueError
    check_time(gc, stage='forward')
    check_time(gc, stage='backward')


if __name__ == '__main__':
    # datasets = ["reddit"]
    # features_dim = [32]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    datasets = ['cora', 'citeseer', 'pubmed', 'ppi']
    features_dim = [32, 64, 128]
    for dataset in datasets:
        for in_dim in features_dim:
            test_spmm_time(dataset, in_dim, device, reduce='sum')
            test_spmm_time(dataset, in_dim, device, reduce='max')
            test_spmm_time(dataset, in_dim, device, reduce='min')
            test_spmm_time(dataset, in_dim, device, reduce='mean')
