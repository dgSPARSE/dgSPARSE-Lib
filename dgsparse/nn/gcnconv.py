import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_sparse
from torch_sparse import fill_diag, mul
from torch_sparse import sum as sparsesum
from dgsparse import spmm_sum, SparseTensor


class GCNConv(nn.Module):
    def __init__(self, in_size, out_size, cached=False):
        super().__init__()
        self.W = nn.Linear(in_size, out_size, bias=False)
        self.cached = cached
        self._cached_adj_t = None
    
    def forward(self, edge_index, x, num_nodes):
        cache = self._cached_adj_t
        if cache is None:
            adj_t = self.gcn_norm(edge_index, num_nodes)
            rowptr, col, value = adj_t.csr()
            rowptr = rowptr.int()
            col = col.int()
            tcsr = torch.sparse_csr_tensor(
                rowptr, col, value, dtype=torch.float, size=(num_nodes, num_nodes),
                requires_grad=True,
                device=edge_index.device
            )
            dcsr = SparseTensor.from_torch_sparse_csr_tensor(
                tcsr.clone().detach(), True, requires_grad=True
            )
            if self.cached:
                self._cached_adj_t = dcsr
        else:
            dcsr = cache
        x = self.W(x)
        return spmm_sum(dcsr, x, 0)

    def gcn_norm(self, edge_index, num_nodes, add_self_loops=True):
        adj_t = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes))
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1.)
        if add_self_loops:
            adj_t = fill_diag(adj_t, 1.)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t


class GCN(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, cached=False):
        super().__init__()
        self.conv1 = GCNConv(in_size, hidden_size, cached)
        self.conv2 = GCNConv(hidden_size, out_size, cached)

    def forward(self, edge_index, x, num_nodes):
        x = self.conv1(edge_index, x, num_nodes)
        x = F.relu(x)
        x = self.conv2(edge_index, x, num_nodes)
        return x


