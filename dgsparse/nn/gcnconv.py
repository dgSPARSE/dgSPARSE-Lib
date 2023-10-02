import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_sparse
from torch_sparse import fill_diag, mul
from torch_sparse import sum as sparsesum
from dgsparse import spmm_sum, SparseTensor


class GCNConv(nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()
        self.W = nn.Linear(in_size, out_size, bias=False)

    def forward(self, dcsr, x):
        x = self.W(x)
        x = spmm_sum(dcsr, x, 0)
        return x


class GCN(nn.Module):

    def __init__(self, in_size, out_size, hidden_size):
        super().__init__()
        self.conv1 = GCNConv(in_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, out_size)

    def forward(self, dcsr, x):
        x = self.conv1(dcsr, x)
        x = F.relu(x)
        x = self.conv2(dcsr, x)

        return x


def gcn_norm_from_edge_index(edge_index, num_nodes, add_self_loops=True):
    adj_t = torch_sparse.SparseTensor(row=edge_index[0],
                                      col=edge_index[1],
                                      sparse_sizes=(num_nodes, num_nodes))
    if not adj_t.has_value():
        adj_t = adj_t.fill_value(1.0)
    if add_self_loops:
        adj_t = fill_diag(adj_t, 1.0)
    deg = sparsesum(adj_t, dim=1)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.0)
    adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
    adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
    return adj_t


def get_gcn_dcsr_from_edge_index(edge_index, num_nodes):
    adj_t = gcn_norm_from_edge_index(edge_index, num_nodes)
    rowptr, col, value = adj_t.csr()
    rowptr = rowptr.int()
    col = col.int()
    tcsr = torch.sparse_csr_tensor(
        rowptr,
        col,
        value,
        dtype=torch.float,
        size=(num_nodes, num_nodes),
        requires_grad=True,
        device=adj_t.device(),
    )
    dcsr = SparseTensor.from_torch_sparse_csr_tensor(tcsr.clone().detach(),
                                                     True,
                                                     requires_grad=True)
    return dcsr
