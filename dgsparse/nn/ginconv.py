import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_sparse
from dgsparse import spmm_sum, spmm_max, spmm_mean, SparseTensor


class GINConv(nn.Module):

    def __init__(
        self,
        apply_func=None,
        aggregator_type='sum',
        init_eps=0,
        learn_eps=False,
        activation=None,
        cached=False,
    ):
        super().__init__()
        self.apply_func = apply_func
        self._aggregator_type = aggregator_type
        self.activation = activation
        self.cached = cached
        self._cached_dcsr = None
        if learn_eps:
            self.eps = nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))

    def forward(self, edge_index, X, num_nodes):
        neigh = self.aggregate_neigh(edge_index, X, num_nodes, 0)
        rst = (1 + self.eps) * X + neigh

        if self.apply_func is not None:
            rst = self.apply_func(rst)
        if self.activation is not None:
            rst = self.activation(rst)
        return rst

    def aggregate_neigh(self, edge_index, X, num_nodes, algorithm):
        adj_t = torch_sparse.SparseTensor(row=edge_index[0],
                                          col=edge_index[1],
                                          sparse_sizes=(num_nodes, num_nodes))
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1.0)
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
            device=edge_index.device,
        )
        dcsr = SparseTensor.from_torch_sparse_csr_tensor(tcsr.clone().detach(),
                                                         True,
                                                         requires_grad=True)
        if self._aggregator_type == 'sum':
            rst = spmm_sum(dcsr, X, algorithm)
        elif self._aggregator_type == 'max':
            rst = spmm_max(dcsr, X, algorithm)
        elif self._aggregator_type == 'mean':
            rst = spmm_mean(dcsr, X, algorithm)
        else:
            rst = spmm_sum(dcsr, X, algorithm)
        return rst


class GIN(nn.Module):

    def __init__(
        self,
        in_size,
        out_size,
        hidden_size,
        aggregator_type='sum',
        init_eps=0,
        learn_eps=False,
        activation=F.relu,
        cached=False,
    ):
        super().__init__()
        self.conv1 = GINConv(
            nn.Linear(in_size, hidden_size),
            aggregator_type,
            init_eps,
            learn_eps,
            activation,
            cached,
        )
        self.conv2 = GINConv(
            nn.Linear(hidden_size, out_size),
            aggregator_type,
            init_eps,
            learn_eps,
            activation,
            cached,
        )

    def forward(self, edge_index, X, num_nodes):
        X = self.conv1(edge_index, X, num_nodes)
        X = self.conv2(edge_index, X, num_nodes)

        return X

    @property
    def eps(self):
        return self.conv1.eps
