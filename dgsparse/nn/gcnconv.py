import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.sparse as dglsp
from dgsparse import spmm_sum, SparseTensor


class GCNConv(nn.Module):
    def __init__(self, in_size, out_size):
        super(GCNConv, self).__init__()
        self.W = nn.Linear(in_size, out_size, bias=False)

    def forward(self, A, X):
        I = dglsp.identity(A.shape, device=A.device)
        A_hat = A + I
        D_hat = dglsp.diag(A_hat.sum(0))
        D_hat_invsqrt = D_hat**-0.5
        H = D_hat_invsqrt @ A_hat @ D_hat_invsqrt
        H_csr = H.csr()
        H_d_csr = SparseTensor(
            row=None,
            rowptr=H_csr[0].to(torch.int),
            col=H_csr[1].to(torch.int),
            values=H.val,
            has_value=True,
        )
        X_hat = self.W(X)
        return spmm_sum(H_d_csr, X_hat, algorithm=0)


class GCN(nn.Module):
    def __init__(self, in_size, out_size, hidden_size):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, out_size)

    def forward(self, A, X):
        X = self.conv1(A, X)
        X = F.relu(X)
        return self.conv2(A, X)
