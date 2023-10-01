import dgl
import torch as th
from dgl.nn import GINConv
import torch
import dgl.sparse as dglsp

device = 'cuda' if torch.cuda.is_available() else 'cpu'

g = dgl.graph(([0, 1, 2, 3, 2, 5], [1, 2, 3, 4, 0, 3]))
feat = th.ones(6, 10)
lin = th.nn.Linear(10, 10)
conv = GINConv()
res = conv(g, feat)
print(res)

indices = torch.stack(g.edges()).to(device)
N = g.num_nodes()
A = dglsp.spmatrix(indices, shape=(N, N))
print(A)
