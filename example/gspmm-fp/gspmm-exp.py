import torch
import scipy
import os
import numpy as np
from scipy.io import mmread
from torch.utils.cpp_extension import load
from torch.nn import Parameter, init
import torch.nn.functional as F
import time
import util
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sparsePath = "../data/p2p-Gnutella31.mtx"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

k = int(sys.argv[1])

sparsecsr = mmread(sparsePath).tocsc().astype('float32')

rowptr = torch.from_numpy(sparsecsr.indptr).to(device).int()
colind = torch.from_numpy(sparsecsr.indices).to(device).int()
nnz = colind.shape[0]
n = rowptr.shape[0] - 1
weight = torch.from_numpy(sparsecsr.data).to(device).float()
node_feature = torch.from_numpy(np.random.rand(n, k)).to(device).float()
edge_feature = torch.from_numpy(np.random.rand(nnz, 1)).to(device).float()

a = time.time()
ue = util.u_sub_e_sum(rowptr, colind, edge_feature, node_feature)
torch.cuda.synchronize()
b = time.time()
time_our_ue = b-a
print(f"running u_sub_e_sum our time is: {time_our_ue:.4f}")