import torch
# import mkl
from scipy.io import mmread
from dgsparse import SparseTensor
from dgsparse import csr2csc
# from dgsparse import spmm_sum


class Csr2Csc:

    def __init__(self, path, device):
        sparsecsr = mmread(path).astype('float32').tocsr()
        self.device = device
        self.sparsecsr = sparsecsr
        self.rowptr = torch.from_numpy(sparsecsr.indptr).to(device).int()
        self.colind = torch.from_numpy(sparsecsr.indices).to(device).int()
        self.weight = torch.from_numpy(sparsecsr.data).to(device).float()

        nodes = self.rowptr.size(0) - 1
        in_dim = 32
        self.input_feature = torch.rand((nodes, in_dim),
                                        requires_grad=True,
                                        device=device)
        shape = sparsecsr.shape
        self.rows = shape[0]
        self.cols = shape[1]

        self.tcsr = torch.sparse_csr_tensor(
            self.rowptr,
            self.colind,
            self.weight,
            dtype=torch.float,
            size=(self.rows, self.cols),
            requires_grad=True,
            device=self.device,
        )
        self.dcsr = SparseTensor.from_torch_sparse_csr_tensor(
            self.tcsr.clone().detach(), True, requires_grad=True)

    def check(self):
        colptr, row, weight_transpose = csr2csc(self.dcsr)
        tran_s = self.sparsecsr.tocsc()
        colptr_check = torch.from_numpy(tran_s.indptr).to(self.device).int()
        row_check = torch.from_numpy(tran_s.indices).to(self.device).int()
        data_check = torch.from_numpy(tran_s.data).to(self.device).float()

        assert torch.allclose(colptr, colptr_check)
        assert torch.allclose(row, row_check)
        assert torch.allclose(weight_transpose, data_check)


def test_csr2csc():
    gc = Csr2Csc('../example/data/p2p-Gnutella31.mtx', 0)
    gc.check()
    # cpu check
    # gc = Csr2Csc("../example/data/p2p-Gnutella31.mtx", "cpu")
    # gc.check()
