import torch_geometric.datasets as datasets
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_scipy_sparse_matrix
import torch
import torch_sparse


class GraphDataset:
    def __init__(self, name: str, device):
        self.name = name
        self.device = device
        self.get_dataset()

    def get_dataset(self):
        if self.name == 'arxiv':
            arxiv = PygNodePropPredDataset(root='./data/', name='ogbn-arxiv')
            graph = arxiv[0]
        elif self.name == 'proteins':
            proteins = PygNodePropPredDataset(root='./data/',
                                              name='ogbn-proteins')
            graph = proteins[0]
        elif self.name == 'products':
            products = PygNodePropPredDataset(root='./data/',
                                              name='ogbn-products')
            graph = products[0]
        elif self.name == 'pubmed':
            dataset = datasets.Planetoid(root='./data/', name='Pubmed')
            graph = dataset[0]
        elif self.name == 'citeseer':
            dataset = datasets.Planetoid(root='./data/', name='Citeseer')
            graph = dataset[0]
        elif self.name == 'cora':
            dataset = datasets.Planetoid(root='./data/', name='Cora')
            graph = dataset[0]
        elif self.name == 'ppi':
            dataset = datasets.PPI(root='./data/')
            graph = dataset[0]
        elif self.name == 'reddit':
            dataset = datasets.Reddit(root='./data/Reddit')
            graph = dataset[0]
        elif self.name == 'github':
            dataset = datasets.GitHub(root='./data/')
            graph = dataset[0]
        else:
            raise KeyError('Unknown dataset {}.'.format(self.name))
        scipy_coo = to_scipy_sparse_matrix(graph.edge_index,
                                           num_nodes=graph.num_nodes)
        scipy_csr = scipy_coo.tocsr()
        rowptr = scipy_csr.indptr
        col = scipy_csr.indices
        weight = torch.ones(col.shape, requires_grad=True)
        self.num_nodes = graph.num_nodes
        self.tcsr = torch.sparse_csr_tensor(
            rowptr,
            col,
            weight,
            dtype=torch.float,
            size=(self.num_nodes, self.num_nodes),
            requires_grad=True,
            device=self.device,
        )
        adj_t = torch.sparse_csr_tensor(
            torch.tensor(rowptr, dtype=torch.long),
            torch.tensor(col, dtype=torch.long),
            weight,
            dtype=torch.float,
            size=(self.num_nodes, self.num_nodes),
            requires_grad=True,
            device=self.device,
        )
        self.adj_t = torch_sparse.SparseTensor.from_torch_sparse_csr_tensor(
            adj_t)
        self.features = graph.x.to(self.device)
