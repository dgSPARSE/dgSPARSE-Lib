import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import time

import dgl.sparse as dglsp
from dgsparse.nn.gcnconv import GCN

device = "cuda" if torch.cuda.is_available() else "cpu"


class GCNLayer(nn.Module):
    def __init__(self, in_size, out_size):
        super(GCNLayer, self).__init__()
        self.W = nn.Linear(in_size, out_size)

    def forward(self, A, X):
        ########################################################################
        # (HIGHLIGHT) Compute the symmetrically normalized adjacency matrix with
        # Sparse Matrix API
        ########################################################################
        I = dglsp.identity(A.shape, device=A.device)
        A_hat = A + I
        D_hat = dglsp.diag(A_hat.sum(0))
        D_hat_invsqrt = D_hat**-0.5
        return D_hat_invsqrt @ A_hat @ D_hat_invsqrt @ self.W(X)


# Create a GCN with the GCN layer.
class GCN_dgl(nn.Module):
    def __init__(self, in_size, out_size, hidden_size):
        super(GCN, self).__init__()
        self.conv1 = GCNLayer(in_size, hidden_size)
        self.conv2 = GCNLayer(hidden_size, out_size)

    def forward(self, A, X):
        X = self.conv1(A, X)
        X = F.relu(X)
        return self.conv2(A, X)


def evaluate(g, pred):
    label = g.ndata["label"].to(device)
    val_mask = g.ndata["val_mask"].to(device)
    test_mask = g.ndata["test_mask"].to(device)

    # Compute accuracy on validation/test set.
    val_acc = (pred[val_mask] == label[val_mask]).float().mean()
    test_acc = (pred[test_mask] == label[test_mask]).float().mean()
    return val_acc, test_acc


def train(model, g):
    features = g.ndata["feat"].to(device)
    label = g.ndata["label"].to(device)
    train_mask = g.ndata["train_mask"].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    loss_fcn = nn.CrossEntropyLoss()

    # Preprocess to get the adjacency matrix of the graph.
    indices = torch.stack(g.edges()).to(device)
    N = g.num_nodes()
    A = dglsp.spmatrix(indices, shape=(N, N))

    for epoch in range(100):
        model.train()

        # Forward.
        logits = model(A, features)

        # Compute loss with nodes in the training set.
        loss = loss_fcn(logits[train_mask], label[train_mask])

        # Backward.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute prediction.
        pred = logits.argmax(dim=1)

        # Evaluate the prediction.
        val_acc, test_acc = evaluate(g, pred)
        if epoch % 5 == 0:
            print(
                f"In epoch {epoch}, loss: {loss:.3f}, val acc: {val_acc:.3f}"
                f", test acc: {test_acc:.3f}"
            )


# Load graph from the existing dataset.
dataset = dgl.data.CoraGraphDataset()
g = dataset[0]

# Create model.s
feature = g.ndata["feat"]
in_size = feature.shape[1]
out_size = dataset.num_classes
gcn_model = GCN_dgl(in_size, out_size, 16)
gcn_model = gcn_model.to(device)
dg_gcn_model = GCN(in_size, out_size, 16)
dg_gcn_model = dg_gcn_model.to(device)

# Kick off training.
start = time.time()
train(gcn_model, g)
end = time.time()
print(f"dgl time is: {end - start}")

start = time.time()
train(dg_gcn_model, g)
end = time.time()
print(f"dgsparse time is: {end - start}")

# features = g.ndata["feat"]
# label = g.ndata["label"]
# train_mask = g.ndata["train_mask"]

# indices = torch.stack(g.edges()).to(device)
# N = g.num_nodes()
# A = dglsp.spmatrix(indices, shape=(N, N))

# print(type(features), features.shape)
# print(type(label), label.shape)
# print(type(train_mask), train_mask.shape)
# print(type(A), A.shape, A.device)
# print(A)
# print(A.csr())
# for i in A.csr():
#     print(i.shape)

# indices = torch.tensor([[1, 2, 1], [2, 4, 3]])
# A = dglsp.spmatrix(indices, val=torch.tensor([1, 2, 3]))
# print(A)
# print(A.csr())
# print(A.val)
# print(A.row)

# I = dglsp.identity(A.shape)
# A_hat = A + I
# D_hat = dglsp.diag(A_hat.sum(0))
# D_hat_invsqrt = D_hat ** -0.5
# print(type(A_hat), A_hat)
# print(type(D_hat), D_hat)
# H = D_hat_invsqrt @ A_hat @ D_hat_invsqrt
# print(type(H), H)
