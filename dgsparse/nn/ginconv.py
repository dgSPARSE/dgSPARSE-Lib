import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.sparse as dglsp
from dgsparse import spmm_sum, spmm_max, spmm_mean, SparseTensor


class GINConv(nn.Module):
    def __init__(
        self,
        apply_func=None,
        aggregator_type="sum",
        init_eps=0,
        learn_eps=False,
        activation=None,
    ):
        super().__init__()
        self.apply_func = apply_func
        self._aggregator_type = aggregator_type
        self.activation = activation
        if learn_eps:
            self.eps = nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer("eps", torch.FloatTensor([init_eps]))

    def forward(self, A, X):
        I_plus_eps = dglsp.diag(
            torch.zeros(A.shape[0], device=A.device) + self.eps, A.shape
        )
        A_hat = A + I_plus_eps
        A_hat_csr = A_hat.csr()
        A_hat_d_csr = SparseTensor(
            row=None,
            rowptr=A_hat_csr[0].to(torch.int),
            col=A_hat_csr[1].to(torch.int),
            values=A_hat.val,
            has_value=True,
        )
        if self._aggregator_type == "sum":
            spmm_func = spmm_sum
        elif self._aggregator_type == "max":
            spmm_func = spmm_max
        else:
            spmm_func = spmm_mean
        rst = spmm_func(A_hat_d_csr, X, algorithm=1)

        if self.apply_func is not None:
            rst = self.apply_func(rst)
        if self.activation is not None:
            rst = self.activation(rst)

        return rst


class GIN(nn.Module):
    def __init__(
        self,
        in_size,
        hidden_size,
        out_size,
        aggregator_type="sum",
        init_eps=0,
        learn_eps=False,
        activation=None,
    ):
        super().__init__()
        self.conv1 = GINConv(
            nn.Linear(in_size, hidden_size, bias=False),
            aggregator_type,
            init_eps,
            learn_eps,
            activation,
        )
        self.conv2 = GINConv(
            nn.Linear(hidden_size, out_size, bias=False),
            aggregator_type,
            init_eps,
            learn_eps,
            activation,
        )

    def forward(self, A, X):
        X = self.conv1(A, X)
        X = self.conv2(A, X)

        return X

    @property
    def eps(self):
        return self.conv1.eps


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = dgl.data.CoraGraphDataset()
    g = dataset[0]
    feature = g.ndata["feat"]
    in_size = feature.shape[1]
    out_size = dataset.num_classes
    gin_model = GIN(in_size, 32, out_size, activation=F.rrelu)
    gin_model = gin_model.to(device)

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

        for epoch in range(101):
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

    def evaluate(g, pred):
        label = g.ndata["label"].to(device)
        val_mask = g.ndata["val_mask"].to(device)
        test_mask = g.ndata["test_mask"].to(device)

        # Compute accuracy on validation/test set.
        val_acc = (pred[val_mask] == label[val_mask]).float().mean()
        test_acc = (pred[test_mask] == label[test_mask]).float().mean()
        return val_acc, test_acc

    train(gin_model, g)
