from torch_geometric.nn import (
    GCNConv,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
)
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GraphConvolutionalNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, n_layers: int, dropout: float = 0.5):
        super(GraphConvolutionalNetwork, self).__init__()

        self.gcn = nn.ModuleList(
            [GCNConv(input_dim, hidden_size)] +
            [GCNConv(hidden_size, hidden_size)] * (n_layers - 1)
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Sequential(
            nn.Linear(3 * hidden_size, 3 * hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(3 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

        self.input = None
        self.final_conv_activations = None
        self.final_conv_grads = None

    def activation_hook(self, grad):
        self.final_conv_grads = grad.detach().clone()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        self.input = x
        if x.is_leaf:
            x.requires_grad = True
        h = x.clone()
        for conv in self.gcn[:-1]:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = self.dropout(h)
        last_conv = self.gcn[-1]
        with torch.enable_grad():
            self.final_conv_activations = last_conv(h, edge_index)
        self.final_conv_activations.register_hook(self.activation_hook)
        h = F.relu(self.final_conv_activations)
        h = self.dropout(h)

        pool1 = global_mean_pool(h, batch)
        pool2 = global_max_pool(h, batch)
        pool3 = global_add_pool(h, batch)
        h = torch.cat([pool1, pool2, pool3], dim=1)
        h = self.linear(h)
        h = nn.Sigmoid()(h)
        return h

    def train_model(self, train_loader, test_loader, epochs=10, lr=0.001, early_stopping=None, verbose=True):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                output = self(batch)
                loss = F.binary_cross_entropy(output.squeeze(), batch.y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            with torch.no_grad():
                self.eval()
                test_loss = 0.0
                for batch in test_loader:
                    batch = batch.to(device)
                    output = self(batch)
                    loss = F.binary_cross_entropy(output.squeeze(), batch.y)
                    test_loss += loss.item()
                test_loss /= len(test_loader)
                if verbose:
                    print(f'Epoch: {epoch}, train loss: {train_loss:.4f}, test loss: {test_loss:.4f}')

            if early_stopping:
                early_stopping(self, test_loss)
                if early_stopping.stop:
                    if verbose:
                        print(f"Early stopping on epoch {epoch}")
                    self.load_state_dict(early_stopping.get_best_model_parameters())
                    break

    def r2_score(self, test_loader):
        """
        Calculate the R^2 score of the model on the test set
        """
        with torch.no_grad():
            data_sum = 0.0
            n_observations = 0
            predictions = []
            actuals = []
            for batch in test_loader:
                batch.x, batch.y = batch.x.to(device), batch.y.to(device)
                batch.edge_index, batch.batch = batch.edge_index.to(device), batch.batch.to(device)
                output = self(batch.x, batch.edge_index, batch.batch)
                data_sum += batch.y.sum().item()
                predictions.append(output.squeeze().detach().cpu())
                actuals.append(batch.y.detach().cpu())
                n_observations += batch.y.shape[0]
            predictions = torch.concatenate(predictions)
            actuals = torch.concatenate(actuals)
            data_mean = data_sum / n_observations
            r_squared = ((predictions - data_mean) ** 2).sum() / ((actuals - data_mean) ** 2).sum()
        return r_squared.item()
