import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from catboost import CatBoostClassifier
import torch
from torch_geometric.loader import DataLoader as GraphDataLoader

from src.dataset import (
    load_herg_data_split,
    load_pampa_data_split,
    load_cyp_data_split,
    load_synthetic_data_split
)
from src.early_stopping import EarlyStopping
from src.featurizers import ECFPFeaturizer, GraphFeaturizer
from src.models.gnn import GraphConvolutionalNetwork
from tuning_results import (
    herg_gnn_params,
    pampa_gnn_params,
    cyp_gnn_params,
    synthetic_gnn_params,
    herg_catboost_params,
    pampa_catboost_params,
    cyp_catboost_params,
    synthetic_catboost_params
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_tuned_gnn(train, test, y_col, best_params, model_path):
    graph_featurizer = GraphFeaturizer(y_col=y_col, log_target_transform=False)
    graph_train = graph_featurizer(train)
    graph_test = graph_featurizer(test)

    model = GraphConvolutionalNetwork(
        input_dim=graph_train[0].x.shape[1],
        hidden_size=best_params["hidden_size"],
        n_layers=best_params["num_layers"],
        dropout=best_params["dropout"]
    ).to(device)

    batch_size = 64
    graph_train_loader = GraphDataLoader(graph_train, batch_size, shuffle=True)
    graph_test_loader = GraphDataLoader(graph_test, batch_size, shuffle=False)

    early_stopping = EarlyStopping(patience=10, min_delta=0.0001)
    model.train_model(graph_train_loader, graph_test_loader, epochs=9999, lr=best_params["lr"],
                      early_stopping=early_stopping)
    torch.save(model.state_dict(), model_path)


def train_tuned_catboost(train, test, y_col, best_params, model_path):
    fp_size = 2048
    fp_radius = 2
    featurizer = ECFPFeaturizer(y_col, length=fp_size, radius=fp_radius, log_target_transform=False)
    X_train, y_train = featurizer(train)
    X_test, y_test = featurizer(test)

    model = CatBoostClassifier(
        iterations=9999,
        learning_rate=best_params["lr"],
        max_depth=best_params["max_depth"],
        l2_leaf_reg=best_params["l2_leaf_reg"],
        verbose=100
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=100)
    model.save_model(model_path)


if __name__ == "__main__":
    train_herg, _, test_herg = load_herg_data_split()
    train_tuned_gnn(train_herg, test_herg, "y", herg_gnn_params, "models/gnn_tuned_herg.pth")
    train_tuned_catboost(train_herg, test_herg, "y", herg_catboost_params, "models/catboost_tuned_herg.cbm")

    train_pampa, _, test_pampa = load_pampa_data_split()
    train_tuned_gnn(train_pampa, test_pampa, "y", pampa_gnn_params, "models/gnn_tuned_pampa.pth")
    train_tuned_catboost(train_pampa, test_pampa, "y", pampa_catboost_params, "models/catboost_tuned_pampa.cbm")

    train_cyp, _, test_cyp = load_cyp_data_split()
    train_tuned_gnn(train_cyp, test_cyp, "y", cyp_gnn_params, "models/gnn_tuned_cyp.pth")
    train_tuned_catboost(train_cyp, test_cyp, "y", cyp_catboost_params, "models/catboost_tuned_cyp.cbm")

    train_synthetic, _, test_synthetic = load_synthetic_data_split()
    train_tuned_gnn(train_synthetic, test_synthetic, "y", synthetic_gnn_params, "models/gnn_tuned_synthetic.pth")
    train_tuned_catboost(train_synthetic, test_synthetic, "y", synthetic_catboost_params, "models/catboost_tuned_synthetic.cbm")