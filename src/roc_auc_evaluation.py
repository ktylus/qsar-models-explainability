import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import torch
from torch_geometric.loader import DataLoader as GraphDataLoader

from src.dataset import(
    load_herg_data_split,
    load_pampa_data_split,
    load_cyp_data_split,
    load_synthetic_data_split
)
from src.featurizers import ECFPFeaturizer, GraphFeaturizer
from src.models.gnn import GraphConvolutionalNetwork
from tuning_results import (
    herg_gnn_params,
    pampa_gnn_params,
    cyp_gnn_params,
    synthetic_gnn_params
)

device = "cuda"


def evaluate_gnn(test_data, y_col, dataset_name, best_params):
    graph_featurizer = GraphFeaturizer(y_col=y_col, log_target_transform=False)
    graph_test = graph_featurizer(test_data)

    batch_size = 64
    graph_test_loader = GraphDataLoader(graph_test, batch_size, shuffle=False)

    model_path = f"models/gnn_tuned_{dataset_name}.pth"
    model = GraphConvolutionalNetwork(
        input_dim=graph_test[0].x.shape[1],
        hidden_size=best_params["hidden_size"],
        n_layers=best_params["num_layers"],
        dropout=best_params["dropout"]
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    auc = model.evaluate_roc(graph_test_loader)
    return auc


def evaluate_catboost(test_data, y_col, dataset_name):
    fp_size = 2048
    fp_radius = 2
    featurizer = ECFPFeaturizer(y_col, length=fp_size, radius=fp_radius, log_target_transform=False)
    X_test, y_test = featurizer(test_data)

    model_path = f"models/catboost_tuned_{dataset_name}.cbm"
    model = CatBoostClassifier()
    model.load_model(model_path)
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    return auc


if __name__ == "__main__":
    herg_train, _, herg_test = load_herg_data_split()
    herg_auc_gnn = evaluate_gnn(herg_test, "y", "herg", herg_gnn_params)
    herg_auc_catboost = evaluate_catboost(herg_test, "y", "herg")

    pampa_train, _, pampa_test = load_pampa_data_split()
    pampa_auc_gnn = evaluate_gnn(pampa_test, "y", "pampa", pampa_gnn_params)
    pampa_auc_catboost = evaluate_catboost(pampa_test, "y", "pampa")

    cyp_train, _, cyp_test = load_cyp_data_split()
    cyp_auc_gnn = evaluate_gnn(cyp_test, "y", "cyp", cyp_gnn_params)
    cyp_auc_catboost = evaluate_catboost(cyp_test, "y", "cyp")

    synthetic_train, _, synthetic_test = load_synthetic_data_split()
    synthetic_auc_gnn = evaluate_gnn(synthetic_test, "y", "synthetic", synthetic_gnn_params)
    synthetic_auc_catboost = evaluate_catboost(synthetic_test, "y", "synthetic")

    results_file = "evaluation_results.txt"
    with open(results_file, "w") as f:
        f.write("GNN:\n\n")
        f.write(f"hERG dataset: {herg_auc_gnn:.4f}\n")
        f.write(f"PAMPA dataset: {pampa_auc_gnn:.4f}\n")
        f.write(f"CYP3A4 dataset: {cyp_auc_gnn:.4f}\n")
        f.write(f"Synthetic dataset: {synthetic_auc_gnn:.4f}\n\n")
        f.write("\nCatboost:\n\n")
        f.write(f"hERG dataset: {herg_auc_catboost:.4f}\n")
        f.write(f"PAMPA dataset: {pampa_auc_catboost:.4f}\n")
        f.write(f"CYP3A4 dataset: {cyp_auc_catboost:.4f}\n")
        f.write(f"Synthetic dataset: {synthetic_auc_catboost:.4f}\n")