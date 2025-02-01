import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from catboost import CatBoostClassifier
import optuna
from torch_geometric.loader import DataLoader as GraphDataLoader

from src.dataset import (
    load_cyp_data_split,
    load_herg_data_split,
    load_pampa_data_split,
    load_synthetic_data_split
)
from src.featurizers import GraphFeaturizer, ECFPFeaturizer
from src.models.gnn import GraphConvolutionalNetwork
from src.early_stopping import EarlyStopping

device = "cuda"


def tune_gnn_hyperparameters(train, val, y_col, n_trials):
    def objective(trial):
        graph_featurizer = GraphFeaturizer(y_col=y_col, log_target_transform=False)
        graph_train = graph_featurizer(train)
        graph_val = graph_featurizer(val)

        train_loader = GraphDataLoader(graph_train, batch_size=64, shuffle=True)
        val_loader = GraphDataLoader(graph_val, batch_size=64, shuffle=False)

        hidden_size = trial.suggest_int("hidden_size", 32, 256)
        n_layers = trial.suggest_int("num_layers", 1, 4)
        dropout = trial.suggest_float("dropout", 0.2, 0.5)
        lr = trial.suggest_float("lr", 1e-3, 1e-2)

        model = GraphConvolutionalNetwork(
            input_dim=graph_train[0].x.shape[1],
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout=dropout,
        ).to(device)
        early_stopping = EarlyStopping(patience=10, min_delta=0.0001)
        model.train_model(train_loader, val_loader, lr=lr, epochs=9999, early_stopping=early_stopping, verbose=False)
        return early_stopping.val_loss_min
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def tune_catboost_hyperparameters(train, val, y_col, n_trials):
    def objective(trial):
        fp_size = 2048
        fp_radius = 2
        featurizer = ECFPFeaturizer(y_col, length=fp_size, radius=fp_radius, log_target_transform=False)
        X_train, y_train = featurizer(train)
        X_val, y_val = featurizer(val)

        lr = trial.suggest_float("lr", 1e-2, 1e-1)
        max_depth = trial.suggest_int("max_depth", 3, 9)
        l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1e-2, 1e1)
        model = CatBoostClassifier(
            iterations=9999,
            learning_rate=lr,
            max_depth=max_depth,
            l2_leaf_reg=l2_leaf_reg,
            verbose=100
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100)
        return model.best_score_["validation"]["Logloss"]
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


if __name__ == "__main__":
    train, val, _ = load_herg_data_split()
    herg_best_params = tune_gnn_hyperparameters(train, val, y_col="y", n_trials=50)

    train, val, _ = load_pampa_data_split()
    pampa_best_params = tune_gnn_hyperparameters(train, val, y_col="y", n_trials=50)

    train, val, _ = load_cyp_data_split()
    cyp_best_params = tune_gnn_hyperparameters(train, val, y_col="y", n_trials=50)

    train, val, _ = load_synthetic_data_split()
    synthetic_best_params = tune_gnn_hyperparameters(train, val, y_col="y", n_trials=50)

    results_file = "tuning_results.txt"
    with open(results_file, "w") as f:
        f.write("GNN:\n\n")
        f.write(f"hERG dataset: \n{herg_best_params}\n")
        f.write(f"PAMPA dataset: \n{pampa_best_params}\n")
        f.write(f"CYP3A4 dataset: \n{cyp_best_params}\n")
        f.write(f"Synthetic dataset: \n{synthetic_best_params}\n")

    herg_best_params = tune_catboost_hyperparameters(train, val, y_col="y", n_trials=50)
    pampa_best_params = tune_catboost_hyperparameters(train, val, y_col="y", n_trials=50)
    cyp_best_params = tune_catboost_hyperparameters(train, val, y_col="y", n_trials=50)
    synthetic_best_params = tune_catboost_hyperparameters(train, val, y_col="y", n_trials=50)

    with open(results_file, "a") as f:
        f.write("\n\nCatBoost:\n\n")
        f.write(f"hERG dataset: \n{herg_best_params}\n")
        f.write(f"PAMPA dataset: \n{pampa_best_params}\n")
        f.write(f"CYP3A4 dataset: \n{cyp_best_params}\n")
        f.write(f"Synthetic dataset: \n{synthetic_best_params}\n")
