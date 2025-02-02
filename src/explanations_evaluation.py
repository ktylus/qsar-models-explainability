import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from torch_geometric.loader import DataLoader as GraphDataLoader

from src.dataset import (
    load_herg_data_split,
    load_pampa_data_split,
    load_cyp_data_split,
    load_synthetic_data_split
)
from src.explanations import batch_grad_cam, batch_saliency_map
from src.featurizers import GraphFeaturizer
from src.metrics import calculate_batch_explanation_contrastivity, calculate_batch_explanation_sparsity
from src.utils import load_gnn_model
from tuning_results import (
    cyp_gnn_params,
    herg_gnn_params,
    pampa_gnn_params,
    synthetic_gnn_params,
    cyp_catboost_params,
    herg_catboost_params,
    pampa_catboost_params,
    synthetic_catboost_params
)


def evaluate_contrastivity_gnn(model, graph_data, explanation_fn):
    batch_size = 64
    data_loader = GraphDataLoader(graph_data, batch_size, shuffle=False)

    total_contrastivity_score = 0
    total_examples = 0
    for batch in data_loader:
        batch_inv = batch.clone()
        batch_inv.y = 1 - batch.y
        explanation_scores = explanation_fn(model, batch)
        explanation_scores_inv = explanation_fn(model, batch_inv)
        total_contrastivity_score += len(batch) * calculate_batch_explanation_contrastivity(
            explanation_scores, explanation_scores_inv, batch.batch, threshold=0.5)
        total_examples += len(batch)
    return total_contrastivity_score / total_examples


def evaluate_sparsity_gnn(model, graph_data, explanation_fn):
    batch_size = 64
    data_loader = GraphDataLoader(graph_data, batch_size, shuffle=False)

    total_sparsity_score = 0
    total_examples = 0
    for batch in data_loader:
        explanation_scores = explanation_fn(model, batch)
        total_sparsity_score += len(batch) * calculate_batch_explanation_sparsity(explanation_scores, batch.batch,
                                                                                  threshold=0.5)
        total_examples += len(batch)
    return total_sparsity_score / total_examples


def run_experiment(test_data, dataset_name, best_params):
    graph_test_data = graph_featurizer(test_data)
    model = load_gnn_model(test_data, dataset_name, best_params)
    contrastivity_gradcam = evaluate_contrastivity_gnn(model, graph_test_data, batch_grad_cam)
    contrastivity_saliency_map = evaluate_contrastivity_gnn(model, graph_test_data, batch_saliency_map)
    print(f"Contrastivity for GradCAM on {dataset_name} dataset: {contrastivity_gradcam:.4f}")
    print(f"Contrastivity for Saliency map on {dataset_name} dataset: {contrastivity_saliency_map:.4f}")
    sparsity_gradcam = evaluate_sparsity_gnn(model, graph_test_data, batch_grad_cam)
    sparsity_saliency_map = evaluate_sparsity_gnn(model, graph_test_data, batch_saliency_map)
    print(f"Sparsity for GradCAM on {dataset_name} dataset: {sparsity_gradcam:.4f}")
    print(f"Sparsity for Saliency map on {dataset_name} dataset: {sparsity_saliency_map:.4f}")


if __name__ == "__main__":
    graph_featurizer = GraphFeaturizer(y_col="y", log_target_transform=False)

    _, _, herg_test_data = load_herg_data_split()
    run_experiment(herg_test_data, "herg", herg_gnn_params)

    _, _, pampa_test = load_pampa_data_split()
    run_experiment(pampa_test, "pampa", pampa_gnn_params)

    _, _, cyp_test = load_cyp_data_split()
    run_experiment(cyp_test, "cyp", cyp_gnn_params)

    _, _, synthetic_test = load_synthetic_data_split()
    run_experiment(synthetic_test, "synthetic", synthetic_gnn_params)
