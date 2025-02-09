from src.utils import min_max_scale_graph_batch


def calculate_batch_explanation_contrastivity(
        explanation_batch,
        explanation_batch_inv,
        node_assignments,
        threshold=0.1
):
    exp_batch_scaled = min_max_scale_graph_batch(explanation_batch, node_assignments)
    exp_batch_inv_scaled = min_max_scale_graph_batch(explanation_batch_inv, node_assignments)
    exp_or = ((exp_batch_scaled > threshold) | (exp_batch_inv_scaled > threshold))
    total_nodes_found = exp_or.sum().item()
    exp_xor = (((exp_batch_scaled > threshold) & (exp_batch_inv_scaled <= threshold))
               | ((exp_batch_scaled <= threshold) & (exp_batch_inv_scaled > threshold)))
    return exp_xor.sum().item() / total_nodes_found


def calculate_batch_explanation_sparsity(
        explanation_batch,
        node_assignments,
        threshold=0.1
):
    exp_batch_scaled = min_max_scale_graph_batch(explanation_batch, node_assignments)
    return 1 - ((exp_batch_scaled > threshold).sum() / exp_batch_scaled.numel()).item()
