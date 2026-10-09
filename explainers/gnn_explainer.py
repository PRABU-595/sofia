import torch
from torch_geometric.explain import Explainer, GNNExplainer

def get_gnn_explainer(model, epochs=100, lr=0.01):
    """
    Returns a PyG Explainer configured with GNNExplainer algorithm.
    """
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs, lr=lr),
        explanation_type='model',
        node_mask_type='object',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
    )
    return explainer
