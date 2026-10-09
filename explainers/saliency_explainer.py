import torch
from torch_geometric.explain import Explainer, CaptumExplainer

def get_saliency_explainer(model):
    """
    Returns a PyG Explainer configured with Saliency (Captum) algorithm.
    This provides a fast, robust gradient-based proxy for neighbor influence.
    """
    explainer = Explainer(
        model=model,
        algorithm=CaptumExplainer('Saliency'),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
    )
    return explainer
