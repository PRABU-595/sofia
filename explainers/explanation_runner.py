import torch
from tqdm import tqdm
from .gnn_explainer import get_gnn_explainer
from .saliency_explainer import get_saliency_explainer

class ExplanationRunner:
    def __init__(self, explainer_config):
        self.epochs = explainer_config.get('epochs', 100)
        self.lr = explainer_config.get('lr', 0.01)
        self.top_k = explainer_config.get('top_k', 10)
       
        self.type = explainer_config.get('type', 'gnnexplainer')
       
    def _run_explainer(self, model, data, node_indices, desc="Explaining"):
        if self.type == 'saliency':
            explainer = get_saliency_explainer(model)
        elif self.type == 'gnnexplainer':
            explainer = get_gnn_explainer(model, epochs=self.epochs, lr=self.lr)
        else:
            raise ValueError(f"Unknown explainer type: '{self.type}'. Use 'saliency' or 'gnnexplainer'.")
       
        explanation_bundle = {}
        model.eval()
       
        class ModelWrapper(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
            def forward(self, x, edge_index):
                return torch.log_softmax(self.base_model(x, edge_index), dim=-1)
               
        wrapped_model = ModelWrapper(model)
        explainer.model = wrapped_model
       
        from torch_geometric.loader import NeighborLoader
       
        # Cap the explainer subgraph to [100, 100] to prevent OOM on super-hubs
        loader = NeighborLoader(
            data,
            num_neighbors=[100, 100],
            batch_size=1,
            input_nodes=torch.tensor(node_indices, dtype=torch.long),
            shuffle=False
        )
       
        device = next(model.parameters()).device
       
        for batch in tqdm(loader, desc=desc, leave=False):
            batch = batch.to(device)
            node_id_int = batch.n_id[0].item()
           
            explanation = explainer(batch.x, batch.edge_index, index=0)
            edge_mask = explanation.edge_mask
           
            row, col = batch.edge_index
            top_k_edges = torch.topk(edge_mask, min(self.top_k, len(edge_mask))).indices
           
            top_k_neighbors = set()
            for edge_idx in top_k_edges:
                src_relabeled = row[edge_idx].item()
                dst_relabeled = col[edge_idx].item()
                src_orig = batch.n_id[src_relabeled].item()
                dst_orig = batch.n_id[dst_relabeled].item()
               
                if src_orig != node_id_int:
                    top_k_neighbors.add(src_orig)
                elif dst_orig != node_id_int:
                    top_k_neighbors.add(dst_orig)
                   
            explanation_bundle[node_id_int] = list(top_k_neighbors)[:self.top_k]
           
        return explanation_bundle

    def run(self, model, data, node_indices, desc="Explaining (Clean)"):
        return self._run_explainer(model, data, node_indices, desc=desc)

    def run_with_noise(self, model, data, node_indices, noise_std=0.01):
        noisy_x = data.x + torch.randn_like(data.x) * noise_std
        import copy
        noisy_data = copy.copy(data)
        noisy_data.x = noisy_x
        return self._run_explainer(model, noisy_data, node_indices, desc="Explaining (Noisy)")
