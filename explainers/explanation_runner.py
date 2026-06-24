import torch
from tqdm import tqdm
from .gnn_explainer import get_gnn_explainer

class ExplanationRunner:
    def __init__(self, explainer_config):
        self.epochs = explainer_config.get('epochs', 100)
        self.lr = explainer_config.get('lr', 0.01)
        self.top_k = explainer_config.get('top_k', 10)
        
    def _run_explainer(self, model, data, node_indices, desc="Explaining"):
        explainer = get_gnn_explainer(model, epochs=self.epochs, lr=self.lr)
        
        explanation_bundle = {}
        # Ensure model is in eval mode and edges are correct
        model.eval()
        
        # We need a wrapper to ensure output matches expected log_probs
        class ModelWrapper(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
            def forward(self, x, edge_index):
                return torch.log_softmax(self.base_model(x, edge_index), dim=-1)
                
        wrapped_model = ModelWrapper(model)
        explainer.model = wrapped_model
        
        for node_id in tqdm(node_indices, desc=desc, leave=False):
            node_id_int = int(node_id)
            # PyG explanation format
            explanation = explainer(data.x, data.edge_index, index=node_id_int)
            
            # Get edge mask
            edge_mask = explanation.edge_mask
            
            # Find incoming edges to the target node
            # The explainer often extracts a subgraph.
            # In PyG 2.4+, the edge_mask maps directly to data.edge_index
            row, col = data.edge_index
            
            # Which edges point to the target node? (Actually we want any edges influential to the node)
            # Usually GNNExplainer returns importance for the k-hop subgraph.
            # We want the top-k most important neighbors.
            # Neighbors are sources of edges pointing to node_id, or generally just nodes connected.
            
            # To simplify and ensure we only get actual neighbors in the 1-hop or k-hop:
            # We just take the edges with the highest edge_mask values
            top_k_edges = torch.topk(edge_mask, min(self.top_k, len(edge_mask))).indices
            
            # We extract the source nodes of these highly ranked edges
            top_k_neighbors = set()
            for edge_idx in top_k_edges:
                src = row[edge_idx].item()
                dst = col[edge_idx].item()
                # We usually care about nodes that send messages (src). 
                # If src is the node itself (self-loop), we can optionally exclude it.
                if src != node_id_int:
                    top_k_neighbors.add(src)
                elif dst != node_id_int:
                    top_k_neighbors.add(dst)
                    
            # Ensure it's a list up to top_k
            explanation_bundle[node_id_int] = list(top_k_neighbors)[:self.top_k]
            
        return explanation_bundle

    def run(self, model, data, node_indices):
        \"\"\"
        Runs the explainer for every node in node_indices.
        Returns a dict: node_id -> list of top_k_neighbor_indices
        \"\"\"
        return self._run_explainer(model, data, node_indices, desc="Explaining (Clean)")

    def run_with_noise(self, model, data, node_indices, noise_std=0.01):
        \"\"\"
        Perturbs node features slightly and re-explains the same model.
        \"\"\"
        noisy_x = data.x + torch.randn_like(data.x) * noise_std
        
        # Create a shallow copy of data with noisy features
        import copy
        noisy_data = copy.copy(data)
        noisy_data.x = noisy_x
        
        return self._run_explainer(model, noisy_data, node_indices, desc="Explaining (Noisy)")
