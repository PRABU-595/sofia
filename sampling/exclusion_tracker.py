import torch
import pandas as pd
from collections import defaultdict
from torch_geometric.utils import degree

class ExclusionTracker:
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        
        # excluded_neighbors[v] = set of neighbors excluded across all batches where v was sampled
        self.excluded_neighbors = defaultdict(set)
        
        # To compute exclusion rate: 
        # how many times was a node present as a target in a batch?
        self.target_appearances = torch.zeros(num_nodes, dtype=torch.long)
        self.total_neighbors_possible = torch.zeros(num_nodes, dtype=torch.long)
        self.total_neighbors_sampled = torch.zeros(num_nodes, dtype=torch.long)
        
        # We need a quick lookup for full graph neighbors
        self.full_adj = None
        
    def _build_adj(self, full_edge_index):
        if self.full_adj is None:
            self.full_adj = defaultdict(set)
            row, col = full_edge_index.cpu().numpy()
            for r, c in zip(row, col):
                # r is source, c is target (message passes r -> c)
                self.full_adj[c].add(r)
                
    def record_batch(self, batch, full_edge_index):
        self._build_adj(full_edge_index)
        
        # batch.n_id maps local batch index to global node index
        n_id = batch.n_id.cpu().numpy()
        
        # Sampled edges in this batch (local indices)
        batch_row, batch_col = batch.edge_index.cpu().numpy()
        
        # Map to global indices
        global_row = n_id[batch_row]
        global_col = n_id[batch_col]
        
        # Which neighbors were sampled for each target node in this batch?
        sampled_adj = defaultdict(set)
        for r, c in zip(global_row, global_col):
            sampled_adj[c].add(r)
            
        # For every target node in the batch (these are typically all nodes in n_id
        # that have incoming edges, but we can just iterate over unique global_cols)
        unique_targets = set(global_col)
        
        for v in unique_targets:
            self.target_appearances[v] += 1
            
            true_neighbors = self.full_adj[v]
            sampled_neighbors = sampled_adj[v]
            
            excluded = true_neighbors - sampled_neighbors
            
            self.excluded_neighbors[v].update(excluded)
            self.total_neighbors_possible[v] += len(true_neighbors)
            self.total_neighbors_sampled[v] += len(sampled_neighbors)
            
    def compute_exclusion_rate(self):
        # fraction of neighbors excluded across all batches
        # = (possible - sampled) / possible
        rate = torch.zeros(self.num_nodes, dtype=torch.float)
        mask = self.total_neighbors_possible > 0
        rate[mask] = (self.total_neighbors_possible[mask] - self.total_neighbors_sampled[mask]).float() / self.total_neighbors_possible[mask].float()
        return rate
        
    def get_influential_exclusion_score(self, node_id: int, top_k_explanation: set) -> float:
        \"\"\"
        Given a node and its post-hoc explanation (set of top-k influential neighbor IDs),
        what fraction of those top-k neighbors were in the excluded set during training?
        \"\"\"
        if len(top_k_explanation) == 0:
            return 0.0
            
        excluded = self.excluded_neighbors.get(node_id, set())
        overlap = excluded.intersection(top_k_explanation)
        return len(overlap) / len(top_k_explanation)
        
    def get_dataframe(self, explanation_bundle, degree_per_node, homophily_per_node):
        \"\"\"
        Builds the final dataframe for analysis.
        explanation_bundle: dict mapping node_id -> top_k_neighbor_indices (set or list)
        \"\"\"
        exclusion_rates = self.compute_exclusion_rate().numpy()
        degree_per_node = degree_per_node.numpy()
        homophily_per_node = homophily_per_node.numpy()
        
        records = []
        for v in range(self.num_nodes):
            # Only consider nodes that were actually targeted during training (e.g., connected to train set)
            # and have an explanation
            if self.target_appearances[v] > 0 and v in explanation_bundle:
                top_k = set(explanation_bundle[v])
                inf_score = self.get_influential_exclusion_score(v, top_k)
                
                records.append({
                    'node_id': v,
                    'degree': degree_per_node[v],
                    'exclusion_rate': exclusion_rates[v],
                    'influential_exclusion_score': inf_score,
                    'homophily': homophily_per_node[v]
                })
                
        return pd.DataFrame(records)
