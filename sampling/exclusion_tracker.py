import torch
import pandas as pd
from collections import defaultdict
from torch_geometric.utils import degree
def _create_inner_dict():
    return defaultdict(int)

class ExclusionTracker:
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.neighbor_sampled_count = defaultdict(_create_inner_dict)
        self.target_appearances = torch.zeros(num_nodes, dtype=torch.long)
        self.total_neighbors_sampled = torch.zeros(num_nodes, dtype=torch.long)
        self.full_adj = None
        
    def _build_adj(self, full_edge_index):
        if self.full_adj is None:
            self.full_adj = defaultdict(set)
            row, col = full_edge_index.cpu().numpy()
            for r, c in zip(row, col):
                self.full_adj[c].add(r)
                
    def record_batch(self, batch, full_edge_index):
        self._build_adj(full_edge_index)
        n_id = batch.n_id.cpu().numpy()
        batch_row, batch_col = batch.edge_index.cpu().numpy()
        
        global_row = n_id[batch_row]
        global_col = n_id[batch_col]
        
        sampled_adj = defaultdict(set)
        for r, c in zip(global_row, global_col):
            sampled_adj[c].add(r)
            
        unique_targets = set(global_col)
        
        for v in unique_targets:
            self.target_appearances[v] += 1
            sampled_neighbors = sampled_adj[v]
            self.total_neighbors_sampled[v] += len(sampled_neighbors)
            for u in sampled_neighbors:
                self.neighbor_sampled_count[v][u] += 1
            
    def compute_exclusion_rate(self):
        rate = torch.zeros(self.num_nodes, dtype=torch.float)
        possible = torch.zeros(self.num_nodes, dtype=torch.long)
        for v in range(self.num_nodes):
            if self.target_appearances[v] > 0:
                possible[v] = self.target_appearances[v] * len(self.full_adj[v])
                
        mask = possible > 0
        rate[mask] = (possible[mask] - self.total_neighbors_sampled[mask]).float() / possible[mask].float()
        return rate
        
    def get_influential_exclusion_score(self, node_id: int, top_k_explanation: set) -> float:
        if len(top_k_explanation) == 0 or self.target_appearances[node_id] == 0:
            return 0.0
            
        total_exclusion = 0.0
        apps = float(self.target_appearances[node_id])
        for u in top_k_explanation:
            sampled_count = self.neighbor_sampled_count[node_id].get(u, 0)
            exclusion = 1.0 - (sampled_count / apps)
            total_exclusion += exclusion
            
        return total_exclusion / len(top_k_explanation)
        
    def get_dataframe(self, explanation_bundle, degree_per_node, homophily_per_node):
        exclusion_rates = self.compute_exclusion_rate().numpy()
        degree_per_node = degree_per_node.numpy()
        homophily_per_node = homophily_per_node.numpy()
        
        records = []
        for v in range(self.num_nodes):
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