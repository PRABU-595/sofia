import numpy as np
from typing import Dict, List
from collections import defaultdict
from .agreement import compute_jaccard

class NoiseFloorEstimator:
    def __init__(self, runner, n_runs=10, noise_std=0.01):
        self.runner = runner
        self.n_runs = n_runs
        self.noise_std = noise_std
        
    def estimate(self, model, data, node_indices) -> Dict[int, float]:
        \"\"\"
        Runs the explainer multiple times with noise and computes pairwise Jaccard agreement
        to establish a baseline for each node.
        \"\"\"
        all_runs_explanations = []
        
        # We need at least 2 runs to compare, but n_runs usually >= 5
        for _ in range(self.n_runs):
            exps = self.runner.run_with_noise(model, data, node_indices, self.noise_std)
            all_runs_explanations.append(exps)
            
        # Compute pairwise agreement per node
        per_node_agreements = defaultdict(list)
        
        for i in range(self.n_runs):
            for j in range(i + 1, self.n_runs):
                exps_i = all_runs_explanations[i]
                exps_j = all_runs_explanations[j]
                
                for v in node_indices:
                    v_int = int(v)
                    if v_int in exps_i and v_int in exps_j:
                        jaccard = compute_jaccard(set(exps_i[v_int]), set(exps_j[v_int]))
                        per_node_agreements[v_int].append(jaccard)
                        
        mean_agreements = {}
        for v in per_node_agreements:
            if per_node_agreements[v]:
                mean_agreements[v] = float(np.mean(per_node_agreements[v]))
            else:
                mean_agreements[v] = 0.0
                
        return mean_agreements
