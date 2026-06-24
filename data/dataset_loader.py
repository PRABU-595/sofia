import torch
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor
from torch_geometric.utils import degree
from dataclasses import dataclass
import numpy as np
import os

@dataclass
class DatasetBundle:
    data: torch.Any
    homophily_per_node: torch.Tensor
    degree_per_node: torch.Tensor
    name: str

class DatasetLoader:
    def __init__(self, root: str = "./data"):
        self.root = root

    def load(self, name: str) -> DatasetBundle:
        name = name.lower()
        if name in ["cora", "citeseer"]:
            dataset = Planetoid(root=os.path.join(self.root, "Planetoid"), name=name)
            data = dataset[0]
        elif name == "chameleon":
            dataset = WikipediaNetwork(root=os.path.join(self.root, "WikipediaNetwork"), name="chameleon", geom_gcn_preprocess=True)
            data = dataset[0]
            # Wiki networks can have multiple masks, use the first one if present
            if data.train_mask.dim() > 1:
                data.train_mask = data.train_mask[:, 0]
                data.val_mask = data.val_mask[:, 0]
                data.test_mask = data.test_mask[:, 0]
        elif name == "actor":
            dataset = Actor(root=os.path.join(self.root, "Actor"))
            data = dataset[0]
            # Actor similarly can have multiple masks
            if data.train_mask.dim() > 1:
                data.train_mask = data.train_mask[:, 0]
                data.val_mask = data.val_mask[:, 0]
                data.test_mask = data.test_mask[:, 0]
        else:
            raise ValueError(f"Unknown dataset: {name}")

        homophily_per_node = self._compute_node_homophily(data)
        degree_per_node = self._compute_node_degree(data)

        return DatasetBundle(
            data=data,
            homophily_per_node=homophily_per_node,
            degree_per_node=degree_per_node,
            name=name
        )

    def _compute_node_homophily(self, data) -> torch.Tensor:
        row, col = data.edge_index
        labels = data.y
        
        homophily = torch.zeros(data.num_nodes, dtype=torch.float32)
        node_degrees = degree(row, num_nodes=data.num_nodes)
        
        for i in range(data.num_nodes):
            if node_degrees[i] > 0:
                neighbors = col[row == i]
                same_label_count = (labels[neighbors] == labels[i]).sum().item()
                homophily[i] = same_label_count / node_degrees[i].item()
            else:
                homophily[i] = -1.0  # isolated nodes
                
        return homophily

    def _compute_node_degree(self, data) -> torch.Tensor:
        row, col = data.edge_index
        return degree(row, num_nodes=data.num_nodes)

    @staticmethod
    def get_homophily_bins(homophily_per_node: torch.Tensor, n_bins: int = 5) -> dict:
        \"\"\"
        Returns node indices stratified by homophily score.
        Ignores isolated nodes (homophily == -1.0).
        \"\"\"
        valid_nodes = (homophily_per_node >= 0).nonzero(as_tuple=True)[0]
        valid_homophily = homophily_per_node[valid_nodes]
        
        bins = np.linspace(0, 1, n_bins + 1)
        # Using digitize: values 1 to n_bins
        bin_indices = np.digitize(valid_homophily.numpy(), bins)
        # Fix for values exactly equal to 1.0
        bin_indices[bin_indices == n_bins + 1] = n_bins
        
        strata = {}
        for b in range(1, n_bins + 1):
            lower = bins[b-1]
            upper = bins[b]
            stratum_name = f"{lower:.1f}-{upper:.1f}"
            nodes_in_bin = valid_nodes[bin_indices == b]
            strata[stratum_name] = nodes_in_bin
            
        return strata
