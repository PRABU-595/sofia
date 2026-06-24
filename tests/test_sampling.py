import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from sampling.exclusion_tracker import ExclusionTracker
import pandas as pd

def test_exclusion_tracker():
    # Graph: 0 -> 1, 2 -> 1, 3 -> 1, 4 -> 1
    # 4 neighbors pointing to 1
    edge_index = torch.tensor([
        [0, 2, 3, 4],
        [1, 1, 1, 1]
    ])
    data = Data(edge_index=edge_index, num_nodes=5)
    
    # Restrict fanout to 2. Node 1 has 4 true neighbors, so 2 should be excluded.
    loader = NeighborLoader(
        data,
        num_neighbors=[2],
        batch_size=1,
        input_nodes=torch.tensor([1])
    )
    
    tracker = ExclusionTracker(num_nodes=5)
    
    for batch in loader:
        tracker.record_batch(batch, data.edge_index)
        
    df_meta = tracker.get_dataframe(explanation_bundle={1: [0, 2, 3, 4]}, 
                                    degree_per_node=torch.tensor([0, 4, 0, 0, 0]), 
                                    homophily_per_node=torch.tensor([-1.0, 1.0, -1.0, -1.0, -1.0]))
    
    assert len(df_meta) == 1
    row = df_meta.iloc[0]
    
    # 2 out of 4 neighbors should be excluded
    assert row['exclusion_rate'] == 0.5
    # Since explanation has all 4, and 2 were excluded, score = 0.5
    assert row['influential_exclusion_score'] == 0.5
