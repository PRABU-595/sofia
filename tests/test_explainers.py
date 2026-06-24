import torch
import pytest
import numpy as np
from torch_geometric.data import Data
from models.gnn_full import GNNFull
from explainers.explanation_runner import ExplanationRunner

@pytest.fixture
def dummy_data():
    x = torch.randn(10, 16)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 0],
        [1, 0, 2, 1, 4, 3, 6, 5, 5, 8, 7, 0, 9]
    ])
    y = torch.randint(0, 2, (10,))
    return Data(x=x, edge_index=edge_index, y=y)

@pytest.fixture
def dummy_model():
    return GNNFull(in_channels=16, hidden_channels=8, out_channels=2, seed=42)

def test_explainer_determinism(dummy_data, dummy_model):
    runner = ExplanationRunner({'epochs': 10, 'lr': 0.1, 'top_k': 3})
    
    # Run once
    torch.manual_seed(42)
    np.random.seed(42)
    exp1 = runner.run(dummy_model, dummy_data, [0, 1])
    
    # Run twice
    torch.manual_seed(42)
    np.random.seed(42)
    exp2 = runner.run(dummy_model, dummy_data, [0, 1])
    
    assert exp1[0] == exp2[0]
    assert exp1[1] == exp2[1]
