import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from typing import Optional

def train_full(model, data, optimizer, epochs=200):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    return model

def train_sampled(model, data, optimizer, fanout=[10, 10], batch_size=512, epochs=200, exclusion_tracker=None):
    model.train()
    
    # We create a custom wrapper for NeighborLoader if we need to track exclusions
    loader = NeighborLoader(
        data,
        num_neighbors=fanout,
        batch_size=batch_size,
        input_nodes=data.train_mask,
        shuffle=True
    )
    
    for epoch in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            
            # The tracker records which neighbors were missing in the sampled batch
            if exclusion_tracker is not None:
                exclusion_tracker.record_batch(batch, data.edge_index)
                
            out = model(batch.x, batch.edge_index)
            # The first `batch_size` nodes in the batch are the seed nodes
            batch_size_curr = batch.batch_size
            loss = F.cross_entropy(out[:batch_size_curr], batch.y[:batch_size_curr])
            loss.backward()
            optimizer.step()
            
    return model

def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = (pred[mask] == data.y[mask]).sum()
        acc = int(correct) / int(mask.sum())
    return acc
