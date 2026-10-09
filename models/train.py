import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from typing import Optional

def train_full(model, data, optimizer, epochs=200):
    model.train()
    loader = NeighborLoader(
        data,
        num_neighbors=[25, 10],  # Standard GraphSAGE full baseline
        batch_size=512,
        input_nodes=data.train_mask,
        shuffle=True, num_workers=0
    )
    device = next(model.parameters()).device
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"GNNFull Epoch {epoch+1}/{epochs}", leave=False)
        for batch in pbar:
            optimizer.zero_grad()
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            bs = batch.batch_size
            loss = F.cross_entropy(out[:bs], batch.y[:bs])
            loss.backward()
            optimizer.step()
    return model

def train_sampled(model, data, optimizer, fanout=[10, 10], batch_size=512, epochs=200, exclusion_tracker=None):
    model.train()
    loader = NeighborLoader(
        data,
        num_neighbors=fanout,
        batch_size=batch_size,
        input_nodes=data.train_mask,
        shuffle=True, num_workers=0
    )
    device = next(model.parameters()).device
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"GNNSampled Epoch {epoch+1}/{epochs}", leave=False)
        for batch in pbar:
            optimizer.zero_grad()
            if exclusion_tracker is not None:
                exclusion_tracker.record_batch(batch, data.edge_index)
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            bs = batch.batch_size
            loss = F.cross_entropy(out[:bs], batch.y[:bs])
            loss.backward()
            optimizer.step()
    return model

def evaluate(model, data, mask):
    model.eval()
    loader = NeighborLoader(
        data,
        num_neighbors=[25, 10],
        batch_size=512,
        input_nodes=mask,
        shuffle=False
    )
    correct = 0
    total = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            pred = out[:batch.batch_size].argmax(dim=1)
            correct += int((pred == batch.y[:batch.batch_size]).sum())
            total += batch.batch_size
    return correct / total if total > 0 else 0
