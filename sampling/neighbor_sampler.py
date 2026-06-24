from torch_geometric.loader import NeighborLoader

def create_neighbor_loader(data, fanout=[10, 10], batch_size=512, shuffle=True, num_workers=0):
    \"\"\"
    Helper to create PyG NeighborLoader.
    \"\"\"
    return NeighborLoader(
        data,
        num_neighbors=fanout,
        batch_size=batch_size,
        input_nodes=data.train_mask,
        shuffle=shuffle,
        num_workers=num_workers
    )
