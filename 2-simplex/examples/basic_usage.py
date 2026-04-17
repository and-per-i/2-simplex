"""
Basic usage example for TwoSimplicialAttention.
"""

import torch
from src.models.two_simplicial_attention import TwoSimplicialAttention


def main():
    # Create model
    model = TwoSimplicialAttention(
        in_dim=32,
        out_dim=64,
        num_heads=4,
        dropout=0.1,
        with_residual=True,
        use_triton_kernel=False,
    )

    # Dummy input
    n_triangles = 128
    in_dim = 32
    max_neighbors = 8

    tri_feats = torch.randn(n_triangles, in_dim)
    edge_index = torch.full((n_triangles, max_neighbors), -1, dtype=torch.long)

    for i in range(n_triangles):
        n_neigh = min(max_neighbors, max(1, n_triangles // 8))
        neighbors = torch.randperm(n_triangles)[:n_neigh]
        neighbors = neighbors[neighbors != i]
        edge_index[i, : len(neighbors)] = neighbors

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(tri_feats, edge_index)

    print(f"Input shape:  {tri_feats.shape}")
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()
