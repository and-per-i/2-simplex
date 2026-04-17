import torch
import pytest

from src.models import TwoSimplicialAttention


def make_dummy_graph(N=8, in_dim=32, out_dim=64):
    tri_feats = torch.randn(N, in_dim)
    max_deg = 3
    edge_index = torch.full((N, max_deg), -1, dtype=torch.long)
    for i in range(N - 1):
        edge_index[i, 0] = i + 1
        edge_index[i + 1 if i + 1 < N else i, 0] = i
    edge_index = edge_index.clamp(min=-1)
    return tri_feats, edge_index


def test_forward_shape_basic():
    N = 8
    in_dim = 32
    out_dim = 64
    model = TwoSimplicialAttention(in_dim, out_dim, num_heads=4, dropout=0.0, with_residual=True, use_triton_kernel=False)
    tri_feats, edge_index = make_dummy_graph(N=N, in_dim=in_dim, out_dim=out_dim)
    out = model(tri_feats, edge_index)
    assert out.shape == (N, out_dim)


def test_forward_with_no_neighbors():
    N = 6
    in_dim = 32
    out_dim = 64
    model = TwoSimplicialAttention(in_dim, out_dim, num_heads=4, dropout=0.0, with_residual=True, use_triton_kernel=False)
    tri_feats = torch.randn(N, in_dim)
    edge_index = torch.full((N, 3), -1, dtype=torch.long)
    out = model(tri_feats, edge_index)
    assert out.shape == (N, out_dim)


def test_forward_shape_in_dim_equals_out_dim():
    N = 6
    dim = 16
    model = TwoSimplicialAttention(dim, num_heads=4, dropout=0.0, with_residual=True, use_triton_kernel=False)
    model.eval()
    tri_feats = torch.randn(N, dim)
    edge_index = torch.tensor([
        [1, 2, -1],
        [0, 3, -1],
        [0, 1, -1],
        [1, -1, -1],
        [-1, -1, -1],
        [-1, -1, -1],
    ], dtype=torch.long)
    out = model(tri_feats, edge_index)
    assert out.shape == (N, dim)


def test_forward_shape_no_residual():
    N = 5
    in_dim = 16
    out_dim = 32
    model = TwoSimplicialAttention(in_dim, out_dim, num_heads=4, dropout=0.0, with_residual=False, use_triton_kernel=False)
    model.eval()
    tri_feats = torch.randn(N, in_dim)
    edge_index = torch.tensor([
        [1, -1],
        [0, 2],
        [1, 3],
        [2, 4],
        [3, -1],
    ], dtype=torch.long)
    out = model(tri_feats, edge_index)
    assert out.shape == (N, out_dim)


def test_forward_shape_single_head():
    N = 4
    in_dim = 16
    out_dim = 16
    model = TwoSimplicialAttention(in_dim, out_dim, num_heads=1, dropout=0.0, with_residual=True, use_triton_kernel=False)
    model.eval()
    tri_feats = torch.randn(N, in_dim)
    edge_index = torch.tensor([
        [1, -1],
        [0, 2],
        [1, 3],
        [2, -1],
    ], dtype=torch.long)
    out = model(tri_feats, edge_index)
    assert out.shape == (N, out_dim)


def test_forward_shape_many_heads():
    N = 4
    in_dim = 16
    out_dim = 32
    model = TwoSimplicialAttention(in_dim, out_dim, num_heads=8, dropout=0.0, with_residual=True, use_triton_kernel=False)
    model.eval()
    tri_feats = torch.randn(N, in_dim)
    edge_index = torch.tensor([
        [1, -1],
        [0, 2],
        [1, 3],
        [2, -1],
    ], dtype=torch.long)
    out = model(tri_feats, edge_index)
    assert out.shape == (N, out_dim)
