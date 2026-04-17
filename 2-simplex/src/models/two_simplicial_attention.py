import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoSimplicialAttention(nn.Module):
    """
    Minimal, PyTorch 2.0 vanilla implementation of the 2-simplicial attention layer
    (paper: Fast and Simplex: 2-Simplicial Attention in Triton). This MVP handles
    a single graph (batch size = 1) and expects input as a set of triangulations
    with adjacency provided as an edge_index-like structure.
    The core equations implemented (simplified to MVP) are:
      - Q = X W_Q, K = X W_K, V = X W_V
      - K' = X W_Kp, V' = X W_Vp
      - A_ijk^(2s) = (1 / sqrt(d)) < q_i, k_j, k'_k >
      - S_ijk^(2s) = softmax_{j,k} (A_ijk^(2s))
      - v~_i^(2s) = sum_{j,k} S_ijk^(2s) ( v_j ∘ v'_k )
      - y_i = W_O concat_heads(v~_i^(2s)) with final projection
    Notes:
      - edge_index must be provided as a tensor of shape (N, max_deg) with -1 padding
        for non-existing entries. Each row i contains indices of neighbors j of triangolo i.
      - Core (j,k) loops are vectorized via torch.einsum for performance;
        the per-node loop is kept since each node has a variable number of neighbors.
    """

    def __init__(self, in_dim, out_dim=None, num_heads=4, dropout=0.0, with_residual=True, use_triton_kernel=False):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim) if out_dim is not None else self.in_dim
        self.num_heads = int(num_heads)
        assert self.out_dim % self.num_heads == 0, "out_dim must be divisible by num_heads"
        self.head_dim = self.out_dim // self.num_heads
        self.dropout = dropout
        self.with_residual = with_residual
        self.use_triton_kernel = bool(use_triton_kernel)

        # Projections
        self.q_proj = nn.Linear(self.in_dim, self.out_dim, bias=True)
        self.k_proj = nn.Linear(self.in_dim, self.out_dim, bias=True)
        self.v_proj = nn.Linear(self.in_dim, self.out_dim, bias=True)
        self.kp_proj = nn.Linear(self.in_dim, self.out_dim, bias=True)
        self.vp_proj = nn.Linear(self.in_dim, self.out_dim, bias=True)

        self.out_proj = nn.Linear(self.out_dim, self.out_dim, bias=True)
        self.norm = nn.LayerNorm(self.out_dim)
        self.drop = nn.Dropout(self.dropout)

    def forward(self, tri_feats, edge_index, batch=None):
        """
        tri_feats: Tensor of shape (N, in_dim) for MVP
        edge_index: LongTensor of shape (N, max_deg) with -1 padding, containing neighbor indices
        batch: not used in MVP
        Returns: Tensor of shape (N, out_dim)
        """
        if tri_feats.dim() != 2:
            raise ValueError("tri_feats must be (N, in_dim)")
        N, _ = tri_feats.shape
        Q = self.q_proj(tri_feats).view(N, self.num_heads, self.head_dim)  # (N, H, D)
        K = self.k_proj(tri_feats).view(N, self.num_heads, self.head_dim)
        V = self.v_proj(tri_feats).view(N, self.num_heads, self.head_dim)
        Kp = self.kp_proj(tri_feats).view(N, self.num_heads, self.head_dim)
        Vp = self.vp_proj(tri_feats).view(N, self.num_heads, self.head_dim)

        if edge_index is None:
            raise ValueError("edge_index must be provided for the MVP forward.")
        if not isinstance(edge_index, torch.Tensor):
            raise TypeError("edge_index must be a Tensor of shape (N, max_deg).")
        max_deg = edge_index.size(1)
        deg = (edge_index >= 0).sum(dim=1).to(dtype=torch.int64).cpu().tolist()

        Z_rows = []
        for i in range(N):
            d_i = int(deg[i])
            if d_i == 0:
                Z_rows.append(Q[i])
                continue

            neigh_i = edge_index[i, :d_i].to(tri_feats.device)
            K_j = K[neigh_i]    # (d_i, H, D)
            V_j = V[neigh_i]
            Kp_j = Kp[neigh_i]
            Vp_j = Vp[neigh_i]

            q_i = Q[i]  # (H, D)
            head_outs = []
            for h in range(self.num_heads):
                qi = q_i[h]  # (D,)
                kj = K_j[:, h, :]   # (d_i, D)
                kjp = Kp_j[:, h, :]
                vj = V_j[:, h, :]
                vjp = Vp_j[:, h, :]

                A_ijk = torch.einsum('d,jd,kd->jk', qi, kj, kjp) / (self.head_dim ** 0.5)

                S_flat = F.softmax(A_ijk.reshape(-1), dim=0)
                S = self.drop(S_flat.view(d_i, d_i))

                head_outs.append(torch.einsum('jk,jd,kd->d', S, vj, vjp))
            Z_rows.append(torch.stack(head_outs))

        Z = torch.stack(Z_rows)

        # Optional Triton path (disabled by default). Keeps API stable for future enablement.
        if self.use_triton_kernel:
            try:
                # Lazy import; tests/local envs may not have Triton kernel available yet
                from ..kernels.two_simplicial_autograd import TwoSimplicialAttentionFunction  # type: ignore
                y = TwoSimplicialAttentionFunction.apply(
                    tri_feats, edge_index, Q, K, V, Kp, Vp, self.out_dim, self.num_heads, self.head_dim
                )
                Z_concat = y.reshape(N, self.out_dim)
                out = self.out_proj(Z_concat)
                if self.with_residual and out.shape == tri_feats.shape:
                    out = out + tri_feats
                out = self.norm(out)
                return out
            except Exception:
                # Fall back to PyTorch MVP if Triton kernel is unavailable or fails to load
                pass

        Z_concat = Z.reshape(N, self.out_dim)
        out = self.out_proj(Z_concat)
        if self.with_residual and out.shape == tri_feats.shape:
            out = out + tri_feats
        out = self.norm(out)
        return out
