"""
Autograd Function wrapper for 2-simplicial attention.
Enables PyTorch autograd integration with the Triton kernels.
"""
import torch
from . import triton_2s_forward, triton_2s_backward


class TwoSimplicialAttentionFunction(torch.autograd.Function):
    """Custom autograd function for 2-simplicial attention."""
    
    @staticmethod
    def forward(ctx, tri_feats, edge_index, Q, K, V, Kp, Vp, out_dim, num_heads, head_dim, w1=8, w2=8):
        """Forward pass."""
        O, M = triton_2s_forward.forward(
            tri_feats, edge_index, Q, K, V, Kp, Vp, out_dim, num_heads, head_dim, w1, w2
        )
        
        # Save tensors for backward
        ctx.save_for_backward(Q, K, V, Kp, Vp, M, O, tri_feats)
        ctx.edge_index = edge_index
        ctx.out_dim = out_dim
        ctx.num_heads = num_heads
        ctx.head_dim = head_dim
        ctx.w1 = w1
        ctx.w2 = w2
        
        return O
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass."""
        Q, K, V, Kp, Vp, M, O, tri_feats = ctx.saved_tensors
        
        # The backward kernel expects layout [N, H, D] or similar
        # Our triton_2s_backward.backward returns (dQ, dK, dV, dKp, dVp)
        dQ, dK, dV, dKp, dVp = triton_2s_backward.backward(
            grad_output,
            tri_feats,
            ctx.edge_index,
            Q, K, V, Kp, Vp,
            ctx.out_dim, ctx.num_heads, ctx.head_dim,
            ctx.w1, ctx.w2
        )
        
        # We need to return gradients for all 12 arguments of forward:
        # tri_feats, edge_index, Q, K, V, Kp, Vp, out_dim, num_heads, head_dim, w1, w2
        # dQ, dK, etc. are already [N, H, D], but nn.Linear expects flattened or matched shape if it's the result of view.
        # However, autograd will handle the view/reshape if we return the correctly shaped tensor.
        
        return (
            None, # tri_feats
            None, # edge_index
            dQ,
            dK,
            dV,
            dKp,
            dVp,
            None, # out_dim
            None, # num_heads
            None, # head_dim
            None, # w1
            None  # w2
        )
