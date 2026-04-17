"""
Tests for the Triton backward kernels (2-simplicial attention).

These tests require Triton + CUDA GPU. They will be automatically skipped
in local environments without GPU support. Run on a cloud GPU instance.

Note: The backward wrapper (backward()) currently raises NotImplementedError.
These tests call the Triton kernels directly via the lower-level API,
adapting the layout to [batch, seq_len, num_heads, head_dim] in bfloat16.
"""
import torch
import pytest

from tests.triton.conftest import skip_no_triton


def _make_qkv(B, S, num_heads, head_dim, device, dtype=torch.bfloat16, seed=42):
    torch.manual_seed(seed)
    Q = torch.randn(B, S, num_heads, head_dim, device=device, dtype=dtype)
    K1 = torch.randn(B, S, num_heads, head_dim, device=device, dtype=dtype)
    K2 = torch.randn(B, S, num_heads, head_dim, device=device, dtype=dtype)
    V1 = torch.randn(B, S, num_heads, head_dim, device=device, dtype=dtype)
    V2 = torch.randn(B, S, num_heads, head_dim, device=device, dtype=dtype)
    return Q, K1, K2, V1, V2


def _call_fwd_kernel(Q, K1, K2, V1, V2, w1, w2):
    from src.kernels.triton_2s_forward import _forward_kernel_call, _check_triton
    assert _check_triton()
    B, S, num_heads, head_dim = Q.shape
    O = torch.zeros_like(Q)
    M = torch.zeros(B, num_heads, S, dtype=torch.float32, device=Q.device)
    _forward_kernel_call(
        Q, K1, K2, V1, V2, O, M,
        B, S, num_heads, head_dim, w1, w2,
        *Q.stride(), *K1.stride(), *K2.stride(), *V1.stride(), *V2.stride(),
        *O.stride(), *M.stride(),
    )
    return O, M


def _pytorch_fwd(Q, K1, K2, V1, V2, w1, w2, sm_scale):
    B, S, H, D = Q.shape
    O = torch.zeros(B, S, H, D, dtype=torch.float32, device=Q.device)
    M = torch.full((B, H, S), float("-inf"), dtype=torch.float32, device=Q.device)

    for b in range(B):
        for h in range(H):
            for i in range(S):
                q_i = Q[b, i, h].float()
                max_s = float("-inf")
                weighted_sum = torch.zeros(D, dtype=torch.float32, device=Q.device)
                denom = 0.0
                for j in range(S):
                    if not (i - w1 < j <= i):
                        continue
                    for k in range(S):
                        if not (i - w2 < k <= i):
                            continue
                        s = torch.dot(q_i, K1[b, j, h].float() * K2[b, k, h].float()) * sm_scale
                        if s.item() > max_s:
                            max_s = s.item()
                for j in range(S):
                    if not (i - w1 < j <= i):
                        continue
                    for k in range(S):
                        if not (i - w2 < k <= i):
                            continue
                        s = torch.dot(q_i, K1[b, j, h].float() * K2[b, k, h].float()) * sm_scale
                        e = torch.exp(s - max_s)
                        denom += e.item()
                        weighted_sum += e * (V1[b, j, h].float() * V2[b, k, h].float())
                if denom > 0:
                    O[b, i, h] = weighted_sum / denom
                    M[b, h, i] = max_s + torch.log(torch.tensor(denom, dtype=torch.float32, device=Q.device))
    return O, M


def _pytorch_bwd(dO, Q, K1, K2, V1, V2, O, M, w1, w2, sm_scale):
    B, S, H, D = Q.shape
    dQ = torch.zeros_like(Q, dtype=torch.float32)
    dK1 = torch.zeros_like(K1, dtype=torch.float32)
    dK2 = torch.zeros_like(K2, dtype=torch.float32)
    dV1 = torch.zeros_like(V1, dtype=torch.float32)
    dV2 = torch.zeros_like(V2, dtype=torch.float32)

    for b in range(B):
        for h in range(H):
            for i in range(S):
                q_i = Q[b, i, h].float()
                dO_i = dO[b, i, h].float()
                Di = torch.dot(dO_i, O[b, i, h].float())
                for j in range(S):
                    if not (i - w1 < j <= i):
                        continue
                    for k in range(S):
                        if not (i - w2 < k <= i):
                            continue
                        s = torch.dot(q_i, K1[b, j, h].float() * K2[b, k, h].float()) * sm_scale
                        p = torch.exp(s - M[b, h, i])
                        v1v2 = V1[b, j, h].float() * V2[b, k, h].float()
                        dp = torch.dot(dO_i, v1v2)
                        ds = p * (dp - Di)

                        dQ[b, i, h] += ds * K1[b, j, h].float() * K2[b, k, h].float() * sm_scale
                        dK1[b, j, h] += ds * q_i * K2[b, k, h].float() * sm_scale
                        dK2[b, k, h] += ds * q_i * K1[b, j, h].float() * sm_scale
                        dV1[b, j, h] += p * dO_i * V2[b, k, h].float()
                        dV2[b, k, h] += p * dO_i * V1[b, j, h].float()

    return dQ, dK1, dK2, dV1, dV2


@skip_no_triton
class TestTritonBackwardKernel:
    def test_backward_output_shapes(self, cuda_device):
        B, S, H, D = 1, 64, 2, 64
        w1, w2 = S, S
        Q, K1, K2, V1, V2 = _make_qkv(B, S, H, D, cuda_device, seed=0)
        Q.requires_grad_(True)
        K1.requires_grad_(True)
        K2.requires_grad_(True)
        V1.requires_grad_(True)
        V2.requires_grad_(True)

        O, M = _call_fwd_kernel(Q, K1, K2, V1, V2, w1, w2)
        dO = torch.randn_like(O)
        O.backward(dO)

        assert Q.grad is not None
        assert K1.grad is not None
        assert K2.grad is not None
        assert V1.grad is not None
        assert V2.grad is not None
        assert Q.grad.shape == Q.shape
        assert K1.grad.shape == K1.shape
        assert K2.grad.shape == K2.shape
        assert V1.grad.shape == V1.shape
        assert V2.grad.shape == V2.shape

    def test_backward_grads_finite(self, cuda_device):
        B, S, H, D = 1, 64, 2, 64
        w1, w2 = S, S
        Q, K1, K2, V1, V2 = _make_qkv(B, S, H, D, cuda_device, seed=1)
        Q.requires_grad_(True)
        K1.requires_grad_(True)
        K2.requires_grad_(True)
        V1.requires_grad_(True)
        V2.requires_grad_(True)

        O, M = _call_fwd_kernel(Q, K1, K2, V1, V2, w1, w2)
        dO = torch.randn_like(O)
        O.backward(dO)

        for name, tensor in [("Q", Q), ("K1", K1), ("K2", K2), ("V1", V1), ("V2", V2)]:
            assert torch.isfinite(tensor.grad.float()).all(), f"Non-finite gradient for {name}"

    def test_backward_matches_pytorch_small(self, cuda_device):
        B, S, H, D = 1, 8, 1, 32
        w1, w2 = S, S
        sm_scale = 1.0 / (D ** 0.5)
        Q, K1, K2, V1, V2 = _make_qkv(B, S, H, D, cuda_device, seed=2)

        O_triton, M_triton = _call_fwd_kernel(Q, K1, K2, V1, V2, w1, w2)
        O_ref, M_ref = _pytorch_fwd(Q, K1, K2, V1, V2, w1, w2, sm_scale)

        dO = torch.randn_like(O_triton)

        dQ_ref, dK1_ref, dK2_ref, dV1_ref, dV2_ref = _pytorch_bwd(
            dO, Q, K1, K2, V1, V2, O_ref, M_ref, w1, w2, sm_scale
        )

        Q2 = Q.clone().detach().requires_grad_(True)
        K1_2 = K1.clone().detach().requires_grad_(True)
        K2_2 = K2.clone().detach().requires_grad_(True)
        V1_2 = V1.clone().detach().requires_grad_(True)
        V2_2 = V2.clone().detach().requires_grad_(True)
        O2, M2 = _call_fwd_kernel(Q2, K1_2, K2_2, V1_2, V2_2, w1, w2)
        O2.backward(dO)

        torch.testing.assert_close(Q2.grad.float(), dQ_ref.float(), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(K1_2.grad.float(), dK1_ref.float(), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(V1_2.grad.float(), dV1_ref.float(), atol=5e-2, rtol=5e-2)

    def test_backward_local_window(self, cuda_device):
        B, S, H, D = 1, 16, 1, 32
        w1, w2 = 4, 8
        sm_scale = 1.0 / (D ** 0.5)
        Q, K1, K2, V1, V2 = _make_qkv(B, S, H, D, cuda_device, seed=3)

        Q = Q.clone().detach().requires_grad_(True)
        K1 = K1.clone().detach().requires_grad_(True)
        K2 = K2.clone().detach().requires_grad_(True)
        V1 = V1.clone().detach().requires_grad_(True)
        V2 = V2.clone().detach().requires_grad_(True)

        O, M = _call_fwd_kernel(Q, K1, K2, V1, V2, w1, w2)
        dO = torch.randn_like(O)
        O.backward(dO)

        for name, t in [("Q", Q), ("K1", K1), ("K2", K2), ("V1", V1), ("V2", V2)]:
            assert torch.isfinite(t.grad.float()).all(), f"Non-finite grad for {name} with local window"

    def test_backward_large_input_no_nan(self, cuda_device):
        B, S, H, D = 1, 64, 2, 64
        w1, w2 = S, S
        Q, K1, K2, V1, V2 = _make_qkv(B, S, H, D, cuda_device, seed=4)
        Q = Q * 50.0
        Q = Q.clone().detach().requires_grad_(True)
        K1 = K1.clone().detach().requires_grad_(True)
        K2 = K2.clone().detach().requires_grad_(True)
        V1 = V1.clone().detach().requires_grad_(True)
        V2 = V2.clone().detach().requires_grad_(True)

        O, M = _call_fwd_kernel(Q, K1, K2, V1, V2, w1, w2)
        dO = torch.randn_like(O)
        O.backward(dO)

        for name, t in [("Q", Q), ("K1", K1), ("K2", K2), ("V1", V1), ("V2", V2)]:
            assert torch.isfinite(t.grad.float()).all(), f"Non-finite grad for {name} with large input"

    def test_backward_multi_head(self, cuda_device):
        B, S, H, D = 1, 32, 4, 32
        w1, w2 = S, S
        Q, K1, K2, V1, V2 = _make_qkv(B, S, H, D, cuda_device, seed=5)
        Q = Q.clone().detach().requires_grad_(True)
        K1 = K1.clone().detach().requires_grad_(True)
        K2 = K2.clone().detach().requires_grad_(True)
        V1 = V1.clone().detach().requires_grad_(True)
        V2 = V2.clone().detach().requires_grad_(True)

        O, M = _call_fwd_kernel(Q, K1, K2, V1, V2, w1, w2)
        dO = torch.randn_like(O)
        O.backward(dO)

        for name, t in [("Q", Q), ("K1", K1), ("K2", K2), ("V1", V1), ("V2", V2)]:
            assert torch.isfinite(t.grad.float()).all(), f"Non-finite grad for {name} in multi-head backward"
