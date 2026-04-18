# 2-Simplex

Fast and Simple: **2-Simplicial Attention** implemented in PyTorch with high-performance **Triton GPU kernels**.

## Overview

This project implements a novel attention mechanism that operates on **simplicial complexes** (triangulations) rather than standard sequences. It extends the attention paradigm to higher-order structures using triple-product attention scores over triangle neighborhoods.

### Key Features

- **2-Simplicial Attention**: High-order attention over (j,k) neighbor pairs using triple-product scoring.
- **Triton Kernels**: Fully functional, GPU-optimized Forward and Backward kernels.
- **Mathematical Parity**: Rigorously validated against PyTorch reference implementations (72+ tests passing on CUDA).
- **Multi-head support**: Configurable number of attention heads.
- **Residual connections**: Optional residual mapping and LayerNorm.
- **Vanilla PyTorch fallback**: Transparent fallback to standard PyTorch for CPU or non-Triton environments.

## Project Structure

```
2-simplex/
├── configs/                  # YAML configuration files
├── src/                      # Source code
│   ├── config/               # Config loading utilities
│   ├── kernels/              # Triton GPU kernels (Forward/Backward/Launcher)
│   └── models/               # PyTorch model definitions
├── tests/                    # Comprehensive Test Suite
│   ├── core/                 # Model logic and math correctness
│   ├── edge_cases/           # Numerical stability and graph validation
│   └── triton/               # Triton kernel parity and performance tests
├── scripts/                  # Training and utility scripts
└── requirements.txt          # Python dependencies
```

## Installation

```bash
# Clone the repository
git clone https://github.com/and-per-i/2-simplex.git
cd 2-simplex

# Install dependencies
pip install -r requirements.txt

# Triton requires an NVIDIA GPU with compatible drivers
pip install triton
```

## Usage

### Basic Example

```python
import torch
from src.models.two_simplicial_attention import TwoSimplicialAttention

# Initialize model with Triton kernels enabled
model = TwoSimplicialAttention(
    in_dim=32,
    out_dim=64,
    num_heads=4,
    use_triton_kernel=True
).cuda()

# Input: 128 triangles, 32-dim features
tri_feats = torch.randn(128, 32).cuda()
# edge_index: (N, max_deg) with -1 padding for neighbors
edge_index = torch.randint(-1, 128, (128, 8)).cuda()

# Forward pass (uses optimized Triton kernels)
output = model(tri_feats, edge_index)
print(output.shape)  # (128, 64)
```

## Mathematical Foundation

The core 2-simplicial attention mechanism computes:

- **Projections**: $Q = XW_Q, K = XW_K, V = XW_V, K' = XW_{K'}, V' = XW_{V'}$
- **Attention score**: $A_{ijk} = \frac{1}{\sqrt{d}} \langle q_i, k_j \odot k'_k \rangle$
- **Softmax**: $S_{ijk} = \text{softmax}_{j,k}(A_{ijk})$
- **Output**: $v_i = \sum_{j,k} S_{ijk} (v_j \odot v'_k)$

## Performance & Validation

The implementation has been validated on NVIDIA hardware:
- **Numerical Parity**: Triton kernels match PyTorch reference within $10^{-2}$ absolute tolerance for BF16/TF32.
- **Autograd Integration**: Seamlessly integrated into PyTorch `autograd` via custom `torch.autograd.Function`.
- **Layout**: Optimized for `[Batch, Seq, Head, Dim]` memory layout.

## Running Tests

To run the full validation suite (requires CUDA for kernel tests):

```bash
export PYTHONPATH=$PYTHONPATH:.
pytest tests/ -v
```

To run only the core PyTorch logic (CPU compatible):

```bash
pytest tests/core/ -v
```

## Status

- [x] Functional 2-Simplicial Attention (PyTorch)
- [x] Optimized Triton Forward Kernel
- [x] Optimized Triton Backward Kernel
- [x] Comprehensive CUDA Test Suite (72/72 Passing)
- [ ] Multi-Batch Support (currently optimized for $B=1$)
- [ ] Real-world dataset integration (e.g., Shrec)

## License

MIT
