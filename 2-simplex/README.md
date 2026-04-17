# 2-Simplex

Fast and Simple: **2-Simplicial Attention** implemented in PyTorch with optional Triton GPU kernels.

## Overview

This project implements a novel attention mechanism that operates on **simplicial complexes** (triangulations) rather than standard sequences. It extends the attention paradigm to higher-order structures using triple-product attention scores over triangle neighborhoods.

### Key Features

- **2-Simplicial Attention**: Attention over (j,k) neighbor pairs using triple-product scoring
- **Multi-head support**: Configurable number of attention heads
- **Residual connections**: Optional residual + LayerNorm
- **Triton kernels**: Optional GPU-optimized forward/backward kernels (work in progress)
- **Vanilla PyTorch fallback**: Full PyTorch reference implementation

## Project Structure

```
2-simplex/
├── configs/                  # YAML configuration files
│   └── train_config.yaml     # Training hyperparameters
├── src/                      # Source code
│   ├── config/               # Config loading utilities
│   ├── kernels/              # Triton GPU kernels
│   │   ├── triton_2s_forward.py
│   │   ├── triton_2s_backward.py
│   │   └── triton_launcher.py
│   └── models/               # PyTorch model definitions
│       └── two_simplicial_attention.py
├── tests/                    # Test suite
│   ├── config/               # Config loading tests
│   ├── core/                 # Core functionality tests
│   ├── edge_cases/           # Edge case and validation tests
│   ├── kernels/              # Kernel tests
│   └── triton/               # Triton-specific tests
├── scripts/                  # Training and utility scripts
├── data/                     # Dataset storage (gitignored)
├── notebooks/                # Jupyter notebooks for exploration
├── examples/                 # Usage examples
├── checkpoints/              # Model checkpoints (gitignored)
├── requirements.txt          # Python dependencies
└── README.md
```

## Installation

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: Triton for GPU acceleration (requires NVIDIA GPU)
pip install triton
```

## Usage

### Basic Example

```python
import torch
from src.models.two_simplicial_attention import TwoSimplicialAttention

# Create model
model = TwoSimplicialAttention(
    in_dim=32,
    out_dim=64,
    num_heads=4,
    dropout=0.1,
    with_residual=True,
    use_triton_kernel=False
)

# Dummy input: 128 triangles, 32-dim features
tri_feats = torch.randn(128, 32)

# Edge index: (N, max_deg) with -1 padding
edge_index = torch.randint(-1, 128, (128, 8))

# Forward pass
output = model(tri_feats, edge_index)
print(output.shape)  # (128, 64)
```

### Training

```bash
# Run training with default config
python scripts/train.py

# Or with custom config
python scripts/train.py --config configs/train_config.yaml
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run core tests only
pytest tests/core/

# Run with verbose output
pytest tests/ -v

# Run Triton tests (requires CUDA + Triton)
pytest tests/triton/ -v
```

## Configuration

Training parameters are centralized in `configs/train_config.yaml`:

```yaml
model:
  in_dim: 32
  out_dim: 64
  num_heads: 4
  dropout: 0.0
  with_residual: true
  use_triton_kernel: false

trainer:
  epochs: 50
  batch_size: 1
  learning_rate: 0.001
  device: cpu
  seed: 42
```

## Implemented Equations

The core attention mechanism computes:

- **Projections**: Q = XW_Q, K = XW_K, V = XW_V, K' = XW_K', V' = XW_V'
- **Attention score**: A_ijk = (1/√d) <q_i, k_j, k'_k>
- **Softmax**: S_ijk = softmax_{j,k}(A_ijk)
- **Output**: v_i = Σ_{j,k} S_ijk (v_j ∘ v'_k)

## Status

This is an **MVP (Minimum Viable Product)** with the following limitations:

- Batch size = 1 only
- Triton backward kernel is a placeholder (returns zero gradients)
- Synthetic data only (no real dataset integration yet)

## License

MIT
