"""
Training script for TwoSimplicialAttention model.

Usage:
    python scripts/train.py [--config configs/train_config.yaml]
"""

import os
import argparse
import torch
import torch.nn.functional as F
from src.config import load_config
from src.models.two_simplicial_attention import TwoSimplicialAttention


def create_synthetic_data(n_triangles, in_dim, max_neighbors, device):
    """Create synthetic triangle features and edge index for MVP testing."""
    tri_feats = torch.randn(n_triangles, in_dim, device=device)

    # Create synthetic edge_index with -1 padding
    edge_index = torch.full((n_triangles, max_neighbors), -1, dtype=torch.long, device=device)
    for i in range(n_triangles):
        n_neighbors = min(max_neighbors, max(1, n_triangles // 8))
        neighbors = torch.randperm(n_triangles)[:n_neighbors]
        neighbors = neighbors[neighbors != i]  # Remove self-loops
        edge_index[i, :len(neighbors)] = neighbors

    return tri_feats, edge_index


def train(config_path="configs/train_config.yaml"):
    """Main training loop."""
    cfg = load_config(config_path)

    model_cfg = cfg["model"]
    trainer_cfg = cfg["trainer"]
    data_cfg = cfg["data"]

    device = torch.device(trainer_cfg.get("device", "cpu"))
    seed = trainer_cfg.get("seed", 42)
    torch.manual_seed(seed)

    model = TwoSimplicialAttention(
        in_dim=model_cfg["in_dim"],
        out_dim=model_cfg["out_dim"],
        num_heads=model_cfg["num_heads"],
        dropout=model_cfg["dropout"],
        with_residual=model_cfg["with_residual"],
        use_triton_kernel=model_cfg.get("use_triton_kernel", False),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=trainer_cfg["learning_rate"],
        weight_decay=trainer_cfg.get("weight_decay", 0.0),
    )

    epochs = trainer_cfg["epochs"]
    log_interval = cfg.get("logging", {}).get("log_interval", 10)

    print(f"Training on {device} for {epochs} epochs")
    print(f"Model: {model_cfg['in_dim']} -> {model_cfg['out_dim']}, {model_cfg['num_heads']} heads")

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        tri_feats, edge_index = create_synthetic_data(
            n_triangles=data_cfg["n_triangles"],
            in_dim=model_cfg["in_dim"],
            max_neighbors=data_cfg["max_neighbors"],
            device=device,
        )

        output = model(tri_feats, edge_index)

        # Dummy loss: encourage output to have unit norm per row
        loss = F.mse_loss(output, torch.zeros_like(output))

        loss.backward()

        grad_clip = trainer_cfg.get("grad_clip")
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        if epoch % log_interval == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.6f}")

    # Save checkpoint
    if cfg.get("logging", {}).get("save_checkpoint", True):
        ckpt_dir = cfg["logging"]["checkpoint_dir"]
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "model_final.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TwoSimplicialAttention")
    parser.add_argument("--config", default="configs/train_config.yaml", help="Path to config file")
    args = parser.parse_args()
    train(args.config)
