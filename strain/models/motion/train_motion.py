"""Unsupervised training loop for the CarMEN motion estimation network.

Usage::

    python -m strain.models.motion.train_motion --config configs/motion.yaml

The training is fully unsupervised: the similarity loss (NCC) compares the
warped source to the target, and the regularization loss penalises non-smooth
displacement fields.  Validation quality is measured by SSIM between the
warped source and target on a held-out split.  TensorBoard logs are written
for both scalar metrics and displacement-field visualisations.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import yaml

from strain.data.dataset import MotionDataset
from strain.models.motion.carmen import CarMEN
from strain.models.motion.losses import (
    BendingEnergy,
    CyclicConsistencyLoss,
    DiffusionRegularizer,
    NCC,
)
from strain.models.motion.warp import spatial_transform

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SSIM (structural similarity) — lightweight implementation for validation
# ---------------------------------------------------------------------------

def _ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Compute mean SSIM between two (B, 1, H, W) images."""
    pad = window_size // 2
    pool = nn.AvgPool2d(window_size, stride=1, padding=pad)

    mu1 = pool(img1)
    mu2 = pool(img2)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = pool(img1 * img1) - mu1_sq
    sigma2_sq = pool(img2 * img2) - mu2_sq
    sigma12 = pool(img1 * img2) - mu12

    num = (2 * mu12 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    return (num / den).mean()


# ---------------------------------------------------------------------------
# Displacement field visualisation helper
# ---------------------------------------------------------------------------

def _displacement_to_rgb(displacement: torch.Tensor) -> torch.Tensor:
    """Convert a single (2, H, W) displacement to an RGB image tensor (3, H, W).

    Uses HSV encoding: hue = angle, value = magnitude (clamped to [0, 1]).
    Returns an (3, H, W) float tensor in [0, 1].
    """
    dx = displacement[0].detach().cpu().numpy()
    dy = displacement[1].detach().cpu().numpy()

    mag = np.sqrt(dx ** 2 + dy ** 2)
    angle = np.arctan2(dy, dx)  # [-pi, pi]

    # Normalise
    max_mag = mag.max() + 1e-8
    mag_norm = np.clip(mag / max_mag, 0, 1)
    hue = (angle + np.pi) / (2 * np.pi)  # [0, 1]

    # HSV -> RGB (vectorised)
    h6 = hue * 6.0
    sector = h6.astype(int) % 6
    f = h6 - sector
    v = mag_norm
    p = np.zeros_like(v)
    q = v * (1 - f)
    t = v * f

    rgb = np.zeros((*mag.shape, 3), dtype=np.float32)
    for s, (r, g, b) in enumerate(
        [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    ):
        mask = sector == s
        rgb[mask, 0] = r[mask]
        rgb[mask, 1] = g[mask]
        rgb[mask, 2] = b[mask]

    return torch.from_numpy(rgb.transpose(2, 0, 1))  # (3, H, W)


# ---------------------------------------------------------------------------
# Build regularizer from config
# ---------------------------------------------------------------------------

def _build_regularizer(config: dict) -> nn.Module:
    """Instantiate the configured regularizer."""
    name = config.get("regularization", "diffusion").lower()
    if name == "diffusion":
        return DiffusionRegularizer()
    if name in ("bending", "bending_energy"):
        return BendingEnergy()
    raise ValueError(f"Unknown regularizer: {name}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(config_path: str) -> None:
    """Run the full unsupervised motion estimation training.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file (see ``configs/motion.yaml``).
    """
    # ------------------------------------------------------------------
    # 1.  Load config
    # ------------------------------------------------------------------
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    path_cfg = config.get("paths", {})

    device = torch.device(config.get("inference", {}).get("device", "cuda")
                          if torch.cuda.is_available() else "cpu")

    checkpoint_dir = Path(path_cfg.get("checkpoint_dir", "./checkpoints/motion"))
    log_dir = Path(path_cfg.get("log_dir", "./logs/motion"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2.  Dataset + dataloaders
    # ------------------------------------------------------------------
    data_root = path_cfg.get("data_root", "./data/ACDC")
    crop_size = model_cfg.get("crop_size", 128)
    batch_size = train_cfg.get("batch_size", 8)

    dataset = MotionDataset(root=data_root, crop_size=crop_size)

    # 80/20 train/val split
    n_val = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=min(4, os.cpu_count() or 1), pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=min(4, os.cpu_count() or 1), pin_memory=True,
    )

    # ------------------------------------------------------------------
    # 3.  Model, losses, optimizer, scheduler
    # ------------------------------------------------------------------
    features = tuple(model_cfg.get("features", [16, 32, 64, 128]))
    in_channels = model_cfg.get("in_channels", 1)

    model = CarMEN(
        in_channels=in_channels,
        features=features,
        multi_scale=model_cfg.get("multi_scale", False),
        use_seg_attention=model_cfg.get("use_seg_attention", False),
    ).to(device)

    ncc_window = train_cfg.get("ncc_window", 9)
    sim_loss_fn = NCC(window_size=ncc_window)
    reg_loss_fn = _build_regularizer(train_cfg)
    reg_weight = float(train_cfg.get("reg_weight", 1.0))

    lr = float(train_cfg.get("learning_rate", 1e-4))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler_step = int(train_cfg.get("scheduler_step", 100))
    scheduler_gamma = float(train_cfg.get("scheduler_gamma", 0.5))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step, gamma=scheduler_gamma,
    )

    max_epochs = int(train_cfg.get("max_epochs", 300))

    # ------------------------------------------------------------------
    # TensorBoard (optional — graceful fallback if not installed)
    # ------------------------------------------------------------------
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(log_dir))
    except ImportError:
        logger.warning("TensorBoard not available — skipping logging")
        writer = None

    # ------------------------------------------------------------------
    # 4.  Training loop
    # ------------------------------------------------------------------
    best_val_ssim = -1.0
    global_step = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_sim_loss = 0.0
        epoch_reg_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            source = batch["source"].to(device)
            target = batch["target"].to(device)

            displacement = model(source, target)

            # Handle multi-scale output: use only the finest for the main loss
            if isinstance(displacement, list):
                displacement_full = displacement[-1]
            else:
                displacement_full = displacement

            warped = spatial_transform(source, displacement_full)

            loss_sim = sim_loss_fn(warped, target)
            loss_reg = reg_loss_fn(displacement_full)
            loss = loss_sim + reg_weight * loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_sim_loss += loss_sim.item()
            epoch_reg_loss += loss_reg.item()
            n_batches += 1
            global_step += 1

            if writer is not None:
                writer.add_scalar("train/loss_sim", loss_sim.item(), global_step)
                writer.add_scalar("train/loss_reg", loss_reg.item(), global_step)
                writer.add_scalar("train/loss_total", loss.item(), global_step)

        scheduler.step()

        avg_sim = epoch_sim_loss / max(n_batches, 1)
        avg_reg = epoch_reg_loss / max(n_batches, 1)
        logger.info(
            "Epoch %3d/%d  sim=%.4f  reg=%.4f  lr=%.2e",
            epoch, max_epochs, avg_sim, avg_reg, scheduler.get_last_lr()[0],
        )

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        model.eval()
        val_ssim_sum = 0.0
        val_n = 0

        with torch.no_grad():
            for batch in val_loader:
                source = batch["source"].to(device)
                target = batch["target"].to(device)

                displacement = model(source, target)
                if isinstance(displacement, list):
                    displacement = displacement[-1]

                warped = spatial_transform(source, displacement)
                ssim_val = _ssim(warped, target)
                val_ssim_sum += ssim_val.item() * source.size(0)
                val_n += source.size(0)

        avg_val_ssim = val_ssim_sum / max(val_n, 1)
        logger.info("  val SSIM = %.4f", avg_val_ssim)

        if writer is not None:
            writer.add_scalar("val/ssim", avg_val_ssim, epoch)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

            # Log displacement field visualisation every 10 epochs
            if epoch % 10 == 0:
                vis = _displacement_to_rgb(displacement[0])
                writer.add_image("val/displacement_field", vis, epoch)
                writer.add_image("val/source", source[0], epoch)
                writer.add_image("val/target", target[0], epoch)
                writer.add_image("val/warped", warped[0], epoch)

        # ------------------------------------------------------------------
        # Checkpoint (save best model by val SSIM)
        # ------------------------------------------------------------------
        if avg_val_ssim > best_val_ssim:
            best_val_ssim = avg_val_ssim
            ckpt_path = checkpoint_dir / "best_motion.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_ssim": avg_val_ssim,
                    "config": config,
                },
                ckpt_path,
            )
            logger.info("  Saved best checkpoint (SSIM=%.4f) -> %s", avg_val_ssim, ckpt_path)

    # Save final model
    final_path = checkpoint_dir / "final_motion.pt"
    torch.save(
        {
            "epoch": max_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_ssim": avg_val_ssim,
            "config": config,
        },
        final_path,
    )
    logger.info("Training complete. Final model saved to %s", final_path)

    if writer is not None:
        writer.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train CarMEN motion estimation network")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/motion.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
