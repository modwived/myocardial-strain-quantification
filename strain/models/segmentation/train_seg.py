"""Training loop for cardiac segmentation U-Net."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment,misc]

from strain.data.dataset import SegmentationDataset
from strain.models.segmentation.losses import DiceCELoss, DiceLoss
from strain.models.segmentation.metrics import dice_score
from strain.models.segmentation.unet import UNet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path) -> dict:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Configuration dictionary.

    Raises:
        RuntimeError: If PyYAML is not installed.
    """
    if yaml is None:
        raise RuntimeError("PyYAML is required to load config files. Install with: pip install pyyaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _default_config() -> dict:
    """Return a default configuration matching configs/segmentation.yaml."""
    return {
        "model": {
            "architecture": "unet",
            "in_channels": 1,
            "num_classes": 4,
            "features": [64, 128, 256, 512],
            "dropout": 0.1,
            "deep_supervision": False,
        },
        "training": {
            "optimizer": "adamw",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "scheduler": "cosine_warm_restarts",
            "scheduler_T0": 50,
            "scheduler_Tmult": 2,
            "loss": "dice_ce",
            "dice_weight": 0.5,
            "ce_weight": 0.5,
            "max_epochs": 200,
            "early_stopping_patience": 20,
            "batch_size": 16,
            "num_workers": 4,
            "val_interval": 1,
            "deep_supervision_weight": 0.3,
        },
        "paths": {
            "data_root": "./data/ACDC",
            "checkpoint_dir": "./checkpoints/segmentation",
            "log_dir": "./logs/segmentation",
        },
    }


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def build_model(cfg: dict) -> UNet:
    """Instantiate a UNet from a model configuration block."""
    model_cfg = cfg.get("model", {})
    return UNet(
        in_channels=model_cfg.get("in_channels", 1),
        num_classes=model_cfg.get("num_classes", 4),
        features=tuple(model_cfg.get("features", [64, 128, 256, 512])),
        dropout=model_cfg.get("dropout", 0.1),
        deep_supervision=model_cfg.get("deep_supervision", False),
    )


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """Build an optimizer from the training configuration block."""
    train_cfg = cfg.get("training", {})
    name = train_cfg.get("optimizer", "adamw").lower()
    lr = train_cfg.get("learning_rate", 1e-3)
    wd = train_cfg.get("weight_decay", 1e-4)

    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: dict
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Build a learning-rate scheduler from the training configuration."""
    train_cfg = cfg.get("training", {})
    name = train_cfg.get("scheduler", "cosine_warm_restarts").lower()

    if name in ("cosine_warm_restarts", "cosine"):
        t0 = train_cfg.get("scheduler_T0", 50)
        t_mult = train_cfg.get("scheduler_Tmult", 2)
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t0, T_mult=t_mult
        )
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {name}")


def build_loss(cfg: dict) -> nn.Module:
    """Build a loss function from the training configuration."""
    train_cfg = cfg.get("training", {})
    name = train_cfg.get("loss", "dice_ce").lower()

    if name == "dice_ce":
        return DiceCELoss(
            dice_weight=train_cfg.get("dice_weight", 0.5),
            ce_weight=train_cfg.get("ce_weight", 0.5),
        )
    elif name == "dice":
        return DiceLoss()
    elif name == "ce":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss: {name}")


def build_dataloaders(
    cfg: dict,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders.

    The dataset is split 80/20 into train/val.

    Args:
        cfg: Full configuration dictionary.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    paths_cfg = cfg.get("paths", {})
    train_cfg = cfg.get("training", {})

    data_root = paths_cfg.get("data_root", "./data/ACDC")
    batch_size = train_cfg.get("batch_size", 16)
    num_workers = train_cfg.get("num_workers", 4)
    crop_size = train_cfg.get("crop_size", 128)

    dataset = SegmentationDataset(root=data_root, crop_size=crop_size, augment=True)

    # Deterministic split
    n_val = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: UNet,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 4,
) -> tuple[float, float]:
    """Run one validation pass.

    Args:
        model: Segmentation model (set to eval mode internally).
        val_loader: Validation dataloader.
        criterion: Loss function.
        device: Torch device.
        num_classes: Number of segmentation classes.

    Returns:
        Tuple of (mean_val_loss, mean_dice) where mean_dice is averaged over
        foreground classes.
    """
    model.eval()
    total_loss = 0.0
    all_dice: list[float] = []
    n_batches = 0

    for batch in val_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        preds = logits.argmax(dim=1).cpu().numpy()
        targets = labels.cpu().numpy()

        for pred_slice, target_slice in zip(preds, targets):
            scores = dice_score(pred_slice, target_slice, num_classes=num_classes)
            all_dice.append(float(np.mean(list(scores.values()))))

        n_batches += 1

    mean_loss = total_loss / max(n_batches, 1)
    mean_dice = float(np.mean(all_dice)) if all_dice else 0.0
    return mean_loss, mean_dice


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config_path: str | None = None, config: dict | None = None) -> Path:
    """Run the full segmentation training pipeline.

    Args:
        config_path: Path to a YAML configuration file. If ``None`` and
            *config* is also ``None``, default hyperparameters are used.
        config: Configuration dictionary. Overrides *config_path* if given.

    Returns:
        Path to the best model checkpoint.
    """
    # --- Configuration ---
    if config is not None:
        cfg = config
    elif config_path is not None:
        cfg = load_config(config_path)
    else:
        cfg = _default_config()

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    paths_cfg = cfg.get("paths", {})

    max_epochs: int = train_cfg.get("max_epochs", 200)
    patience: int = train_cfg.get("early_stopping_patience", 20)
    val_interval: int = train_cfg.get("val_interval", 1)
    num_classes: int = model_cfg.get("num_classes", 4)
    ds_weight: float = train_cfg.get("deep_supervision_weight", 0.3)

    checkpoint_dir = Path(paths_cfg.get("checkpoint_dir", "./checkpoints/segmentation"))
    log_dir = Path(paths_cfg.get("log_dir", "./logs/segmentation"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # --- Build components ---
    model = build_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    criterion = build_loss(cfg)
    train_loader, val_loader = build_dataloaders(cfg)

    # TensorBoard
    writer = None
    if SummaryWriter is not None:
        writer = SummaryWriter(log_dir=str(log_dir))

    # --- Training loop ---
    best_dice = 0.0
    epochs_without_improvement = 0
    best_ckpt_path = checkpoint_dir / "best_model.pth"

    logger.info("Starting training for up to %d epochs", max_epochs)
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            output = model(images)

            # Handle deep supervision
            if isinstance(output, tuple):
                main_logits, aux_list = output
                loss = criterion(main_logits, labels)
                for aux_logits in aux_list:
                    loss = loss + ds_weight * criterion(aux_logits, labels)
            else:
                loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if scheduler is not None:
            scheduler.step(epoch)

        mean_train_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        # Logging
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %d/%d | train_loss=%.4f | lr=%.2e | %.1fs",
            epoch, max_epochs, mean_train_loss, current_lr, elapsed,
        )
        if writer is not None:
            writer.add_scalar("train/loss", mean_train_loss, epoch)
            writer.add_scalar("train/lr", current_lr, epoch)

        # Validation
        if epoch % val_interval == 0:
            val_loss, val_dice = validate(model, val_loader, criterion, device, num_classes)
            logger.info(
                "  val_loss=%.4f | val_dice=%.4f | best_dice=%.4f",
                val_loss, val_dice, best_dice,
            )
            if writer is not None:
                writer.add_scalar("val/loss", val_loss, epoch)
                writer.add_scalar("val/dice", val_dice, epoch)

            if val_dice > best_dice:
                best_dice = val_dice
                epochs_without_improvement = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_dice": best_dice,
                        "config": cfg,
                    },
                    best_ckpt_path,
                )
                logger.info("  Saved new best model (dice=%.4f)", best_dice)
            else:
                epochs_without_improvement += 1

            # Early stopping
            if epochs_without_improvement >= patience:
                logger.info(
                    "Early stopping at epoch %d (no improvement for %d epochs)",
                    epoch, patience,
                )
                break

    # Save final checkpoint
    final_ckpt_path = checkpoint_dir / "final_model.pth"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dice": best_dice,
            "config": cfg,
        },
        final_ckpt_path,
    )

    if writer is not None:
        writer.close()

    logger.info("Training complete. Best Dice: %.4f", best_dice)
    return best_ckpt_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line entry point for segmentation training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train cardiac segmentation U-Net")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/segmentation.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()
    train(config_path=args.config)


if __name__ == "__main__":
    main()
