"""PyTorch datasets for cardiac MRI segmentation and motion estimation.

Supports ACDC and M&Ms directory layouts, per-slice iteration, deterministic
stratified splitting, data caching, and configurable data-loaders with
augmentation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from strain.data.augmentation import (
    Compose,
    random_elastic_deformation,
    random_flip,
    random_gamma,
    random_noise,
    random_rotation,
    random_scale,
)
from strain.data.loader import load_nifti
from strain.data.preprocessing import center_crop, normalize_intensity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Segmentation dataset — iterates ALL slices
# ---------------------------------------------------------------------------

class SegmentationDataset(Dataset):
    """Dataset for cardiac segmentation training.

    Iterates over **every** 2-D slice (not just mid-ventricular) so the
    network sees the full short-axis stack.

    Supports two directory layouts:

    **ACDC-style** (default):
    ::

        root/
            patient001/
                patient001_frame01.nii.gz
                patient001_frame01_gt.nii.gz
                ...

    **M&Ms-style** (``dataset_format="mnms"``):
    ::

        root/
            Vendor_A/
                patient001/
                    patient001_sa_ED.nii.gz
                    patient001_sa_ED_gt.nii.gz
                    ...

    Each ``__getitem__`` returns::

        {"image": Tensor(1, crop_size, crop_size), "label": Tensor(crop_size, crop_size)}
    """

    def __init__(
        self,
        root: str | Path,
        crop_size: int = 128,
        augment: bool = False,
        augment_pipeline: Compose | None = None,
        dataset_format: str = "acdc",
        cache: bool = False,
    ):
        self.root = Path(root)
        self.crop_size = crop_size
        self.augment = augment
        self.augment_pipeline = augment_pipeline
        self.dataset_format = dataset_format.lower()
        self.cache = cache

        # (img_path, gt_path, slice_index) for every 2-D slice
        self.samples: list[tuple[Path, Path, int]] = []
        # Optional RAM cache:  key = (img_path, slice_idx) -> (image_slice, label_slice)
        self._cache: dict[tuple[Path, int], tuple[np.ndarray, np.ndarray]] = {}

        self._discover_samples()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover_samples(self) -> None:
        """Find all (image, ground_truth, slice_index) triples."""
        pairs = self._find_pairs()
        for img_path, gt_path in pairs:
            try:
                image, _ = load_nifti(img_path)
                n_slices = image.shape[0] if image.ndim == 3 else 1
                for s in range(n_slices):
                    self.samples.append((img_path, gt_path, s))
            except Exception as exc:
                logger.warning("Skipping %s: %s", img_path, exc)

    def _find_pairs(self) -> list[tuple[Path, Path]]:
        if self.dataset_format == "mnms":
            return self._find_pairs_mnms()
        return self._find_pairs_acdc()

    def _find_pairs_acdc(self) -> list[tuple[Path, Path]]:
        pairs = []
        for gt_path in sorted(self.root.rglob("*_gt.nii.gz")):
            img_path = Path(str(gt_path).replace("_gt.nii.gz", ".nii.gz"))
            if img_path.exists():
                pairs.append((img_path, gt_path))
        return pairs

    def _find_pairs_mnms(self) -> list[tuple[Path, Path]]:
        """Discover M&Ms-style pairs (``*_gt.nii.gz`` next to images)."""
        pairs = []
        for gt_path in sorted(self.root.rglob("*_gt.nii.gz")):
            img_path = Path(str(gt_path).replace("_gt.nii.gz", ".nii.gz"))
            if img_path.exists():
                pairs.append((img_path, gt_path))
        return pairs

    # ------------------------------------------------------------------
    # __len__ / __getitem__
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path, gt_path, slice_idx = self.samples[idx]

        image_slice, label_slice = self._load_slice(img_path, gt_path, slice_idx)

        # Preprocessing
        image_slice = normalize_intensity(image_slice.astype(np.float32))
        image_slice = center_crop(image_slice, self.crop_size)
        label_slice = center_crop(label_slice, self.crop_size)

        # Augmentation
        if self.augment and self.augment_pipeline is not None:
            image_slice, label_slice = self.augment_pipeline(image_slice, label_slice)

        image_tensor = torch.from_numpy(np.ascontiguousarray(image_slice)).unsqueeze(0).float()
        label_tensor = torch.from_numpy(np.ascontiguousarray(label_slice)).long()

        return {"image": image_tensor, "label": label_tensor}

    def _load_slice(
        self,
        img_path: Path,
        gt_path: Path,
        slice_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        cache_key = (img_path, slice_idx)
        if self.cache and cache_key in self._cache:
            return self._cache[cache_key]

        image, _ = load_nifti(img_path)
        label, _ = load_nifti(gt_path)

        if image.ndim == 3:
            image_slice = image[slice_idx]
            label_slice = label[slice_idx]
        else:
            # 2-D already
            image_slice = image
            label_slice = label

        if self.cache:
            self._cache[cache_key] = (image_slice, label_slice)

        return image_slice, label_slice


# ---------------------------------------------------------------------------
# Motion dataset
# ---------------------------------------------------------------------------

class MotionDataset(Dataset):
    """Dataset for motion estimation training.

    Provides pairs of consecutive cardiac frames from 4-D cine volumes.

    Each ``__getitem__`` returns::

        {"source": Tensor(1, crop_size, crop_size),
         "target": Tensor(1, crop_size, crop_size)}
    """

    def __init__(
        self,
        root: str | Path,
        crop_size: int = 128,
        cache: bool = False,
    ):
        self.root = Path(root)
        self.crop_size = crop_size
        self.cache = cache
        self._volume_cache: dict[Path, np.ndarray] = {}
        self.sequences = self._discover_sequences()

    def _discover_sequences(self) -> list[Path]:
        """Find all 4D cine volumes."""
        return sorted(self.root.rglob("*_4d.nii.gz"))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        vol_path = self.sequences[idx]
        volume = self._load_volume(vol_path)

        # Take mid-slice if 3-D+time (T, D, H, W)
        if volume.ndim == 4:
            mid = volume.shape[1] // 2
            volume = volume[:, mid]  # (T, H, W)

        # Pick a random consecutive pair
        t = np.random.randint(0, volume.shape[0] - 1)
        frame_a = normalize_intensity(volume[t].astype(np.float32))
        frame_b = normalize_intensity(volume[t + 1].astype(np.float32))

        frame_a = center_crop(frame_a, self.crop_size)
        frame_b = center_crop(frame_b, self.crop_size)

        return {
            "source": torch.from_numpy(np.ascontiguousarray(frame_a)).unsqueeze(0).float(),
            "target": torch.from_numpy(np.ascontiguousarray(frame_b)).unsqueeze(0).float(),
        }

    def _load_volume(self, path: Path) -> np.ndarray:
        if self.cache and path in self._volume_cache:
            return self._volume_cache[path]
        volume, _ = load_nifti(path)
        if self.cache:
            self._volume_cache[path] = volume
        return volume


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_dataset(
    root: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Deterministic stratified split of patient directories.

    The split is performed at the *patient* level (not slice level) to
    prevent data leakage.  Patient directories are shuffled with a fixed
    seed for reproducibility.

    Args:
        root: Root dataset directory containing patient sub-directories.
        train_ratio: Fraction of patients for training.
        val_ratio: Fraction of patients for validation.
        seed: Random seed.

    Returns:
        Three lists of patient directory names: (train, val, test).
    """
    root = Path(root)
    patients = sorted(
        d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    rng = np.random.RandomState(seed)
    rng.shuffle(patients)

    n = len(patients)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    train_patients = patients[:n_train]
    val_patients = patients[n_train : n_train + n_val]
    test_patients = patients[n_train + n_val :]

    return train_patients, val_patients, test_patients


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def _build_augmentation_pipeline(aug_cfg: dict[str, Any]) -> Compose | None:
    """Build augmentation Compose from config dict."""
    if not aug_cfg.get("enabled", False):
        return None

    transforms: list[tuple[callable, float]] = []

    max_angle = aug_cfg.get("rotation_max", 15.0)
    transforms.append(
        (lambda img, msk, _a=max_angle: random_rotation(img, msk, _a), 0.8)
    )

    sr = aug_cfg.get("scale_range", [0.9, 1.1])
    transforms.append(
        (lambda img, msk, _sr=tuple(sr): random_scale(img, msk, _sr), 0.5)
    )

    gr = aug_cfg.get("gamma_range", [0.7, 1.5])
    transforms.append(
        (lambda img, msk, _gr=tuple(gr): (random_gamma(img, _gr), msk), 0.5)
    )

    alpha = aug_cfg.get("elastic_alpha", 100.0)
    sigma = aug_cfg.get("elastic_sigma", 10.0)
    transforms.append(
        (lambda img, msk, _a=alpha, _s=sigma: random_elastic_deformation(img, msk, _a, _s), 0.3)
    )

    noise_std = aug_cfg.get("noise_std", 0.02)
    transforms.append(
        (lambda img, msk, _std=noise_std: (random_noise(img, _std), msk), 0.5)
    )

    if aug_cfg.get("flip_horizontal", False):
        transforms.append(
            (lambda img, msk: random_flip(img, msk, axis=-1, prob=1.0), 0.5)
        )

    return Compose(transforms)


def get_dataloaders(config: dict[str, Any]) -> dict[str, DataLoader]:
    """Create train / val / test DataLoaders from a configuration dict.

    The configuration should match the structure in ``configs/data.yaml``.

    Args:
        config: Full data configuration dictionary.

    Returns:
        Dictionary with keys ``"train"``, ``"val"``, ``"test"`` mapping to
        :class:`~torch.utils.data.DataLoader` instances.
    """
    ds_cfg = config.get("dataset", {})
    pp_cfg = config.get("preprocessing", {})
    aug_cfg = config.get("augmentation", {})
    dl_cfg = config.get("dataloader", {})
    split_cfg = config.get("split", {})

    root = Path(ds_cfg.get("root", "./data/ACDC"))
    dataset_format = ds_cfg.get("name", "acdc")
    crop_size = pp_cfg.get("crop_size", 128)
    cache = ds_cfg.get("cache", False)

    # Patient-level split
    train_patients, val_patients, test_patients = split_dataset(
        root,
        train_ratio=split_cfg.get("train", 0.70),
        val_ratio=split_cfg.get("val", 0.15),
        seed=split_cfg.get("seed", 42),
    )

    aug_pipeline = _build_augmentation_pipeline(aug_cfg)

    def _make_dataset(patients: list[str], augment: bool) -> SegmentationDataset:
        """Create a SegmentationDataset for a subset of patients.

        We create a temporary directory-like structure by filtering samples
        after building the full dataset.
        """
        ds = SegmentationDataset(
            root=root,
            crop_size=crop_size,
            augment=augment,
            augment_pipeline=aug_pipeline if augment else None,
            dataset_format=dataset_format,
            cache=cache,
        )
        # Filter to only samples belonging to the given patients
        patient_set = set(patients)
        filtered = [
            s for s in ds.samples
            if any(p in str(s[0]) for p in patient_set)
        ]
        ds.samples = filtered
        return ds

    train_ds = _make_dataset(train_patients, augment=True)
    val_ds = _make_dataset(val_patients, augment=False)
    test_ds = _make_dataset(test_patients, augment=False)

    batch_size = dl_cfg.get("batch_size", 16)
    num_workers = dl_cfg.get("num_workers", 4)
    pin_memory = dl_cfg.get("pin_memory", True)

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
    return loaders
