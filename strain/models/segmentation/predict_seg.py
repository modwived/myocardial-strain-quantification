"""Inference pipeline for cardiac segmentation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from strain.data.loader import load_nifti
from strain.data.preprocessing import center_crop, normalize_intensity
from strain.models.segmentation.metrics import compute_volumes, dice_score
from strain.models.segmentation.unet import UNet

logger = logging.getLogger(__name__)

# Target crop size expected by the model
_DEFAULT_CROP_SIZE = 128


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cuda",
) -> UNet:
    """Load a trained UNet from a checkpoint.

    Args:
        checkpoint_path: Path to a ``.pth`` checkpoint file saved by
            :func:`train_seg.train`.
        device: Torch device to load the model onto.

    Returns:
        UNet model in eval mode.
    """
    device = torch.device(device)
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})

    model = UNet(
        in_channels=model_cfg.get("in_channels", 1),
        num_classes=model_cfg.get("num_classes", 4),
        features=tuple(model_cfg.get("features", [64, 128, 256, 512])),
        dropout=model_cfg.get("dropout", 0.0),
        deep_supervision=False,  # Not needed during inference
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Preprocessing / postprocessing helpers
# ---------------------------------------------------------------------------

def _preprocess_slice(
    img_slice: np.ndarray,
    crop_size: int = _DEFAULT_CROP_SIZE,
) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
    """Normalize and center-crop a single 2D slice.

    Args:
        img_slice: (H, W) grayscale image.
        crop_size: Target side length.

    Returns:
        Tuple of (tensor, original_hw, pad_or_crop_offsets).
        The tensor has shape (1, 1, crop_size, crop_size).
    """
    original_hw = img_slice.shape[:2]
    normed = normalize_intensity(img_slice.astype(np.float32))

    h, w = normed.shape
    # Pad if smaller than crop_size
    pad_h = max(0, crop_size - h)
    pad_w = max(0, crop_size - w)
    if pad_h > 0 or pad_w > 0:
        normed = np.pad(
            normed,
            ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
            mode="constant",
            constant_values=0,
        )

    cropped = center_crop(normed, crop_size)
    tensor = torch.from_numpy(cropped).unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)

    start_h = max(0, (h + pad_h - crop_size) // 2)
    start_w = max(0, (w + pad_w - crop_size) // 2)

    return tensor, original_hw, (start_h, start_w)


def _postprocess_mask(
    pred_crop: np.ndarray,
    original_hw: tuple[int, int],
    crop_size: int = _DEFAULT_CROP_SIZE,
) -> np.ndarray:
    """Place the cropped prediction back into the original spatial dimensions.

    Args:
        pred_crop: (crop_size, crop_size) integer mask.
        original_hw: Original (H, W) before preprocessing.
        crop_size: Crop size used during preprocessing.

    Returns:
        Segmentation mask with shape *original_hw*.
    """
    oh, ow = original_hw
    result = np.zeros((oh, ow), dtype=pred_crop.dtype)

    # Compute the region in the original image that the crop covers
    start_h = max(0, (oh - crop_size) // 2)
    start_w = max(0, (ow - crop_size) // 2)
    end_h = min(oh, start_h + crop_size)
    end_w = min(ow, start_w + crop_size)

    # Corresponding region in the crop (account for padding)
    crop_start_h = max(0, (crop_size - oh) // 2)
    crop_start_w = max(0, (crop_size - ow) // 2)
    crop_end_h = crop_start_h + (end_h - start_h)
    crop_end_w = crop_start_w + (end_w - start_w)

    result[start_h:end_h, start_w:end_w] = pred_crop[
        crop_start_h:crop_end_h, crop_start_w:crop_end_w
    ]
    return result


# ---------------------------------------------------------------------------
# Test-time augmentation
# ---------------------------------------------------------------------------

def _predict_with_tta(
    model: UNet,
    tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Predict a single (1, 1, H, W) tensor with test-time augmentation.

    Averages softmax probabilities from original, horizontally flipped,
    and vertically flipped versions.

    Args:
        model: Trained UNet in eval mode.
        tensor: Preprocessed input tensor (1, 1, H, W).
        device: Torch device.

    Returns:
        Predicted class labels as (H, W) integer numpy array.
    """
    tensor = tensor.to(device)

    with torch.no_grad():
        # Original
        logits_orig = model(tensor)
        probs = F.softmax(logits_orig, dim=1)

        # Horizontal flip
        tensor_hflip = torch.flip(tensor, dims=[-1])
        logits_hflip = model(tensor_hflip)
        probs_hflip = torch.flip(F.softmax(logits_hflip, dim=1), dims=[-1])
        probs = probs + probs_hflip

        # Vertical flip
        tensor_vflip = torch.flip(tensor, dims=[-2])
        logits_vflip = model(tensor_vflip)
        probs_vflip = torch.flip(F.softmax(logits_vflip, dim=1), dims=[-2])
        probs = probs + probs_vflip

        probs = probs / 3.0

    pred = probs.argmax(dim=1).squeeze(0).cpu().numpy()
    return pred


def _predict_no_tta(
    model: UNet,
    tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Predict a single tensor without test-time augmentation."""
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
    return pred


# ---------------------------------------------------------------------------
# Sliding-window inference
# ---------------------------------------------------------------------------

def _sliding_window_predict(
    model: UNet,
    img_slice: np.ndarray,
    device: torch.device,
    crop_size: int = _DEFAULT_CROP_SIZE,
    stride: int | None = None,
    use_tta: bool = True,
) -> np.ndarray:
    """Predict a slice larger than *crop_size* using sliding windows.

    Overlapping windows are tiled across the image. Softmax probabilities are
    averaged in overlapping regions, then argmax produces the final labels.

    Args:
        model: Trained UNet in eval mode.
        img_slice: (H, W) grayscale image (unnormalized).
        device: Torch device.
        crop_size: Window size.
        stride: Step between windows. Defaults to ``crop_size // 2``.
        use_tta: Whether to apply test-time augmentation.

    Returns:
        (H, W) integer segmentation mask.
    """
    if stride is None:
        stride = crop_size // 2

    h, w = img_slice.shape
    normed = normalize_intensity(img_slice.astype(np.float32))

    # Pad to ensure full coverage
    pad_h = (crop_size - h % stride) % stride
    pad_w = (crop_size - w % stride) % stride
    padded = np.pad(normed, ((0, pad_h), (0, pad_w)), mode="constant")

    ph, pw = padded.shape
    num_classes = model.num_classes
    prob_sum = np.zeros((num_classes, ph, pw), dtype=np.float64)
    count = np.zeros((ph, pw), dtype=np.float64)

    predict_fn = _predict_with_tta if use_tta else _predict_no_tta

    for y in range(0, ph - crop_size + 1, stride):
        for x in range(0, pw - crop_size + 1, stride):
            patch = padded[y : y + crop_size, x : x + crop_size]
            tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float()
            tensor = tensor.to(device)

            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # (C, H, W)

            if use_tta:
                # Horizontal flip
                t_hf = torch.flip(tensor, dims=[-1])
                p_hf = F.softmax(model(t_hf), dim=1)
                p_hf = torch.flip(p_hf, dims=[-1]).squeeze(0).cpu().numpy()
                probs = probs + p_hf

                # Vertical flip
                t_vf = torch.flip(tensor, dims=[-2])
                p_vf = F.softmax(model(t_vf), dim=1)
                p_vf = torch.flip(p_vf, dims=[-2]).squeeze(0).cpu().numpy()
                probs = probs + p_vf

                probs = probs / 3.0

            prob_sum[:, y : y + crop_size, x : x + crop_size] += probs
            count[y : y + crop_size, x : x + crop_size] += 1.0

    # Average probabilities
    count = np.maximum(count, 1.0)
    prob_avg = prob_sum / count[np.newaxis, :, :]
    seg = prob_avg.argmax(axis=0)[:h, :w].astype(np.int64)
    return seg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_volume(
    model: UNet,
    volume: np.ndarray,
    spacing: tuple[float, ...] | None = None,
    device: str | torch.device = "cuda",
    crop_size: int = _DEFAULT_CROP_SIZE,
    use_tta: bool = True,
) -> np.ndarray:
    """Segment all slices of a cardiac volume.

    Args:
        model: Trained UNet (will be set to eval mode).
        volume: Cardiac image volume with shape:
            - (T, D, H, W) — 4D cine volume (time, depth, height, width)
            - (D, H, W) — 3D single time-frame
            - (H, W) — single 2D slice
        spacing: Pixel spacing for preprocessing. Currently unused but kept
            for API compatibility with future resampling.
        device: Torch device string or object.
        crop_size: Input crop size expected by the model.
        use_tta: Whether to apply test-time augmentation.

    Returns:
        Segmentation mask with same spatial dimensions as *volume*.
        Values in {0, 1, 2, 3} following ACDC convention.
    """
    model.eval()
    device = torch.device(device) if isinstance(device, str) else device

    if volume.ndim == 2:
        # Single 2D slice
        return _segment_2d_slice(model, volume, device, crop_size, use_tta)

    if volume.ndim == 3:
        # (D, H, W) — stack of slices
        seg = np.zeros_like(volume, dtype=np.int64)
        for d in range(volume.shape[0]):
            seg[d] = _segment_2d_slice(model, volume[d], device, crop_size, use_tta)
        return seg

    if volume.ndim == 4:
        # (T, D, H, W) — 4D cine
        seg = np.zeros_like(volume, dtype=np.int64)
        for t in range(volume.shape[0]):
            for d in range(volume.shape[1]):
                seg[t, d] = _segment_2d_slice(
                    model, volume[t, d], device, crop_size, use_tta
                )
        return seg

    raise ValueError(f"Unsupported volume shape: {volume.shape}")


def _segment_2d_slice(
    model: UNet,
    img_slice: np.ndarray,
    device: torch.device,
    crop_size: int,
    use_tta: bool,
) -> np.ndarray:
    """Segment a single 2D slice, using sliding window if needed."""
    h, w = img_slice.shape
    if h > crop_size or w > crop_size:
        return _sliding_window_predict(
            model, img_slice, device, crop_size=crop_size, use_tta=use_tta
        )

    tensor, original_hw, _ = _preprocess_slice(img_slice, crop_size)
    predict_fn = _predict_with_tta if use_tta else _predict_no_tta
    pred_crop = predict_fn(model, tensor, device)
    return _postprocess_mask(pred_crop, original_hw, crop_size)


def predict_study(
    model: UNet,
    study_path: str | Path,
    device: str | torch.device = "cuda",
    crop_size: int = _DEFAULT_CROP_SIZE,
    use_tta: bool = True,
) -> dict:
    """Full study prediction: segment ED and ES frames, compute volumes.

    Expects an ACDC-style patient directory containing NIfTI files.

    Args:
        model: Trained UNet.
        study_path: Path to the patient directory.
        device: Torch device.
        crop_size: Input crop size for the model.
        use_tta: Whether to apply test-time augmentation.

    Returns:
        Dictionary with keys:
            - ``ed_segmentation``: (D, H, W) segmentation at end-diastole.
            - ``es_segmentation``: (D, H, W) segmentation at end-systole.
            - ``volumes``: dict of computed volumes and EF.
            - ``ed_path``: path to the ED NIfTI file used.
            - ``es_path``: path to the ES NIfTI file used.
    """
    study_path = Path(study_path)
    device = torch.device(device) if isinstance(device, str) else device

    # Discover ED and ES frames
    frame_files = sorted(study_path.glob("*_frame??.nii.gz"))
    # Exclude ground-truth files
    frame_files = [f for f in frame_files if "_gt" not in f.stem]

    if len(frame_files) < 2:
        raise FileNotFoundError(
            f"Expected at least 2 frame files in {study_path}, found {len(frame_files)}"
        )

    # Load and segment all frames
    segmentations = {}
    for frame_path in frame_files:
        vol, meta = load_nifti(frame_path)
        spacing = meta.get("spacing", (1.0, 1.0, 1.0))
        seg = predict_volume(model, vol, spacing=spacing, device=device,
                             crop_size=crop_size, use_tta=use_tta)
        # Compute LV cavity volume (label 3)
        lv_voxels = int((seg == 3).sum())
        segmentations[frame_path] = {
            "seg": seg,
            "lv_voxels": lv_voxels,
            "spacing": spacing,
        }

    # ED = max LV volume, ES = min LV volume
    ed_path = max(segmentations, key=lambda p: segmentations[p]["lv_voxels"])
    es_path = min(segmentations, key=lambda p: segmentations[p]["lv_voxels"])

    ed_seg = segmentations[ed_path]["seg"]
    es_seg = segmentations[es_path]["seg"]
    ed_spacing = segmentations[ed_path]["spacing"]
    es_spacing = segmentations[es_path]["spacing"]

    # Compute volumes from ED and ES segmentations
    # Build a time-resolved stack with [ED, ES] for compute_volumes
    combined = np.stack([ed_seg, es_seg], axis=0)
    # Use ED spacing (should be the same for both frames)
    # spacing from SimpleITK is (x, y, z); we need spatial dims
    if len(ed_spacing) >= 3:
        vol_spacing = (ed_spacing[2], ed_spacing[1], ed_spacing[0])
    else:
        vol_spacing = tuple(ed_spacing)

    volumes = compute_volumes(combined, vol_spacing)

    return {
        "ed_segmentation": ed_seg,
        "es_segmentation": es_seg,
        "volumes": volumes,
        "ed_path": str(ed_path),
        "es_path": str(es_path),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line entry point for segmentation inference."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run cardiac segmentation inference")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--study", type=str, required=True, help="Path to patient study directory"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Torch device (cuda or cpu)"
    )
    parser.add_argument(
        "--no-tta", action="store_true", help="Disable test-time augmentation"
    )
    args = parser.parse_args()

    model = load_model(args.checkpoint, device=args.device)
    result = predict_study(
        model, args.study, device=args.device, use_tta=not args.no_tta
    )

    logger.info("Volumes: %s", result["volumes"])
    logger.info("ED path: %s", result["ed_path"])
    logger.info("ES path: %s", result["es_path"])


if __name__ == "__main__":
    main()
