# Agent: Phase 2 — Cardiac Segmentation

## Mission
Implement a complete training and inference pipeline for cardiac MRI segmentation (LV cavity, myocardium, RV cavity) using a U-Net architecture. This model segments the heart at every cardiac phase, providing contours for motion estimation and strain computation.

## Status: IN PROGRESS

## Files You Own
- `strain/models/segmentation/unet.py` — U-Net model (skeleton exists)
- `strain/models/segmentation/losses.py` — loss functions (skeleton exists)
- `strain/models/segmentation/train_seg.py` — training loop (CREATE)
- `strain/models/segmentation/predict_seg.py` — inference pipeline (CREATE)
- `strain/models/segmentation/metrics.py` — evaluation metrics (CREATE)
- `configs/segmentation.yaml` — training config (CREATE)
- `tests/test_segmentation.py` — unit tests (CREATE)

## Detailed Requirements

### 1. unet.py — Verify and improve existing model
- [x] `ConvBlock` — double conv + BN + ReLU
- [x] `ResBlock` — residual connection
- [x] `UNet(in_channels=1, num_classes=4, features=(64,128,256,512))`
- [ ] Add dropout (p=0.1) between encoder stages for regularization
- [ ] Add option for deep supervision (auxiliary loss at intermediate decoder stages)
- [ ] Verify forward pass works with 128×128 input: `assert UNet()(torch.randn(1,1,128,128)).shape == (1,4,128,128)`

### 2. losses.py — Verify existing
- [x] `DiceLoss` — soft Dice for multi-class
- [x] `DiceCELoss` — combined Dice + CE
- [ ] Verify gradient flow with a simple test case

### 3. metrics.py — CREATE
```python
def dice_score(pred, target, num_classes=4) -> dict[str, float]:
    """Per-class Dice scores. Returns {'LV': 0.95, 'Myo': 0.88, 'RV': 0.91}"""

def hausdorff_distance_95(pred, target) -> dict[str, float]:
    """95th percentile Hausdorff distance per class in mm."""

def compute_volumes(segmentation, spacing) -> dict[str, float]:
    """Compute LV-EDV, LV-ESV, LVEF, RV volumes from segmentation."""
```
- Labels: 0=background, 1=RV cavity, 2=myocardium, 3=LV cavity (ACDC convention)
- Use scipy.spatial.distance for Hausdorff computation
- Volume = voxel_count × voxel_volume (from spacing)

### 4. train_seg.py — CREATE
Full training loop with:
```python
def train(config_path: str):
    # 1. Load config
    # 2. Create datasets and dataloaders
    # 3. Initialize model, optimizer, scheduler, loss
    # 4. Training loop:
    #    - Forward pass, compute loss
    #    - Backward pass, optimizer step
    #    - Validation every N epochs
    #    - Save best model (by val Dice)
    #    - Early stopping (patience=20)
    #    - Log to TensorBoard
    # 5. Final evaluation on test set
```

Training hyperparameters:
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | CosineAnnealingWarmRestarts (T_0=50, T_mult=2) |
| Loss | DiceCELoss(dice_weight=0.5, ce_weight=0.5) |
| Batch size | 16 |
| Max epochs | 200 |
| Early stopping | patience=20 on val Dice |
| Input size | 128×128 |

### 5. predict_seg.py — CREATE
```python
def predict_volume(model, volume, spacing, device="cuda") -> np.ndarray:
    """Segment all slices of a cardiac volume.

    Args:
        model: Trained UNet
        volume: (T, D, H, W) or (D, H, W) cine volume
        spacing: pixel spacing for preprocessing
        device: torch device

    Returns:
        Segmentation mask with same spatial dims as input
    """
    # 1. Preprocess each slice (normalize, crop)
    # 2. Run model inference
    # 3. Argmax to get class labels
    # 4. Resize back to original dimensions
    # 5. Return segmentation volume

def predict_study(model, study_path, device="cuda") -> dict:
    """Full study prediction: segment ED and ES frames, compute volumes."""
```
- Add test-time augmentation (TTA): average predictions from original + flipped
- Add sliding-window for images larger than 128×128

### 6. configs/segmentation.yaml
```yaml
model:
  architecture: "unet"
  in_channels: 1
  num_classes: 4
  features: [64, 128, 256, 512]
  dropout: 0.1

training:
  optimizer: "adamw"
  learning_rate: 0.001
  weight_decay: 0.0001
  scheduler: "cosine_warm_restarts"
  scheduler_T0: 50
  scheduler_Tmult: 2
  loss: "dice_ce"
  max_epochs: 200
  early_stopping_patience: 20
  batch_size: 16

inference:
  tta: true
  device: "cuda"

paths:
  data_root: "./data/ACDC"
  checkpoint_dir: "./checkpoints/segmentation"
  log_dir: "./logs/segmentation"
```

### 7. tests/test_segmentation.py
- Test UNet forward pass shape: input (B,1,128,128) → output (B,4,128,128)
- Test DiceLoss returns scalar in [0, 1]
- Test DiceCELoss gradients flow
- Test dice_score with known inputs (perfect prediction → 1.0)
- Test compute_volumes produces reasonable values

## Interface Contract
```python
# Input from Phase 1:
dataset[i] -> {"image": Tensor(1, 128, 128), "label": Tensor(128, 128)}

# Output to Phase 3 (Motion) and Phase 4 (Strain):
segmentation_mask: np.ndarray  # shape (T, H, W), values in {0, 1, 2, 3}
# 0=bg, 1=RV, 2=myocardium, 3=LV cavity

# Output to Phase 5 (Risk):
volumes: {"LV_EDV": float, "LV_ESV": float, "LVEF": float, "RV_EDV": float}
```

## If You Get Stuck
- ACDC label convention: 0=bg, 1=RV, 2=myocardium, 3=LV
- For Hausdorff distance, extract surface points with `skimage.measure.find_contours` or binary erosion
- If Dice is stuck at ~0.5, check label encoding (one-hot vs integer), learning rate, and loss weighting
- Use `torch.cuda.amp` for mixed-precision training if GPU memory is an issue
- TensorBoard: `from torch.utils.tensorboard import SummaryWriter`
