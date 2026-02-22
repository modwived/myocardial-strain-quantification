# Agent: Phase 3 — Motion Estimation

## Mission
Implement an unsupervised cardiac motion estimation network that predicts dense displacement fields between consecutive cine MRI frames. This displacement field is the input to strain computation (Phase 4).

## Status: IN PROGRESS

## Files You Own
- `strain/models/motion/carmen.py` — CarMEN network (skeleton exists)
- `strain/models/motion/warp.py` — spatial transformer (skeleton exists)
- `strain/models/motion/losses.py` — NCC + regularization (skeleton exists)
- `strain/models/motion/train_motion.py` — training loop (CREATE)
- `strain/models/motion/predict_motion.py` — inference: full cardiac cycle (CREATE)
- `configs/motion.yaml` — training config (CREATE)
- `tests/test_motion.py` — unit tests (CREATE)

## Detailed Requirements

### 1. carmen.py — Verify and improve
- [x] `MotionEncoder` — shared encoder for source/target
- [x] `MotionDecoder` — decoder predicting displacement field
- [x] `CarMEN` — full model: source + target → displacement (B,2,H,W)
- [ ] Verify output resolution matches input (128×128)
- [ ] Add multi-scale prediction: output displacement at 3 scales (1/4, 1/2, full) for coarse-to-fine estimation
- [ ] Add optional segmentation-guided attention: weight displacement field by myocardial mask
- [ ] Verify flow head zero-initialization (identity transform at start)

### 2. warp.py — Verify
- [x] `spatial_transform(source, displacement)` — differentiable warping
- [ ] Verify: warping with zero displacement returns source unchanged
- [ ] Verify: warping is differentiable (gradients flow through)
- [ ] Add `compose_displacements(d1, d2)` — compose two displacement fields for multi-step tracking

### 3. losses.py — Verify and improve
- [x] `NCC` — normalized cross-correlation loss
- [x] `DiffusionRegularizer` — smoothness penalty on displacement gradients
- [ ] Add `BendingEnergy` — second-order regularizer for smoother fields
- [ ] Add `CyclicConsistencyLoss` — forward-backward consistency: φ_AB + φ_BA ≈ 0
- [ ] Verify NCC is 0 for identical images, >0 for different images

### 4. train_motion.py — CREATE
```python
def train(config_path: str):
    # 1. Load config
    # 2. Create MotionDataset + dataloaders
    # 3. Initialize CarMEN, optimizer, scheduler
    # 4. Training loop (unsupervised):
    #    for source, target in dataloader:
    #        displacement = model(source, target)
    #        warped = spatial_transform(source, displacement)
    #        loss_sim = ncc(warped, target)
    #        loss_reg = lambda_reg * regularizer(displacement)
    #        loss = loss_sim + loss_reg
    #        loss.backward()
    #        optimizer.step()
    #
    #    Validation: compute SSIM between warped and target
    #    Save best model by val SSIM
    #    Log displacement field visualizations to TensorBoard
```

Training hyperparameters:
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=1e-4) |
| Scheduler | StepLR (step_size=100, gamma=0.5) |
| Similarity loss | NCC (window_size=9) |
| Regularization | Diffusion (λ=1.0) |
| Batch size | 8 |
| Max epochs | 300 |
| Frame pairs | Consecutive + skip-1 (Δt=1 and Δt=2) |

### 5. predict_motion.py — CREATE
```python
def predict_cardiac_cycle(model, cine_volume, device="cuda") -> np.ndarray:
    """Predict displacement fields for full cardiac cycle.

    Args:
        model: Trained CarMEN
        cine_volume: (T, H, W) cine frames (preprocessed)
        device: torch device

    Returns:
        displacements: (T-1, 2, H, W) displacement field for each frame transition
    """

def compute_cumulative_displacement(displacements) -> np.ndarray:
    """Accumulate frame-to-frame displacements into ED-referenced displacements.

    Args:
        displacements: (T-1, 2, H, W) frame-to-frame displacements

    Returns:
        cumulative: (T, 2, H, W) displacement from ED (frame 0) to each frame
        cumulative[0] = zero field (ED to ED)
        cumulative[t] = sum of displacements 0→1→...→t
    """

def track_points(points, displacements) -> np.ndarray:
    """Track a set of points through the cardiac cycle.

    Args:
        points: (N, 2) initial point positions at ED
        displacements: (T, 2, H, W) cumulative displacement fields

    Returns:
        trajectories: (T, N, 2) point positions at each time frame
    """
```

### 6. configs/motion.yaml
```yaml
model:
  architecture: "carmen"
  in_channels: 1
  features: [16, 32, 64, 128]

training:
  optimizer: "adam"
  learning_rate: 0.0001
  scheduler: "step"
  scheduler_step: 100
  scheduler_gamma: 0.5
  similarity_loss: "ncc"
  ncc_window: 9
  regularization: "diffusion"
  reg_weight: 1.0
  max_epochs: 300
  batch_size: 8
  frame_skip: [1, 2]

inference:
  device: "cuda"
  accumulate: true

paths:
  data_root: "./data/ACDC"
  checkpoint_dir: "./checkpoints/motion"
  log_dir: "./logs/motion"
```

### 7. tests/test_motion.py
- Test CarMEN forward: input (B,1,128,128) × 2 → output (B,2,128,128)
- Test spatial_transform with zero displacement → output equals source
- Test spatial_transform with known translation → shifted image
- Test NCC loss: identical images → loss ≈ 0
- Test DiffusionRegularizer: constant field → loss = 0, noisy field → loss > 0
- Test compose_displacements: identity + d = d
- Test cumulative displacement: sum of zero fields = zero

## Interface Contract
```python
# Input from Phase 1:
motion_dataset[i] -> {"source": Tensor(1,128,128), "target": Tensor(1,128,128)}

# Output to Phase 4 (Strain):
displacement_field: np.ndarray  # shape (T-1, 2, H, W) or (T, 2, H, W) cumulative
# displacement[t, 0] = dx, displacement[t, 1] = dy (in pixels)

# Phase 2 segmentation masks can be used to:
# 1. Mask the loss (only penalize warping error in cardiac region)
# 2. Provide attention guidance
# 3. Initialize point tracking on myocardial contours
```

## If You Get Stuck
- If warped images look wrong, check displacement field scale (should be in pixels, not normalized)
- If displacement is all zeros, check flow head initialization and learning rate
- If displacement is too noisy, increase regularization weight (λ)
- Spatial transformer expects (B,2,H,W) displacement where dim 0=x, dim 1=y
- For cyclic consistency: train forward AND backward models, penalize φ_AB + warp(φ_BA, φ_AB)
- VoxelMorph paper (Balakrishnan 2019) is the key reference for unsupervised registration
- Compose displacements: φ_total(x) = φ_2(x + φ_1(x)) → use spatial_transform on displacement field itself
