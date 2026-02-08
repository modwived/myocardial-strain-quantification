# Agent: Phase 1 — Data Pipeline

## Mission
Implement a production-quality data ingestion, preprocessing, and PyTorch data loading pipeline for cardiac MRI (cine bSSFP). This is the foundation that all downstream phases depend on.

## Status: IN PROGRESS

## Files You Own
- `strain/data/loader.py` — DICOM + NIfTI loading (skeleton exists)
- `strain/data/preprocessing.py` — normalization, resampling, cropping (skeleton exists)
- `strain/data/augmentation.py` — training-time transforms (skeleton exists)
- `strain/data/dataset.py` — PyTorch Dataset classes (skeleton exists)
- `tests/test_data_pipeline.py` — unit tests (CREATE)
- `configs/data.yaml` — data configuration (CREATE)

## Detailed Requirements

### 1. loader.py — Enhance the existing skeleton
- [x] `load_nifti(path)` — loads NIfTI, returns (array, metadata)
- [x] `load_dicom_series(directory)` — loads DICOM series
- [ ] Add `load_study(path)` — auto-detect format (DICOM dir vs NIfTI file) and dispatch
- [ ] Add `extract_cardiac_metadata(path)` — extract heart rate, trigger time, slice location, number of phases from DICOM headers
- [ ] Handle 2D+time (short-axis stack) and 3D+time volumes
- [ ] Handle multi-slice DICOM series (group by slice location, sort by trigger time)
- [ ] Robust error handling: missing files, corrupt DICOM, unsupported formats

### 2. preprocessing.py — Enhance the existing skeleton
- [x] `normalize_intensity(image)` — z-score normalization
- [x] `resample_image(image, current_spacing, target_spacing)` — spatial resampling
- [x] `center_crop(image, crop_size)` — center crop
- [ ] Add `detect_lv_center(image)` — detect LV center using image moments or intensity-based heuristic for smarter cropping
- [ ] Add `crop_around_lv(image, center, crop_size)` — crop centered on detected LV
- [ ] Add `build_preprocessing_pipeline(config)` — composable pipeline from config
- [ ] Ensure all functions handle both 3D (D,H,W) and 4D (T,D,H,W) arrays

### 3. augmentation.py — Enhance the existing skeleton
- [x] `random_rotation(image, mask, max_angle)`
- [x] `random_scale(image, mask, scale_range)`
- [x] `random_gamma(image, gamma_range)`
- [ ] Add `random_elastic_deformation(image, mask, alpha, sigma)` — B-spline or thin-plate-spline elastic deformation
- [ ] Add `random_flip(image, mask, axis)` — horizontal/vertical flip
- [ ] Add `random_noise(image, std)` — additive Gaussian noise
- [ ] Add `Compose` class — chain multiple augmentations with per-transform probability
- [ ] Ensure all augmentations jointly transform image AND mask with correct interpolation (bilinear for image, nearest for mask)

### 4. dataset.py — Enhance the existing skeleton
- [x] `SegmentationDataset` — loads ACDC-style data
- [x] `MotionDataset` — loads 4D cine for frame pairs
- [ ] Improve `SegmentationDataset` to iterate over ALL slices, not just mid-slice
- [ ] Add `split_dataset(root, train_ratio, val_ratio, seed)` — deterministic stratified splitting
- [ ] Add `get_dataloaders(config)` — returns train/val/test DataLoaders with appropriate augmentation
- [ ] Add support for M&Ms dataset format (different directory structure)
- [ ] Add data caching option (preload to RAM for faster training)

### 5. configs/data.yaml
```yaml
dataset:
  name: "acdc"
  root: "./data/ACDC"
  format: "nifti"  # or "dicom"

preprocessing:
  target_spacing: [1.25, 1.25]
  crop_size: 128
  normalize: "zscore"

augmentation:
  enabled: true
  rotation_max: 15.0
  scale_range: [0.9, 1.1]
  gamma_range: [0.7, 1.5]
  elastic_alpha: 100.0
  elastic_sigma: 10.0
  noise_std: 0.02
  flip_horizontal: true

dataloader:
  batch_size: 16
  num_workers: 4
  pin_memory: true

split:
  train: 0.70
  val: 0.15
  test: 0.15
  seed: 42
```

### 6. tests/test_data_pipeline.py
- Test `load_nifti` with a synthetic NIfTI file (create using SimpleITK)
- Test `normalize_intensity` produces zero mean / unit variance
- Test `resample_image` changes spatial dimensions correctly
- Test `center_crop` output shape
- Test augmentations preserve shape and mask integrity
- Test `SegmentationDataset.__getitem__` returns correct tensor shapes
- Test `MotionDataset.__getitem__` returns source/target pairs

## Interface Contract (for downstream phases)
Other phases depend on these interfaces:

```python
# Segmentation training (Phase 2) expects:
dataset[i] -> {"image": Tensor(1, 128, 128), "label": Tensor(128, 128)}

# Motion estimation (Phase 3) expects:
dataset[i] -> {"source": Tensor(1, 128, 128), "target": Tensor(1, 128, 128)}

# Preprocessing (all phases) expects:
preprocess(raw_array, spacing) -> normalized_cropped_array
```

## If You Get Stuck
- SimpleITK docs: https://simpleitk.readthedocs.io/
- pydicom docs: https://pydicom.github.io/pydicom/
- ACDC dataset format: patient folders with `_4d.nii.gz`, `_frameXX.nii.gz`, `_frameXX_gt.nii.gz`
- For elastic deformation, use `scipy.ndimage.map_coordinates` with random B-spline control points
- For synthetic test data, use `SimpleITK.GetImageFromArray()` to create NIfTI in-memory
