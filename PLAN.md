# Implementation Plan: AI-Based Myocardial Strain Quantification

## Overview

Build an end-to-end deep learning pipeline that takes standard cine cardiac MRI (bSSFP)
as input and outputs global and segmental myocardial strain values (GLS, GCS, GRS) for
risk stratification following acute myocardial infarction (AMI).

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        INPUT: Cine CMR (DICOM/NIfTI)                │
└──────────────┬───────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Stage 1: Preprocessing      │
│  - DICOM/NIfTI loading       │
│  - ROI detection & cropping  │
│  - Intensity normalization   │
│  - Spatial resampling        │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Stage 2: Segmentation       │
│  - LV endocardium            │
│  - LV epicardium (myocardium)│
│  - RV endocardium            │
│  - All cardiac phases        │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Stage 3: Motion Estimation  │
│  - Frame-to-frame registration│
│  - Dense displacement fields │
│  - Temporal consistency      │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Stage 4: Strain Computation │
│  - Deformation gradient (F)  │
│  - Green-Lagrange tensor (E) │
│  - GLS, GCS, GRS extraction  │
│  - AHA 16-segment mapping    │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Stage 5: Risk Stratification│
│  - Feature aggregation       │
│  - Classification/regression │
│  - MACE prediction           │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  OUTPUT: Strain values (global + segmental) + risk score            │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Data Pipeline & Preprocessing

**Goal:** Robust ingestion and normalization of cardiac MRI data.

### Module: `strain/data/`

| File | Responsibility |
|------|---------------|
| `loader.py` | Load DICOM series and NIfTI volumes with metadata extraction |
| `preprocessing.py` | Intensity normalization, spatial resampling, ROI cropping |
| `augmentation.py` | Training-time transforms (rotation, scaling, elastic deform, gamma) |
| `dataset.py` | PyTorch Dataset classes for segmentation and motion tasks |

### Key Decisions

- **Input format:** Support both DICOM (clinical) and NIfTI (research/ACDC)
- **Resampling:** Resample all images to a common in-plane resolution (1.25 × 1.25 mm)
- **Normalization:** Per-volume z-score normalization (zero mean, unit variance)
- **ROI detection:** Crop 128×128 patch centered on LV using a lightweight CNN or heuristic based on image moments
- **Augmentation:** Random rotation (±15°), scaling (0.9–1.1), elastic deformation, intensity gamma (0.7–1.5)

### Public Datasets

| Dataset | Subjects | Use |
|---------|----------|-----|
| **ACDC** | 150 (5 pathology groups) | Primary development & benchmarking |
| **M&Ms** | 375 (4 vendors, 6 centers) | Multi-vendor generalization testing |
| **M&Ms-2** | 360 | RV segmentation validation |

### Deliverables
- [ ] DICOM and NIfTI loader with metadata extraction
- [ ] Preprocessing pipeline with configurable parameters
- [ ] PyTorch Dataset with on-the-fly augmentation
- [ ] Data split strategy (70/15/15 train/val/test, stratified by pathology)
- [ ] Unit tests for loaders and transforms

---

## Phase 2: Myocardial Segmentation

**Goal:** Pixel-level segmentation of LV cavity, myocardium, and RV across all cardiac phases.

### Module: `strain/models/segmentation/`

| File | Responsibility |
|------|---------------|
| `unet.py` | 2D U-Net backbone with residual blocks |
| `swin_unet.py` | Hybrid CNN-Swin Transformer variant |
| `losses.py` | Dice loss, cross-entropy, combined loss functions |
| `train_seg.py` | Training loop for segmentation models |
| `predict_seg.py` | Inference with sliding-window and test-time augmentation |

### Model Architecture

**Primary:** 2D U-Net with ResNet-34 encoder (pretrained on ImageNet)

```
Encoder: ResNet-34 (4 stages: 64→128→256→512 channels)
    │
Bottleneck: 512 channels
    │
Decoder: 4 upsampling stages with skip connections
    │
Output: 4-class softmax (background, LV cavity, myocardium, RV cavity)
```

**Advanced option:** Hybrid CNN-Swin Transformer (CardSegNet-style)
- CNN encoder for local features in shallow layers
- Swin Transformer blocks in deeper layers for global context
- Adaptive attention fusion at skip connections

### Training Strategy

| Parameter | Value |
|-----------|-------|
| Loss | Dice + Cross-Entropy (equal weight) |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | Cosine annealing with warm restarts |
| Batch size | 16 |
| Epochs | 200 (early stopping, patience=20) |
| Input size | 128 × 128 × 1 (single-channel cine frame) |

### Evaluation Metrics

| Metric | Target |
|--------|--------|
| Dice (LV cavity) | ≥ 0.93 |
| Dice (Myocardium) | ≥ 0.87 |
| Dice (RV cavity) | ≥ 0.90 |
| Hausdorff 95th (mm) | ≤ 5.0 |

### Deliverables
- [ ] U-Net with ResNet encoder
- [ ] Dice + CE combined loss
- [ ] Training loop with logging (TensorBoard/W&B)
- [ ] Evaluation script with Dice, Hausdorff, volume metrics
- [ ] Model checkpointing and best-model selection
- [ ] Optional: Swin Transformer hybrid model

---

## Phase 3: Motion Estimation

**Goal:** Estimate dense displacement fields between consecutive cardiac phases.

### Module: `strain/models/motion/`

| File | Responsibility |
|------|---------------|
| `carmen.py` | Cardiac Motion Estimation Network (unsupervised registration) |
| `losses.py` | NCC / SSIM similarity + deformation regularization losses |
| `warp.py` | Spatial transformer for image warping |
| `train_motion.py` | Training loop for motion estimation |
| `predict_motion.py` | Inference: generate displacement fields for full cardiac cycle |

### Model Architecture

**Unsupervised registration network** (VoxelMorph / CarMEN-inspired):

```
Input: Pair of consecutive frames (I_t, I_{t+1}), each 128×128
    │
    ▼
Siamese Encoder (shared weights, 4 stages)
    │
    ▼
Concatenate encoded features
    │
    ▼
Decoder → Dense displacement field φ (128×128×2)
    │
    ▼
Spatial Transformer: I_t ∘ φ ≈ I_{t+1}
```

### Training Strategy (Unsupervised)

| Parameter | Value |
|-----------|-------|
| Similarity loss | Normalized Cross-Correlation (NCC) |
| Regularization | Diffusion regularizer on displacement field (λ=1.0) |
| Optimizer | Adam (lr=1e-4) |
| Epochs | 300 |
| Pair sampling | Consecutive frames + skip-1 frames for robustness |

### Optional: Supervised Approach (StrainNet-style)

If DENSE displacement ground truth is available:
- Train a 3D U-Net to predict intramyocardial displacement from segmentation contours
- Loss: MSE between predicted and DENSE-derived displacements
- Advantage: Better intramyocardial motion accuracy

### Deliverables
- [ ] Registration network with spatial transformer
- [ ] NCC + regularization loss
- [ ] Training loop (unsupervised, frame pairs)
- [ ] Displacement field visualization tools
- [ ] Warped-image quality checks (SSIM, visual inspection)

---

## Phase 4: Strain Computation

**Goal:** Convert displacement fields into clinically meaningful strain values.

### Module: `strain/mechanics/`

| File | Responsibility |
|------|---------------|
| `deformation.py` | Compute deformation gradient tensor F from displacement fields |
| `strain_tensor.py` | Green-Lagrange strain tensor E = 0.5(F^T F - I) |
| `coordinate_system.py` | Define radial/circumferential/longitudinal axes from myocardial geometry |
| `global_strain.py` | Compute GLS, GCS, GRS by averaging over myocardium |
| `segmental_strain.py` | AHA 16-segment model mapping and per-segment strain |
| `strain_rate.py` | Temporal derivative of strain curves |

### Strain Computation Pipeline

```
Displacement field φ(x,y) at each time point
    │
    ▼
Deformation gradient: F = I + ∇φ  (spatial gradient of displacement)
    │
    ▼
Right Cauchy-Green tensor: C = F^T · F
    │
    ▼
Green-Lagrange strain: E = 0.5 · (C - I)
    │
    ▼
Project onto cardiac coordinates:
  - ê_c (circumferential): tangent to myocardial contour
  - ê_r (radial): normal to myocardial contour (pointing outward)
  - ê_l (longitudinal): base-to-apex direction
    │
    ▼
E_cc → Circumferential strain (GCS)
E_rr → Radial strain (GRS)
E_ll → Longitudinal strain (GLS, from long-axis views)
    │
    ▼
Average over myocardial mask → Global values
Map to AHA segments → Segmental values
```

### Normal Reference Ranges

| Parameter | Normal (CMR-FT) | Abnormal threshold |
|-----------|------------------|--------------------|
| GLS | -20.1 ± 3.2% | > -16% |
| GCS | -23.0 ± 4.8% | > -17% |
| GRS | +34.1 ± 12.6% | < +20% |

### Deliverables
- [ ] Deformation gradient computation from displacement fields
- [ ] Green-Lagrange strain tensor
- [ ] Cardiac coordinate system definition
- [ ] GLS, GCS, GRS computation
- [ ] AHA 16-segment mapping
- [ ] Strain-time curve generation
- [ ] Validation against published normal ranges

---

## Phase 5: Risk Stratification

**Goal:** Predict major adverse cardiovascular events (MACE) from strain + clinical features.

### Module: `strain/risk/`

| File | Responsibility |
|------|---------------|
| `features.py` | Extract features: strain values + LVEF + clinical variables |
| `model.py` | Risk prediction model (logistic regression / gradient boosting / survival) |
| `train_risk.py` | Training with cross-validation |
| `predict_risk.py` | Inference: risk score output |

### Features for Risk Model

| Category | Features |
|----------|----------|
| **Strain** | GLS, GCS, GRS (global); per-segment strain; peak strain rate |
| **Functional** | LVEF, RVEF, LV volumes (EDV, ESV), cardiac output |
| **Clinical** | Age, sex, diabetes, hypertension, Killip class, TIMI flow |
| **Imaging** | Infarct size (if LGE available), microvascular obstruction |

### Model Options

1. **Cox proportional hazards** — interpretable, standard for survival analysis
2. **Gradient boosting (XGBoost)** — handles nonlinear interactions, feature importance
3. **Deep survival model** — DeepSurv for complex feature interactions

### Evaluation

| Metric | Description |
|--------|-------------|
| C-index | Concordance index for survival models |
| AUC-ROC | For binary MACE classification |
| Calibration | Hosmer-Lemeshow or calibration plots |
| NRI/IDI | Net reclassification improvement over LVEF-only model |

### Deliverables
- [ ] Feature extraction from strain pipeline output
- [ ] Baseline model: GLS-threshold classifier
- [ ] Advanced model: Cox or gradient-boosted survival model
- [ ] Cross-validation evaluation framework
- [ ] Risk score calibration and interpretation

---

## Phase 6: API & Deployment

**Goal:** Serve the complete pipeline via a REST API.

### Module: `app/`

| File | Responsibility |
|------|---------------|
| `main.py` | FastAPI application, CORS, lifespan management |
| `routes/analysis.py` | POST /analyze — upload study, return strain + risk |
| `routes/health.py` | GET /health — readiness and liveness probes |
| `schemas.py` | Pydantic models for request/response validation |
| `services.py` | Orchestrate the full pipeline (preprocess → segment → motion → strain → risk) |

### API Endpoints

```
POST /analyze
  Input:  Multipart upload (DICOM zip or NIfTI file)
  Output: {
    "global_strain": { "GLS": -18.2, "GCS": -21.5, "GRS": 30.1 },
    "segmental_strain": { "1": {...}, ... "16": {...} },
    "risk_score": 0.34,
    "risk_category": "intermediate",
    "processing_time_sec": 12.5
  }

GET /health
  Output: { "status": "ok", "gpu_available": true, "model_loaded": true }
```

### Deployment Options

| Method | Use Case |
|--------|----------|
| `docker compose up` | Single-server deployment |
| Kubernetes (Helm chart) | Scalable production deployment |
| AWS SageMaker / GCP Vertex AI | Managed ML inference |

### Deliverables
- [ ] FastAPI application with /analyze and /health endpoints
- [ ] DICOM/NIfTI upload handling
- [ ] Pipeline orchestration service
- [ ] Dockerfile and docker-compose.yml (already created)
- [ ] Logging, error handling, request tracing

---

## Implementation Timeline

| Phase | Description | Estimated Effort | Dependencies |
|-------|-------------|------------------|-------------|
| **1** | Data pipeline & preprocessing | 2 weeks | Dataset access |
| **2** | Segmentation model | 3 weeks | Phase 1 |
| **3** | Motion estimation | 3 weeks | Phase 1 |
| **4** | Strain computation | 2 weeks | Phases 2 + 3 |
| **5** | Risk stratification | 2 weeks | Phase 4 |
| **6** | API & deployment | 1 week | Phase 4 (minimum) |

Phases 2 and 3 can be developed **in parallel** after Phase 1 is complete.

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Deep learning | PyTorch 2.x |
| Medical imaging | SimpleITK, pydicom, nibabel |
| Image processing | OpenCV, scikit-image |
| API framework | FastAPI + Uvicorn |
| Risk modeling | scikit-learn, lifelines (survival), XGBoost |
| Experiment tracking | Weights & Biases or TensorBoard |
| Containerization | Docker + Docker Compose |
| Testing | pytest |
| Linting | ruff |

---

## Key References

1. **DeepStrain** (Morales et al., 2021) — End-to-end DL strain from cine MRI
2. **StrainNet** (Ghadimi et al., 2023) — DENSE-supervised displacement prediction
3. **AI-Automated Strain for Post-AMI Risk** (Eitel et al., 2022) — GLS as independent MACE predictor (HR 1.10)
4. **CMR Strain and Mortality** (Aung et al., 2024) — 45,700 subjects, GLS predicts death (HR 1.18)
5. **ACDC Challenge** — Standard segmentation benchmark
6. **M&Ms Challenge** — Multi-vendor generalization benchmark
