# Agent: Phase 5 — Risk Stratification

## Mission
Implement a clinical risk prediction module that uses strain values (plus optional clinical variables) to stratify patients by MACE (Major Adverse Cardiovascular Events) risk after acute myocardial infarction.

## Status: IN PROGRESS

## Files You Own
- `strain/risk/features.py` — feature extraction (skeleton exists)
- `strain/risk/model.py` — risk models (skeleton exists)
- `strain/risk/train_risk.py` — model training with cross-validation (CREATE)
- `strain/risk/evaluate.py` — evaluation metrics for risk models (CREATE)
- `strain/risk/report.py` — generate clinical report from results (CREATE)
- `configs/risk.yaml` — risk model config (CREATE)
- `tests/test_risk.py` — unit tests (CREATE)

## Detailed Requirements

### 1. features.py — Improve existing
- [x] `extract_strain_features(global_strain, segmental_strain, lv_volumes)`
- [ ] Add strain rate features: peak systolic SR, peak diastolic SR
- [ ] Add strain heterogeneity features: coefficient of variation across segments
- [ ] Add wall motion abnormality score from segmental strain
- [ ] Add temporal features: time to peak strain, strain recovery ratio
- [ ] Add `create_feature_matrix(patients_data)` — convert list of patient results to (N, D) matrix

```python
def extract_strain_features(strain_results: dict, clinical_data: dict | None = None) -> dict[str, float]:
    """Full feature extraction.

    Strain features:
      - gls, gcs, grs (global peak values)
      - gcs_std, gcs_min, gcs_max (segmental heterogeneity)
      - peak_systolic_sr, peak_diastolic_sr
      - time_to_peak_strain (ms)
      - strain_recovery_ratio (end-diastolic / peak strain)
      - wall_motion_score (count of segments with |GCS| < 10%)

    Functional features (from segmentation volumes):
      - lvef, edv, esv, sv, rv_edv

    Clinical features (if provided):
      - age, sex, diabetes, hypertension, killip_class, timi_flow, troponin
    """
```

### 2. model.py — Improve existing
- [x] `GlsThresholdClassifier` — simple threshold (GLS > -16%)
- [x] `StrainRiskModel` — placeholder for ML model
- [ ] Add `CoxSurvivalModel` — Cox proportional hazards for time-to-event MACE prediction
- [ ] Add `GradientBoostingRiskModel` — XGBoost for binary MACE classification
- [ ] Add `MultiParameterThresholdModel` — combined GLS + GCS + LVEF thresholds
- [ ] Add model serialization: save/load with metadata (feature names, normalization params)

```python
class CoxSurvivalModel:
    """Cox proportional hazards model for MACE survival analysis.
    Uses lifelines library.
    """
    def __init__(self):
        self.model = None
        self.feature_names = None

    def fit(self, features_df, duration_col, event_col):
        """Fit Cox model. features_df includes duration and event columns."""

    def predict_risk(self, features: dict) -> dict:
        """Returns {'risk_score': float, 'survival_probability': dict, 'risk_category': str}"""

    def concordance_index(self, features_df, duration_col, event_col) -> float:
        """Compute Harrell's C-index."""

class GradientBoostingRiskModel:
    """XGBoost classifier for binary MACE prediction."""
    def __init__(self, n_estimators=100, max_depth=4, learning_rate=0.1):
        ...

    def fit(self, X, y):
        """Train with built-in cross-validation."""

    def predict(self, features: dict) -> dict:
        """Returns {'risk_score': float, 'risk_category': str, 'feature_importance': dict}"""
```

### 3. train_risk.py — CREATE
```python
def train_risk_model(config_path: str):
    """Train and evaluate risk prediction model.

    Pipeline:
    1. Load strain analysis results for all patients
    2. Extract features
    3. Handle missing values (imputation or exclusion)
    4. Normalize features (StandardScaler, saved for inference)
    5. Train with nested cross-validation:
       - Outer loop: 5-fold for unbiased performance estimate
       - Inner loop: 5-fold for hyperparameter tuning
    6. Evaluate: C-index, AUC-ROC, calibration, NRI
    7. Save best model + scaler + feature names
    """

def compare_models(config_path: str):
    """Compare GLS-threshold vs Cox vs XGBoost models.

    Output: table of C-index, AUC, NRI for each model.
    Determines if ML model adds value over simple GLS threshold.
    """
```

### 4. evaluate.py — CREATE
```python
def compute_auc_roc(y_true, y_score) -> tuple[float, np.ndarray, np.ndarray]:
    """ROC AUC with confidence interval (bootstrap)."""

def compute_concordance_index(event_times, predicted_scores, event_observed) -> float:
    """Harrell's C-index for survival models."""

def compute_nri(y_true, risk_old, risk_new, threshold=0.2) -> dict:
    """Net Reclassification Improvement: does new model improve over old?
    Returns {'nri': float, 'event_nri': float, 'nonevent_nri': float, 'p_value': float}
    """

def compute_calibration(y_true, y_pred, n_bins=10) -> dict:
    """Hosmer-Lemeshow calibration: predicted vs observed risk per decile.
    Returns {'calibration_slope': float, 'calibration_intercept': float, 'hl_statistic': float}
    """

def bootstrap_ci(metric_func, y_true, y_pred, n_bootstrap=1000, ci=0.95):
    """Bootstrap confidence interval for any metric."""
```

### 5. report.py — CREATE
```python
def generate_clinical_report(
    patient_id: str,
    strain_results: dict,
    risk_prediction: dict,
    volumes: dict | None = None,
) -> dict:
    """Generate a structured clinical report.

    Returns:
    {
        "patient_id": "P001",
        "timestamp": "2024-01-15T10:30:00",
        "strain_analysis": {
            "global": {"GLS": -18.2, "GCS": -21.5, "GRS": 30.1},
            "interpretation": "Mildly reduced circumferential strain",
            "abnormal_segments": [3, 4, 9, 10],  # inferoseptal + inferior
        },
        "risk_assessment": {
            "score": 0.34,
            "category": "intermediate",
            "interpretation": "Intermediate risk of MACE within 12 months",
            "contributing_factors": ["Reduced GLS", "Inferoseptal wall motion abnormality"],
        },
        "comparison_to_normal": {
            "GLS": {"value": -18.2, "normal_range": "-20.1 ± 3.2%", "status": "borderline"},
            "GCS": {"value": -21.5, "normal_range": "-23.0 ± 4.8%", "status": "normal"},
            "GRS": {"value": 30.1, "normal_range": "34.1 ± 12.6%", "status": "normal"},
        },
    }
    """

def interpret_strain(strain_results: dict) -> str:
    """Generate human-readable interpretation of strain values."""

def identify_abnormal_segments(segmental_strain: dict, threshold_gcs: float = -10.0) -> list[int]:
    """Find segments with reduced strain (wall motion abnormality)."""
```

### 6. configs/risk.yaml
```yaml
model:
  type: "gradient_boosting"  # or "cox", "threshold"
  # GLS threshold settings
  gls_threshold: -16.0
  # Gradient boosting settings
  n_estimators: 100
  max_depth: 4
  learning_rate: 0.1
  # Cox settings
  penalizer: 0.01

features:
  strain: [gls, gcs, grs, gcs_std, gcs_min, peak_systolic_sr]
  functional: [lvef, edv, esv]
  clinical: [age, sex, diabetes, hypertension]

evaluation:
  cv_folds: 5
  n_bootstrap: 1000
  confidence_level: 0.95

paths:
  results_dir: "./results/strain_analysis"
  model_dir: "./models/risk"
  report_dir: "./reports"
```

### 7. tests/test_risk.py
- Test GlsThresholdClassifier: GLS=-20 → low risk, GLS=-12 → high risk
- Test feature extraction produces expected keys
- Test CoxSurvivalModel with synthetic survival data
- Test AUC-ROC with known predictions (perfect → 1.0, random → ~0.5)
- Test NRI with known reclassification scenario
- Test clinical report generation produces valid structure
- Test interpret_strain for normal and abnormal values

## Clinical Evidence (Key thresholds)
| Parameter | Threshold | Significance |
|-----------|-----------|-------------|
| GLS > -16% | High risk | Independent MACE predictor (HR 1.10) |
| GCS > -11% | High risk | Post-STEMI MACE prediction |
| LVEF < 35% | High risk | Standard HF cutoff |
| GLS > -8.9% + LVEF < 30% | Very high risk | 94.7% sensitivity for MACE |

## Interface Contract
```python
# Input from Phase 4:
strain_results = {
    "global": {"GLS": float, "GCS": float, "GRS": float},
    "segmental": {1: {"GCS": float, "GRS": float}, ...},
    "timeseries": {"GCS": np.array, "GRS": np.array, "time": np.array},
    "strain_rate": {"peak_systolic_sr": float, "peak_diastolic_sr": float},
}

# Input from Phase 2 (optional):
volumes = {"LV_EDV": float, "LV_ESV": float, "LVEF": float}

# Output to Phase 6 (API):
risk_output = {
    "risk_score": float,          # 0-1 probability
    "risk_category": str,         # "low" / "intermediate" / "high"
    "report": dict,               # full clinical report
}
```

## Dependencies
```
scikit-learn>=1.3.0
lifelines>=0.27.0    # Cox proportional hazards
xgboost>=1.7.0       # Gradient boosting
joblib>=1.3.0        # Model serialization
```

## If You Get Stuck
- lifelines docs: https://lifelines.readthedocs.io/
- For Cox model: `from lifelines import CoxPHFitter`
- C-index interpretation: 0.5 = random, 0.7 = acceptable, 0.8 = strong
- NRI is often more clinically meaningful than AUC improvement
- If features have missing values, use `sklearn.impute.SimpleImputer(strategy='median')`
- For calibration, use `sklearn.calibration.calibration_curve`
- Risk thresholds: low < 0.2, intermediate 0.2-0.5, high > 0.5 (adjust based on clinical context)
