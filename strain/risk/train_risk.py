"""Training pipeline for MACE risk prediction models.

Supports nested cross-validation, hyper-parameter tuning, and head-to-head
comparison of threshold, Cox, and gradient boosting models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from strain.risk.evaluate import (
    compute_auc_roc,
    compute_concordance_index,
    compute_nri,
)
from strain.risk.features import create_feature_matrix
from strain.risk.model import (
    CoxSurvivalModel,
    GlsThresholdClassifier,
    GradientBoostingRiskModel,
    MultiParameterThresholdModel,
    save_model,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: str | Path) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


def _default_config() -> dict:
    """Minimal in-code configuration used when no YAML is supplied."""
    return {
        "model": {
            "type": "gradient_boosting",
            "gls_threshold": -16.0,
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "penalizer": 0.01,
        },
        "features": {
            "strain": [
                "gls", "gcs", "grs", "gcs_std", "gcs_min",
                "peak_systolic_sr",
            ],
            "functional": ["lvef", "edv", "esv"],
            "clinical": ["age", "sex", "diabetes", "hypertension"],
        },
        "evaluation": {
            "cv_folds": 5,
            "n_bootstrap": 1000,
            "confidence_level": 0.95,
        },
        "paths": {
            "results_dir": "./results/strain_analysis",
            "model_dir": "./models/risk",
            "report_dir": "./reports",
        },
    }


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(
    patients_data: list[dict],
    feature_names: list[str],
    outcome_key: str = "mace",
    duration_key: str = "duration",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    """Build feature matrix and extract outcomes.

    Args:
        patients_data: List of patient dictionaries.
        feature_names: Feature columns to retain.
        outcome_key: Key in each patient dict for binary outcome.
        duration_key: Key for time-to-event (used by Cox).

    Returns:
        ``(X_df, y, durations)`` -- features, binary outcome, and durations
        (``None`` if *duration_key* is not present).
    """
    X_df, _ = create_feature_matrix(patients_data, feature_names=feature_names)

    y = np.array(
        [float(p.get(outcome_key, 0)) for p in patients_data], dtype=float
    )
    durations: np.ndarray | None = None
    if all(duration_key in p for p in patients_data):
        durations = np.array(
            [float(p[duration_key]) for p in patients_data], dtype=float
        )

    return X_df, y, durations


def impute_and_scale(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    imputer: SimpleImputer | None = None,
    scaler: StandardScaler | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, SimpleImputer, StandardScaler]:
    """Impute missing values and standardize features.

    Fits on *X_train*, transforms both *X_train* and *X_test*.

    Returns:
        ``(X_train_proc, X_test_proc, imputer, scaler)``
    """
    cols = X_train.columns.tolist()

    if imputer is None:
        imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    if scaler is None:
        scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_imp)
    X_test_sc = scaler.transform(X_test_imp)

    return (
        pd.DataFrame(X_train_sc, columns=cols),
        pd.DataFrame(X_test_sc, columns=cols),
        imputer,
        scaler,
    )


# ---------------------------------------------------------------------------
# Nested cross-validation
# ---------------------------------------------------------------------------

def _train_gradient_boosting_fold(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    cfg: dict,
    inner_folds: int = 5,
) -> dict[str, Any]:
    """Train a gradient boosting model on one outer fold, tune on inner folds."""
    from sklearn.model_selection import GridSearchCV

    gb = GradientBoostingRiskModel(
        n_estimators=cfg.get("n_estimators", 100),
        max_depth=cfg.get("max_depth", 4),
        learning_rate=cfg.get("learning_rate", 0.1),
        use_xgboost=True,
    )

    # Light inner-loop search over a small grid
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [3, 4],
        "learning_rate": [0.05, 0.1],
    }

    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=42)
    grid = GridSearchCV(
        gb.model,
        param_grid,
        cv=inner_cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)

    # Replace model with best estimator
    gb.model = grid.best_estimator_
    gb.feature_names = X_train.columns.tolist()

    y_pred = gb.model.predict_proba(X_test)[:, 1]
    auc, _, _ = compute_auc_roc(y_test, y_pred)
    c_idx = compute_concordance_index(
        event_times=np.arange(len(y_test), dtype=float),
        predicted_scores=y_pred,
        event_observed=y_test,
    )

    return {
        "model": gb,
        "best_params": grid.best_params_,
        "auc": auc,
        "c_index": c_idx,
        "y_pred": y_pred,
        "y_test": y_test,
    }


def train_risk_model(
    config_path: str | Path | None = None,
    patients_data: list[dict] | None = None,
) -> dict[str, Any]:
    """Train and evaluate a risk prediction model.

    Pipeline:
      1. Load configuration.
      2. Extract features.
      3. Handle missing values (median imputation).
      4. Normalize features (``StandardScaler``).
      5. Train with nested cross-validation (5x5 by default).
      6. Evaluate: C-index, AUC-ROC.
      7. Save the best model + scaler + feature names.

    Args:
        config_path: Path to ``risk.yaml``.  Uses defaults if ``None``.
        patients_data: Pre-loaded list of patient dicts.  If ``None`` the
            function will look for serialized results under ``paths.results_dir``.

    Returns:
        Dictionary summarising training results.
    """
    cfg = _load_config(config_path) if config_path else _default_config()
    model_cfg = cfg.get("model", {})
    feat_cfg = cfg.get("features", {})
    eval_cfg = cfg.get("evaluation", {})
    paths_cfg = cfg.get("paths", {})

    feature_names = (
        feat_cfg.get("strain", [])
        + feat_cfg.get("functional", [])
        + feat_cfg.get("clinical", [])
    )

    if patients_data is None:
        raise ValueError(
            "patients_data must be provided (or results_dir must contain "
            "serialized patient results)."
        )

    X_df, y, durations = prepare_data(patients_data, feature_names)

    outer_folds = eval_cfg.get("cv_folds", 5)
    inner_folds = eval_cfg.get("cv_folds", 5)
    skf = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=42)

    fold_results: list[dict] = []
    best_auc = -1.0
    best_model = None
    best_scaler = None
    best_imputer = None

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_df, y)):
        logger.info("Outer fold %d / %d", fold_idx + 1, outer_folds)

        X_train = X_df.iloc[train_idx].reset_index(drop=True)
        X_test = X_df.iloc[test_idx].reset_index(drop=True)
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_sc, X_test_sc, imputer, scaler = impute_and_scale(X_train, X_test)

        result = _train_gradient_boosting_fold(
            X_train_sc, y_train, X_test_sc, y_test, model_cfg, inner_folds
        )
        result["fold"] = fold_idx
        fold_results.append(result)

        if result["auc"] > best_auc:
            best_auc = result["auc"]
            best_model = result["model"]
            best_scaler = scaler
            best_imputer = imputer

    mean_auc = float(np.mean([r["auc"] for r in fold_results]))
    std_auc = float(np.std([r["auc"] for r in fold_results]))
    mean_c = float(np.mean([r["c_index"] for r in fold_results]))

    logger.info("Nested CV -- AUC: %.4f +/- %.4f  C-index: %.4f", mean_auc, std_auc, mean_c)

    # Save best model
    model_dir = Path(paths_cfg.get("model_dir", "./models/risk"))
    if best_model is not None:
        save_model(
            model=best_model.model,
            path=model_dir / "best_gradient_boosting.joblib",
            feature_names=feature_names,
            scaler=best_scaler,
            metadata={
                "mean_auc": mean_auc,
                "std_auc": std_auc,
                "mean_c_index": mean_c,
                "outer_folds": outer_folds,
                "inner_folds": inner_folds,
            },
        )

    return {
        "fold_results": fold_results,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "mean_c_index": mean_c,
        "best_model": best_model,
        "best_scaler": best_scaler,
        "best_imputer": best_imputer,
        "feature_names": feature_names,
    }


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def compare_models(
    config_path: str | Path | None = None,
    patients_data: list[dict] | None = None,
) -> pd.DataFrame:
    """Compare GLS-threshold vs. Cox vs. XGBoost models.

    Runs each model through the same cross-validation folds and reports
    C-index, AUC-ROC, and NRI (XGBoost vs. threshold baseline).

    Args:
        config_path: Path to ``risk.yaml``.
        patients_data: Pre-loaded patient data list.

    Returns:
        DataFrame with one row per model and columns for each metric.
    """
    cfg = _load_config(config_path) if config_path else _default_config()
    feat_cfg = cfg.get("features", {})
    eval_cfg = cfg.get("evaluation", {})
    model_cfg = cfg.get("model", {})

    feature_names = (
        feat_cfg.get("strain", [])
        + feat_cfg.get("functional", [])
        + feat_cfg.get("clinical", [])
    )

    if patients_data is None:
        raise ValueError("patients_data must be provided.")

    X_df, y, durations = prepare_data(patients_data, feature_names)
    n_folds = eval_cfg.get("cv_folds", 5)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    results: dict[str, list[dict]] = {
        "threshold": [],
        "multi_threshold": [],
        "gradient_boosting": [],
    }

    # Optionally add Cox if durations exist
    has_durations = durations is not None

    if has_durations:
        results["cox"] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_df, y)):
        X_train = X_df.iloc[train_idx].reset_index(drop=True)
        X_test = X_df.iloc[test_idx].reset_index(drop=True)
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_sc, X_test_sc, imp, scl = impute_and_scale(X_train, X_test)

        # --- GLS threshold ---
        thresh = GlsThresholdClassifier(threshold=model_cfg.get("gls_threshold", -16.0))
        y_pred_thr = np.array([
            thresh.predict(dict(zip(feature_names, row)))["risk_score"]
            for row in X_test_sc.values
        ])
        auc_thr, _, _ = compute_auc_roc(y_test, y_pred_thr)
        results["threshold"].append({"auc": auc_thr})

        # --- Multi-parameter threshold ---
        multi = MultiParameterThresholdModel()
        y_pred_multi = np.array([
            multi.predict(dict(zip(feature_names, row)))["risk_score"]
            for row in X_test_sc.values
        ])
        auc_multi, _, _ = compute_auc_roc(y_test, y_pred_multi)
        results["multi_threshold"].append({"auc": auc_multi})

        # --- Gradient boosting ---
        gb = GradientBoostingRiskModel(
            n_estimators=model_cfg.get("n_estimators", 100),
            max_depth=model_cfg.get("max_depth", 4),
            learning_rate=model_cfg.get("learning_rate", 0.1),
        )
        gb.fit(X_train_sc, y_train)
        y_pred_gb = gb.predict_proba(X_test_sc)[:, 1]
        auc_gb, _, _ = compute_auc_roc(y_test, y_pred_gb)
        results["gradient_boosting"].append({"auc": auc_gb})

        # --- Cox model ---
        if has_durations:
            dur_train = durations[train_idx]
            dur_test = durations[test_idx]
            cox_train_df = X_train_sc.copy()
            cox_train_df["duration"] = dur_train
            cox_train_df["event"] = y_train

            cox = CoxSurvivalModel(penalizer=model_cfg.get("penalizer", 0.01))
            try:
                cox.fit(cox_train_df, duration_col="duration", event_col="event")
                cox_test_df = X_test_sc.copy()
                cox_test_df["duration"] = dur_test
                cox_test_df["event"] = y_test
                c_idx_cox = cox.concordance_index(cox_test_df)
                # Use partial hazard as score for AUC
                y_pred_cox = np.array([
                    cox.predict_risk(dict(zip(feature_names, row)))["risk_score"]
                    for row in X_test_sc.values
                ])
                auc_cox, _, _ = compute_auc_roc(y_test, y_pred_cox)
                results["cox"].append({"auc": auc_cox, "c_index": c_idx_cox})
            except Exception as exc:
                logger.warning("Cox model failed on fold %d: %s", fold_idx, exc)
                results["cox"].append({"auc": float("nan"), "c_index": float("nan")})

    # --- NRI: gradient boosting vs. threshold ---
    nri_values: list[float] = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_df, y)):
        X_test = X_df.iloc[test_idx].reset_index(drop=True)
        y_test = y[test_idx]
        _, X_test_sc, _, _ = impute_and_scale(
            X_df.iloc[train_idx].reset_index(drop=True), X_test
        )
        thresh = GlsThresholdClassifier(threshold=model_cfg.get("gls_threshold", -16.0))
        y_old = np.array([
            thresh.predict(dict(zip(feature_names, row)))["risk_score"]
            for row in X_test_sc.values
        ])
        gb = GradientBoostingRiskModel(
            n_estimators=model_cfg.get("n_estimators", 100),
            max_depth=model_cfg.get("max_depth", 4),
            learning_rate=model_cfg.get("learning_rate", 0.1),
        )
        X_train_sc2, _, _, _ = impute_and_scale(
            X_df.iloc[train_idx].reset_index(drop=True), X_test
        )
        gb.fit(X_train_sc2, y[train_idx])
        y_new = gb.predict_proba(X_test_sc)[:, 1]
        nri_result = compute_nri(y_test, y_old, y_new)
        nri_values.append(nri_result["nri"])

    # Summarize
    summary_rows = []
    for name, folds in results.items():
        aucs = [f["auc"] for f in folds]
        row = {
            "model": name,
            "auc_mean": float(np.nanmean(aucs)),
            "auc_std": float(np.nanstd(aucs)),
        }
        if folds and "c_index" in folds[0]:
            c_idxs = [f["c_index"] for f in folds]
            row["c_index_mean"] = float(np.nanmean(c_idxs))
            row["c_index_std"] = float(np.nanstd(c_idxs))
        summary_rows.append(row)

    # Add NRI row for gradient boosting
    for row in summary_rows:
        if row["model"] == "gradient_boosting":
            row["nri_vs_threshold_mean"] = float(np.nanmean(nri_values))

    comparison_df = pd.DataFrame(summary_rows)
    logger.info("Model comparison:\n%s", comparison_df.to_string(index=False))
    return comparison_df
