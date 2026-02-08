"""Risk stratification models for MACE prediction post-AMI.

Provides threshold-based, Cox proportional hazards, and gradient boosting
models for predicting Major Adverse Cardiovascular Events from cardiac
strain analysis results.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _risk_category(score: float) -> str:
    """Map a 0-1 risk score to a clinical category."""
    if score > 0.5:
        return "high"
    elif score > 0.2:
        return "intermediate"
    return "low"


# ---------------------------------------------------------------------------
# Simple threshold classifier
# ---------------------------------------------------------------------------

class GlsThresholdClassifier:
    """Simple threshold-based risk classifier using GLS.

    Based on literature: GLS > -16 % indicates elevated risk.
    """

    def __init__(self, threshold: float = -16.0):
        self.threshold = threshold

    def predict(self, features: dict[str, float]) -> dict[str, Any]:
        gls = features.get("gls", 0.0)
        # GLS is negative; less negative (closer to 0) = worse function
        is_high_risk = gls > self.threshold
        score = min(max(abs(gls - self.threshold) / abs(self.threshold), 0.0), 1.0)
        return {
            "risk_score": score if is_high_risk else 1.0 - score,
            "risk_category": "high" if is_high_risk else "low",
            "gls_value": gls,
            "threshold": self.threshold,
        }


# ---------------------------------------------------------------------------
# Multi-parameter threshold model
# ---------------------------------------------------------------------------

class MultiParameterThresholdModel:
    """Combined GLS + GCS + LVEF threshold model.

    Uses literature-derived thresholds to classify risk without any training:
      - GLS > -16 %  -> high risk
      - GCS > -11 %  -> high risk
      - LVEF < 35 %  -> high risk
      - GLS > -8.9 % AND LVEF < 30 %  -> very high risk

    Risk score is the fraction of thresholds that are breached, weighted by
    clinical importance.
    """

    DEFAULT_THRESHOLDS = {
        "gls": -16.0,     # GLS more positive than this = bad
        "gcs": -11.0,     # GCS more positive than this = bad
        "lvef": 35.0,     # LVEF below this = bad
    }

    VERY_HIGH_RISK = {
        "gls": -8.9,
        "lvef": 30.0,
    }

    # Relative clinical importance weights
    WEIGHTS = {
        "gls": 0.40,
        "gcs": 0.25,
        "lvef": 0.35,
    }

    def __init__(
        self,
        thresholds: dict[str, float] | None = None,
        weights: dict[str, float] | None = None,
    ):
        self.thresholds = thresholds or dict(self.DEFAULT_THRESHOLDS)
        self.weights = weights or dict(self.WEIGHTS)

    def predict(self, features: dict[str, float]) -> dict[str, Any]:
        gls = features.get("gls", 0.0)
        gcs = features.get("gcs", 0.0)
        lvef = features.get("lvef", 60.0)

        breached: dict[str, bool] = {
            "gls": gls > self.thresholds["gls"],
            "gcs": gcs > self.thresholds["gcs"],
            "lvef": lvef < self.thresholds["lvef"],
        }

        # Very high risk override
        very_high = (
            gls > self.VERY_HIGH_RISK["gls"]
            and lvef < self.VERY_HIGH_RISK["lvef"]
        )

        if very_high:
            score = 0.95
            category = "high"
        else:
            score = sum(
                self.weights[k] for k, v in breached.items() if v
            )
            score = min(max(score, 0.0), 1.0)
            category = _risk_category(score)

        contributing = [k for k, v in breached.items() if v]

        return {
            "risk_score": score,
            "risk_category": category,
            "contributing_factors": contributing,
            "very_high_risk": very_high,
            "thresholds_breached": breached,
        }


# ---------------------------------------------------------------------------
# Cox proportional hazards model
# ---------------------------------------------------------------------------

class CoxSurvivalModel:
    """Cox proportional hazards model for MACE survival analysis.

    Uses the *lifelines* library.  The model is fitted on a DataFrame that
    includes a duration column and an event (censoring) column alongside the
    covariates.
    """

    def __init__(self, penalizer: float = 0.01):
        self.penalizer = penalizer
        self.model = None
        self.feature_names: list[str] | None = None

    # -- Fitting -------------------------------------------------------------

    def fit(
        self,
        features_df: pd.DataFrame,
        duration_col: str = "duration",
        event_col: str = "event",
    ) -> "CoxSurvivalModel":
        """Fit Cox model.

        Args:
            features_df: DataFrame with covariates **plus** *duration_col*
                and *event_col*.
            duration_col: Name of the time-to-event column.
            event_col: Name of the event indicator column (1 = event, 0 = censored).

        Returns:
            self (for chaining).
        """
        from lifelines import CoxPHFitter

        self.feature_names = [
            c for c in features_df.columns if c not in (duration_col, event_col)
        ]
        self.model = CoxPHFitter(penalizer=self.penalizer)
        self.model.fit(
            features_df,
            duration_col=duration_col,
            event_col=event_col,
        )
        logger.info(
            "CoxSurvivalModel fitted  --  C-index: %.4f",
            self.model.concordance_index_,
        )
        return self

    # -- Prediction ----------------------------------------------------------

    def predict_risk(self, features: dict[str, float]) -> dict[str, Any]:
        """Predict risk for a single patient.

        Returns:
            Dictionary with ``risk_score``, ``survival_probability`` (at
            selected time points), and ``risk_category``.
        """
        if self.model is None or self.feature_names is None:
            raise RuntimeError("Model has not been fitted.")

        row = pd.DataFrame(
            [{k: features.get(k, 0.0) for k in self.feature_names}]
        )
        # Partial hazard as risk score (higher = worse)
        partial_hazard = float(self.model.predict_partial_hazard(row).iloc[0])
        # Normalize to 0-1 via sigmoid
        risk_score = 1.0 / (1.0 + np.exp(-partial_hazard))

        # Survival function at common time points (months)
        survival_func = self.model.predict_survival_function(row)
        surv_probs: dict[str, float] = {}
        for t in [6, 12, 24, 36]:
            # Find closest time index
            idx = survival_func.index
            closest = idx[np.argmin(np.abs(idx - t))]
            surv_probs[f"{t}_months"] = float(survival_func.loc[closest].iloc[0])

        return {
            "risk_score": float(risk_score),
            "survival_probability": surv_probs,
            "risk_category": _risk_category(risk_score),
            "partial_hazard": partial_hazard,
        }

    # -- Evaluation ----------------------------------------------------------

    def concordance_index(
        self,
        features_df: pd.DataFrame,
        duration_col: str = "duration",
        event_col: str = "event",
    ) -> float:
        """Compute Harrell's C-index on the given data."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")
        return float(self.model.score(features_df, scoring_method="concordance_index"))

    # -- Summary -------------------------------------------------------------

    def summary(self) -> pd.DataFrame | None:
        """Return the lifelines model summary table."""
        if self.model is None:
            return None
        return self.model.summary


# ---------------------------------------------------------------------------
# Gradient boosting model (XGBoost / sklearn fallback)
# ---------------------------------------------------------------------------

class GradientBoostingRiskModel:
    """XGBoost-based classifier for binary MACE prediction.

    Falls back to ``sklearn.ensemble.GradientBoostingClassifier`` when
    *xgboost* is not installed.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        use_xgboost: bool = True,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.use_xgboost = use_xgboost
        self.model = None
        self.feature_names: list[str] | None = None
        self._build_model()

    def _build_model(self) -> None:
        if self.use_xgboost:
            try:
                from xgboost import XGBClassifier
                self.model = XGBClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=42,
                )
                return
            except ImportError:
                logger.warning(
                    "xgboost not installed; falling back to sklearn "
                    "GradientBoostingClassifier."
                )

        from sklearn.ensemble import GradientBoostingClassifier
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42,
        )

    # -- Training ------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> "GradientBoostingRiskModel":
        """Train the model.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Binary outcome array (1 = MACE, 0 = no MACE).

        Returns:
            self
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f"f{i}" for i in range(X.shape[1])]
        self.model.fit(X, y)
        logger.info("GradientBoostingRiskModel fitted on %d samples.", len(y))
        return self

    # -- Prediction ----------------------------------------------------------

    def predict(self, features: dict[str, float]) -> dict[str, Any]:
        """Predict risk for a single patient.

        Returns:
            Dictionary with ``risk_score``, ``risk_category``, and
            ``feature_importance``.
        """
        if self.model is None or self.feature_names is None:
            raise RuntimeError("Model has not been fitted.")

        X_row = np.array([[features.get(k, 0.0) for k in self.feature_names]])
        score = float(self.model.predict_proba(X_row)[0, 1])
        importance = dict(zip(self.feature_names, self.model.feature_importances_))

        return {
            "risk_score": score,
            "risk_category": _risk_category(score),
            "feature_importance": importance,
        }

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return predicted probabilities for a matrix of patients."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")
        return self.model.predict_proba(X)


# ---------------------------------------------------------------------------
# Generic ML wrapper (kept for backward compatibility)
# ---------------------------------------------------------------------------

class StrainRiskModel:
    """ML-based risk prediction model.

    Wraps a serialized sklearn-compatible model loaded from disk.
    Falls back to :class:`GlsThresholdClassifier` if no model file is found.
    """

    def __init__(self, model_path: str | Path | None = None):
        self.model = None
        self.feature_names: list[str] | None = None
        if model_path and Path(model_path).exists():
            self._load(model_path)

    def _load(self, path: str | Path) -> None:
        import joblib

        payload = joblib.load(path)
        if isinstance(payload, dict):
            self.model = payload.get("model")
            self.feature_names = payload.get("feature_names")
        else:
            self.model = payload

    def predict(self, features: dict[str, float]) -> dict[str, Any]:
        if self.model is None:
            return GlsThresholdClassifier().predict(features)

        if self.feature_names:
            feature_array = np.array(
                [[features.get(k, 0.0) for k in self.feature_names]]
            )
        else:
            feature_array = np.array(
                [[features.get(k, 0.0) for k in sorted(features)]]
            )

        score = float(self.model.predict_proba(feature_array)[0, 1])
        return {
            "risk_score": score,
            "risk_category": _risk_category(score),
        }


# ---------------------------------------------------------------------------
# Model serialization
# ---------------------------------------------------------------------------

def save_model(
    model: Any,
    path: str | Path,
    feature_names: list[str] | None = None,
    scaler: Any | None = None,
    metadata: dict | None = None,
) -> Path:
    """Persist a trained model with its metadata.

    Saves a dictionary with keys ``model``, ``feature_names``, ``scaler``,
    ``metadata`` using *joblib*.

    Args:
        model: Trained model object (sklearn, xgboost, or lifelines).
        path: File path for the serialized model.
        feature_names: Ordered list of feature names the model expects.
        scaler: Optional fitted ``StandardScaler`` (or similar) used during
            training.
        metadata: Arbitrary metadata (e.g. training date, performance).

    Returns:
        Resolved ``Path`` to the saved file.
    """
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = metadata or {}
    meta.setdefault("saved_at", datetime.now(timezone.utc).isoformat())

    payload = {
        "model": model,
        "feature_names": feature_names,
        "scaler": scaler,
        "metadata": meta,
    }
    joblib.dump(payload, path)
    logger.info("Model saved to %s", path)
    return path


def load_model(path: str | Path) -> dict[str, Any]:
    """Load a model saved with :func:`save_model`.

    Returns:
        Dictionary with keys ``model``, ``feature_names``, ``scaler``,
        ``metadata``.
    """
    import joblib

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    payload = joblib.load(path)
    if not isinstance(payload, dict):
        # Legacy format: bare model object
        return {"model": payload, "feature_names": None, "scaler": None, "metadata": {}}
    return payload
