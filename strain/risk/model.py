"""Risk stratification models for MACE prediction post-AMI."""

from pathlib import Path

import numpy as np


class GlsThresholdClassifier:
    """Simple threshold-based risk classifier using GLS.

    Based on literature: GLS > -16% indicates elevated risk.
    """

    def __init__(self, threshold: float = -16.0):
        self.threshold = threshold

    def predict(self, features: dict[str, float]) -> dict[str, float | str]:
        gls = features.get("gls", 0.0)
        # GLS is negative; less negative (closer to 0) = worse function
        is_high_risk = gls > self.threshold
        return {
            "risk_score": abs(gls - self.threshold) / abs(self.threshold),
            "risk_category": "high" if is_high_risk else "low",
            "gls_value": gls,
            "threshold": self.threshold,
        }


class StrainRiskModel:
    """ML-based risk prediction model.

    Placeholder for a trained gradient boosting or survival model.
    To be replaced with a trained model loaded from disk.
    """

    def __init__(self, model_path: str | Path | None = None):
        self.model = None
        if model_path and Path(model_path).exists():
            self._load(model_path)

    def _load(self, path: str | Path) -> None:
        """Load a trained model from disk."""
        import joblib

        self.model = joblib.load(path)

    def predict(self, features: dict[str, float]) -> dict[str, float | str]:
        if self.model is None:
            # Fall back to threshold classifier
            return GlsThresholdClassifier().predict(features)

        feature_array = np.array([[features.get(k, 0.0) for k in sorted(features)]])
        score = float(self.model.predict_proba(feature_array)[0, 1])

        if score > 0.5:
            category = "high"
        elif score > 0.2:
            category = "intermediate"
        else:
            category = "low"

        return {"risk_score": score, "risk_category": category}
