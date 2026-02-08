"""Tests for the risk stratification module: feature extraction and risk models.

Uses synthetic data -- no trained models or real patient data required.
"""

from __future__ import annotations

import numpy as np
import pytest

from strain.risk.features import (
    _compute_strain_rate,
    _extract_heterogeneity_features,
    _extract_strain_rate_features,
    _extract_temporal_features,
    compute_wall_motion_score,
    extract_strain_features,
)
from strain.risk.model import (
    GlsThresholdClassifier,
    MultiParameterThresholdModel,
    StrainRiskModel,
    _risk_category,
)


# ---------------------------------------------------------------------------
# GlsThresholdClassifier tests
# ---------------------------------------------------------------------------


class TestGlsThresholdClassifier:
    """Tests for the simple GLS threshold classifier."""

    def test_low_risk_normal_gls(self):
        """GLS = -20 (well below -16 threshold) should be low risk."""
        clf = GlsThresholdClassifier(threshold=-16.0)
        features = {"gls": -20.0}
        result = clf.predict(features)
        assert result["risk_category"] == "low", (
            f"GLS=-20 should be low risk, got {result['risk_category']}"
        )
        assert result["gls_value"] == -20.0

    def test_high_risk_abnormal_gls(self):
        """GLS = -12 (above -16 threshold, closer to 0) should be high risk."""
        clf = GlsThresholdClassifier(threshold=-16.0)
        features = {"gls": -12.0}
        result = clf.predict(features)
        assert result["risk_category"] == "high", (
            f"GLS=-12 should be high risk, got {result['risk_category']}"
        )

    def test_threshold_boundary(self):
        """GLS exactly at the threshold: should be low risk (not >, only strict)."""
        clf = GlsThresholdClassifier(threshold=-16.0)
        features = {"gls": -16.0}
        result = clf.predict(features)
        # -16.0 > -16.0 is False, so should be low risk
        assert result["risk_category"] == "low"

    def test_severely_impaired(self):
        """GLS = -5 (severely impaired) should be high risk with high score."""
        clf = GlsThresholdClassifier(threshold=-16.0)
        features = {"gls": -5.0}
        result = clf.predict(features)
        assert result["risk_category"] == "high"
        assert result["risk_score"] > 0.5

    def test_returns_expected_keys(self):
        """Result should contain risk_score, risk_category, gls_value, threshold."""
        clf = GlsThresholdClassifier()
        result = clf.predict({"gls": -18.0})
        assert "risk_score" in result
        assert "risk_category" in result
        assert "gls_value" in result
        assert "threshold" in result

    def test_custom_threshold(self):
        """Custom threshold should be respected."""
        clf = GlsThresholdClassifier(threshold=-20.0)
        # GLS = -18 is above -20 (closer to 0) -> high risk
        result = clf.predict({"gls": -18.0})
        assert result["risk_category"] == "high"

    def test_missing_gls_defaults_to_zero(self):
        """If gls key is missing, should default to 0.0 (high risk)."""
        clf = GlsThresholdClassifier(threshold=-16.0)
        result = clf.predict({})
        # gls=0.0 > -16.0 -> high risk
        assert result["risk_category"] == "high"


# ---------------------------------------------------------------------------
# MultiParameterThresholdModel tests
# ---------------------------------------------------------------------------


class TestMultiParameterThresholdModel:
    """Tests for the multi-parameter threshold model."""

    def test_all_normal_low_risk(self):
        """All normal values should produce low risk."""
        model = MultiParameterThresholdModel()
        features = {"gls": -22.0, "gcs": -18.0, "lvef": 60.0}
        result = model.predict(features)
        assert result["risk_category"] == "low"
        assert result["risk_score"] < 0.2

    def test_all_abnormal_high_risk(self):
        """All abnormal values should produce high risk."""
        model = MultiParameterThresholdModel()
        features = {"gls": -10.0, "gcs": -5.0, "lvef": 25.0}
        result = model.predict(features)
        assert result["risk_category"] == "high"

    def test_very_high_risk_override(self):
        """GLS > -8.9 AND LVEF < 30 should trigger very_high_risk."""
        model = MultiParameterThresholdModel()
        features = {"gls": -5.0, "gcs": -15.0, "lvef": 25.0}
        result = model.predict(features)
        assert result["very_high_risk"] is True
        assert result["risk_score"] == 0.95

    def test_returns_contributing_factors(self):
        """Should list which thresholds were breached."""
        model = MultiParameterThresholdModel()
        features = {"gls": -10.0, "gcs": -18.0, "lvef": 60.0}
        result = model.predict(features)
        assert "gls" in result["contributing_factors"]
        assert "gcs" not in result["contributing_factors"]
        assert "lvef" not in result["contributing_factors"]

    def test_returns_expected_keys(self):
        """Should return risk_score, risk_category, contributing_factors, etc."""
        model = MultiParameterThresholdModel()
        result = model.predict({"gls": -20.0, "gcs": -15.0, "lvef": 55.0})
        assert "risk_score" in result
        assert "risk_category" in result
        assert "contributing_factors" in result
        assert "very_high_risk" in result
        assert "thresholds_breached" in result


# ---------------------------------------------------------------------------
# StrainRiskModel (generic wrapper) tests
# ---------------------------------------------------------------------------


class TestStrainRiskModel:
    """Tests for the StrainRiskModel wrapper (fallback mode)."""

    def test_fallback_to_threshold_classifier(self):
        """Without a model file, should fall back to GlsThresholdClassifier."""
        model = StrainRiskModel(model_path=None)
        result = model.predict({"gls": -20.0})
        assert "risk_score" in result
        assert "risk_category" in result

    def test_nonexistent_path_falls_back(self):
        """Non-existent model path should fall back gracefully."""
        model = StrainRiskModel(model_path="/nonexistent/path/model.joblib")
        result = model.predict({"gls": -10.0})
        assert result["risk_category"] == "high"


# ---------------------------------------------------------------------------
# Risk category utility
# ---------------------------------------------------------------------------


class TestRiskCategory:
    """Tests for the _risk_category helper."""

    def test_high(self):
        assert _risk_category(0.6) == "high"
        assert _risk_category(0.99) == "high"

    def test_intermediate(self):
        assert _risk_category(0.3) == "intermediate"
        assert _risk_category(0.5) == "intermediate"

    def test_low(self):
        assert _risk_category(0.1) == "low"
        assert _risk_category(0.0) == "low"
        assert _risk_category(0.2) == "low"


# ---------------------------------------------------------------------------
# Feature extraction tests
# ---------------------------------------------------------------------------


class TestExtractStrainFeatures:
    """Tests for extract_strain_features."""

    def test_returns_expected_keys(self):
        """Should return at least gls, gcs, grs keys."""
        strain_results = {
            "global": {"GLS": -18.0, "GCS": -20.0, "GRS": 35.0},
        }
        features = extract_strain_features(strain_results)
        assert "gls" in features
        assert "gcs" in features
        assert "grs" in features
        assert features["gls"] == -18.0
        assert features["gcs"] == -20.0
        assert features["grs"] == 35.0

    def test_with_segmental_strain(self):
        """Should extract heterogeneity features from segmental data."""
        strain_results = {
            "global": {"GLS": -17.0, "GCS": -19.0, "GRS": 30.0},
            "segmental": {
                1: {"GCS": -18.0, "GRS": 25.0},
                2: {"GCS": -22.0, "GRS": 35.0},
                3: {"GCS": -15.0, "GRS": 28.0},
                4: {"GCS": -20.0, "GRS": 32.0},
            },
        }
        features = extract_strain_features(strain_results)
        assert "gcs_std" in features
        assert "gcs_min" in features
        assert "gcs_max" in features
        assert features["gcs_std"] > 0.0

    def test_with_clinical_data(self):
        """Should incorporate clinical features when provided."""
        strain_results = {"global": {"GCS": -15.0, "GRS": 25.0}}
        clinical = {"age": 65.0, "sex": 1.0, "diabetes": 1.0}
        features = extract_strain_features(strain_results, clinical_data=clinical)
        assert features["age"] == 65.0
        assert features["sex"] == 1.0
        assert features["diabetes"] == 1.0

    def test_with_volumes(self):
        """Should extract volume-based features."""
        strain_results = {
            "global": {"GCS": -18.0, "GRS": 30.0},
            "volumes": {"LV_EDV": 150.0, "LV_ESV": 60.0},
        }
        features = extract_strain_features(strain_results)
        assert "edv" in features
        assert features["edv"] == 150.0
        assert "esv" in features
        assert features["esv"] == 60.0
        assert "sv" in features
        assert features["sv"] == 90.0  # EDV - ESV
        assert "lvef" in features
        assert abs(features["lvef"] - 60.0) < 0.1  # 90/150 * 100 = 60%

    def test_with_timeseries(self):
        """Should extract temporal features from timeseries data."""
        strain_results = {
            "global": {"GCS": -20.0, "GRS": 35.0},
            "timeseries": {
                "GCS": [0.0, -10.0, -20.0, -15.0, -3.0],
                "time": [0.0, 50.0, 100.0, 150.0, 200.0],
            },
        }
        features = extract_strain_features(strain_results)
        assert "time_to_peak_strain" in features
        assert "strain_recovery_ratio" in features
        # Peak is at index 2 (value -20), time = 100 ms
        assert features["time_to_peak_strain"] == 100.0

    def test_with_strain_rate_precomputed(self):
        """Should use precomputed strain rate values."""
        strain_results = {
            "global": {"GCS": -18.0, "GRS": 30.0},
            "strain_rate": {
                "peak_systolic_sr": -1.2,
                "peak_diastolic_sr": 1.5,
            },
        }
        features = extract_strain_features(strain_results)
        assert features["peak_systolic_sr"] == -1.2
        assert features["peak_diastolic_sr"] == 1.5

    def test_empty_results(self):
        """Should handle empty/minimal results without crashing."""
        features = extract_strain_features({})
        assert "gls" in features
        assert features["gls"] == 0.0


# ---------------------------------------------------------------------------
# Internal feature extraction helpers
# ---------------------------------------------------------------------------


class TestComputeStrainRateHelper:
    """Tests for the strain-rate helper in features module."""

    def test_basic_computation(self):
        """Should compute finite differences correctly."""
        strain = np.array([0.0, -10.0, -20.0])
        time = np.array([0.0, 100.0, 200.0])
        sr = _compute_strain_rate(strain, time)
        # dt = diff(time)/1000 = [0.1, 0.1]
        # ds = diff(strain) = [-10, -10]
        # sr = [-10/0.1, -10/0.1] = [-100, -100]
        np.testing.assert_allclose(sr, [-100.0, -100.0])


class TestExtractHeterogeneityFeatures:
    """Tests for segmental heterogeneity feature extraction."""

    def test_with_data(self):
        """Should compute std, min, max, cv from segmental GCS."""
        segmental = {
            1: {"GCS": -18.0, "GRS": 25.0},
            2: {"GCS": -22.0, "GRS": 35.0},
        }
        features = _extract_heterogeneity_features(segmental)
        assert "gcs_std" in features
        assert "gcs_min" in features
        assert "gcs_max" in features
        assert features["gcs_min"] == -22.0
        assert features["gcs_max"] == -18.0

    def test_empty_input(self):
        """Should return empty dict for None/empty segmental data."""
        assert _extract_heterogeneity_features(None) == {}
        assert _extract_heterogeneity_features({}) == {}


class TestComputeWallMotionScore:
    """Tests for compute_wall_motion_score."""

    def test_all_normal(self):
        """All segments with GCS < -10 should give score 0."""
        segmental = {
            1: {"GCS": -18.0},
            2: {"GCS": -22.0},
            3: {"GCS": -15.0},
        }
        score = compute_wall_motion_score(segmental, threshold_gcs=-10.0)
        assert score == 0.0

    def test_some_abnormal(self):
        """Segments with GCS > threshold should be counted."""
        segmental = {
            1: {"GCS": -18.0},  # normal
            2: {"GCS": -8.0},   # abnormal (> -10)
            3: {"GCS": -5.0},   # abnormal (> -10)
        }
        score = compute_wall_motion_score(segmental, threshold_gcs=-10.0)
        assert score == 2.0

    def test_empty_input(self):
        """Should return 0 for None/empty input."""
        assert compute_wall_motion_score(None) == 0.0
        assert compute_wall_motion_score({}) == 0.0


class TestExtractStrainRateFeatures:
    """Tests for _extract_strain_rate_features."""

    def test_precomputed(self):
        """Should use precomputed values when available."""
        strain_results = {
            "strain_rate": {"peak_systolic_sr": -1.3, "peak_diastolic_sr": 1.8}
        }
        features = _extract_strain_rate_features(strain_results)
        assert features["peak_systolic_sr"] == -1.3
        assert features["peak_diastolic_sr"] == 1.8

    def test_fallback_default(self):
        """Should default to 0.0 when no data is available."""
        features = _extract_strain_rate_features({})
        assert features["peak_systolic_sr"] == 0.0
        assert features["peak_diastolic_sr"] == 0.0


class TestExtractTemporalFeatures:
    """Tests for _extract_temporal_features."""

    def test_with_timeseries(self):
        """Should compute time_to_peak_strain and recovery_ratio."""
        strain_results = {
            "timeseries": {
                "GCS": [0.0, -10.0, -20.0, -15.0, -5.0],
                "time": [0.0, 50.0, 100.0, 150.0, 200.0],
            }
        }
        features = _extract_temporal_features(strain_results)
        # Peak at index 2, time 100
        assert features["time_to_peak_strain"] == 100.0
        # Recovery: abs(-5 / -20) = 0.25
        assert abs(features["strain_recovery_ratio"] - 0.25) < 1e-6

    def test_no_timeseries(self):
        """Should default to 0.0 when no timeseries data."""
        features = _extract_temporal_features({})
        assert features["time_to_peak_strain"] == 0.0
        assert features["strain_recovery_ratio"] == 0.0
