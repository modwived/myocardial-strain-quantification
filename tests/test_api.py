"""Tests for the FastAPI application: health endpoint, analysis endpoint, schemas, and config.

Uses fastapi.testclient.TestClient with mocked services -- no real medical
data, trained models, or GPU required.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest

# Guard against missing optional dependencies
try:
    import fastapi
except ImportError:
    pytest.skip("FastAPI is not available", allow_module_level=True)

try:
    import torch
except (ImportError, OSError):
    pytest.skip("PyTorch is not available or broken in this environment", allow_module_level=True)

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Schemas tests (independent of the app)
# ---------------------------------------------------------------------------

from app.schemas import (
    AnalysisResponse,
    ErrorResponse,
    GlobalStrain,
    HealthResponse,
    RiskAssessment,
    SegmentalStrain,
    StrainTimeseries,
    VolumeMetrics,
)


class TestSchemas:
    """Tests for the Pydantic schema models."""

    def test_global_strain_valid(self):
        """GlobalStrain should accept valid data."""
        gs = GlobalStrain(GCS=-20.0, GRS=35.0, GLS=-18.0)
        assert gs.GCS == -20.0
        assert gs.GRS == 35.0
        assert gs.GLS == -18.0

    def test_global_strain_gls_optional(self):
        """GLS should be optional (None by default)."""
        gs = GlobalStrain(GCS=-20.0, GRS=35.0)
        assert gs.GLS is None

    def test_segmental_strain_valid(self):
        """SegmentalStrain should accept valid data."""
        ss = SegmentalStrain(segment=1, name="Basal anterior", GCS=-18.0, GRS=25.0)
        assert ss.segment == 1
        assert ss.name == "Basal anterior"

    def test_segmental_strain_segment_bounds(self):
        """Segment must be between 1 and 16."""
        with pytest.raises(Exception):
            SegmentalStrain(segment=0, name="Invalid", GCS=0.0, GRS=0.0)
        with pytest.raises(Exception):
            SegmentalStrain(segment=17, name="Invalid", GCS=0.0, GRS=0.0)

    def test_risk_assessment_valid(self):
        """RiskAssessment should accept valid data."""
        ra = RiskAssessment(risk_score=0.3, risk_category="low", interpretation="Normal")
        assert ra.risk_score == 0.3

    def test_health_response_valid(self):
        """HealthResponse should accept valid data."""
        hr = HealthResponse(
            status="ok",
            gpu_available=False,
            models_loaded={"segmentation": True, "motion": True, "risk": True},
            version="0.1.0",
        )
        assert hr.status == "ok"

    def test_error_response(self):
        """ErrorResponse should accept error info."""
        er = ErrorResponse(error="Bad request", detail="Missing file")
        assert er.error == "Bad request"

    def test_volume_metrics(self):
        """VolumeMetrics should accept valid data."""
        vm = VolumeMetrics(lv_edv_ml=150.0, lv_esv_ml=60.0, lvef_percent=60.0)
        assert vm.lv_edv_ml == 150.0
        assert vm.rv_edv_ml is None

    def test_strain_timeseries(self):
        """StrainTimeseries should accept valid data."""
        st = StrainTimeseries(
            time_ms=[0.0, 30.0, 60.0],
            GCS=[-0.0, -10.0, -20.0],
            GRS=[0.0, 15.0, 30.0],
        )
        assert len(st.time_ms) == 3
        assert st.GLS is None


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

from app.config import Settings


class TestSettings:
    """Tests for the Settings configuration class."""

    def test_default_values(self):
        """Default settings should be sensible."""
        settings = Settings()
        assert settings.port == 8000
        assert settings.host == "0.0.0.0"
        assert settings.log_level == "info"
        assert settings.max_upload_size_mb == 500
        assert settings.api_version == "0.1.0"

    def test_model_path_default(self):
        """Default model_path should be ./models."""
        settings = Settings()
        assert str(settings.model_path) == "models"

    def test_cors_origins_default(self):
        """Default CORS origins should be ['*']."""
        settings = Settings()
        assert settings.cors_origins == ["*"]


# ---------------------------------------------------------------------------
# App / endpoint tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for the GET /health endpoint."""

    def test_health_returns_200(self):
        """Health endpoint should return 200."""
        # We need to patch the pipeline so it does not try to load real models
        with patch("app.main.pipeline") as mock_pipeline:
            mock_pipeline.models_loaded_status = {
                "segmentation": True,
                "motion": True,
                "risk": True,
            }
            # Re-import to pick up the patched pipeline
            from app.main import app
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self):
        """Health response should contain status, gpu_available, models_loaded, version."""
        with patch("app.main.pipeline") as mock_pipeline:
            mock_pipeline.models_loaded_status = {
                "segmentation": True,
                "motion": True,
                "risk": True,
            }
            from app.main import app
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "gpu_available" in data
        assert "models_loaded" in data
        assert "version" in data

    def test_health_degraded_mode(self):
        """When models are not all loaded, status should be 'degraded'."""
        with patch("app.main.pipeline") as mock_pipeline:
            mock_pipeline.models_loaded_status = {
                "segmentation": True,
                "motion": False,  # not loaded
                "risk": True,
            }
            from app.main import app
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/health")
        data = response.json()
        assert data["status"] == "degraded"


class TestAnalyzeEndpoint:
    """Tests for the POST /analyze endpoint."""

    def test_no_file_returns_422(self):
        """POST /analyze without a file should return 422 (Unprocessable Entity)."""
        with patch("app.main.pipeline") as mock_pipeline:
            mock_pipeline.models_loaded_status = {
                "segmentation": True,
                "motion": True,
                "risk": True,
            }
            from app.main import app
            client = TestClient(app, raise_server_exceptions=False)
            response = client.post("/analyze")
        assert response.status_code == 422

    def test_unsupported_file_type(self):
        """Uploading a .txt file should return 400."""
        with patch("app.main.pipeline") as mock_pipeline:
            mock_pipeline.models_loaded_status = {
                "segmentation": True,
                "motion": True,
                "risk": True,
            }
            from app.main import app
            client = TestClient(app, raise_server_exceptions=False)
            fake_file = io.BytesIO(b"not a real image")
            response = client.post(
                "/analyze",
                files={"file": ("test.txt", fake_file, "text/plain")},
            )
        assert response.status_code == 400

    @pytest.mark.skip(reason="requires trained model and valid input data")
    def test_successful_analysis(self):
        """Full pipeline test with valid NIfTI input (requires trained model)."""
        pass


# ---------------------------------------------------------------------------
# App-level tests
# ---------------------------------------------------------------------------


class TestAppConfiguration:
    """Tests for the FastAPI app configuration."""

    def test_app_title(self):
        """App should have the correct title."""
        from app.main import app
        assert app.title == "Myocardial Strain Quantification"

    def test_app_has_health_route(self):
        """App should have a /health route."""
        from app.main import app
        routes = [r.path for r in app.routes]
        assert "/health" in routes

    def test_app_has_analyze_route(self):
        """App should have a /analyze route."""
        from app.main import app
        routes = [r.path for r in app.routes]
        assert "/analyze" in routes


# ---------------------------------------------------------------------------
# Services tests (unit-level, mocked)
# ---------------------------------------------------------------------------

from app.services import StrainPipeline


class TestStrainPipelineInit:
    """Tests for StrainPipeline initialization."""

    def test_init_no_models(self):
        """Pipeline should initialize with None models."""
        settings = Settings()
        pipeline = StrainPipeline(settings)
        assert pipeline.seg_model is None
        assert pipeline.motion_model is None
        assert pipeline.risk_model is None

    def test_models_loaded_status_before_load(self):
        """Before loading, all models should report False."""
        settings = Settings()
        pipeline = StrainPipeline(settings)
        status = pipeline.models_loaded_status
        assert status["segmentation"] is False
        assert status["motion"] is False
        assert status["risk"] is False

    def test_models_loaded_status_after_load(self):
        """After loading (even without checkpoints), models should be non-None."""
        settings = Settings()
        pipeline = StrainPipeline(settings)
        pipeline.load_models()
        status = pipeline.models_loaded_status
        # Even without checkpoint files, the code instantiates default untrained models
        assert status["segmentation"] is True
        assert status["motion"] is True
        assert status["risk"] is True

    def test_preprocess_static(self):
        """The _preprocess static method should normalize, resample, and crop."""
        import numpy as np
        settings = Settings()
        pipeline = StrainPipeline(settings)
        # Simulate (T, H, W) input
        data = np.random.rand(5, 200, 200).astype(np.float32) * 100
        metadata = {"spacing": (1.25, 1.25)}
        result = pipeline._preprocess(data, metadata)
        # After center_crop with default 128, spatial dims should be 128
        assert result.shape[-2] == 128
        assert result.shape[-1] == 128
