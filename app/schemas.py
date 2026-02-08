"""Pydantic request/response models for the myocardial strain API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GlobalStrain(BaseModel):
    """Global strain measurements across the entire myocardium."""

    GLS: float | None = Field(None, description="Global Longitudinal Strain (%)")
    GCS: float = Field(..., description="Global Circumferential Strain (%)")
    GRS: float = Field(..., description="Global Radial Strain (%)")


class SegmentalStrain(BaseModel):
    """Strain values for a single AHA segment."""

    segment: int = Field(..., ge=1, le=16, description="AHA segment number (1-16)")
    name: str = Field(..., description="AHA segment name")
    GCS: float = Field(..., description="Circumferential strain (%)")
    GRS: float = Field(..., description="Radial strain (%)")


class RiskAssessment(BaseModel):
    """MACE risk stratification result."""

    risk_score: float = Field(..., ge=0.0, description="Continuous risk score")
    risk_category: str = Field(
        ..., description='Risk category: "low", "intermediate", or "high"'
    )
    interpretation: str = Field(
        ..., description="Human-readable interpretation of the risk assessment"
    )


class StrainTimeseries(BaseModel):
    """Strain curves over the cardiac cycle."""

    time_ms: list[float] = Field(..., description="Time points in milliseconds")
    GCS: list[float] = Field(..., description="Circumferential strain curve (%)")
    GRS: list[float] = Field(..., description="Radial strain curve (%)")
    GLS: list[float] | None = Field(
        None, description="Longitudinal strain curve (%) if available"
    )


class VolumeMetrics(BaseModel):
    """Cardiac volume measurements."""

    lv_edv_ml: float = Field(..., description="LV end-diastolic volume (mL)")
    lv_esv_ml: float = Field(..., description="LV end-systolic volume (mL)")
    lvef_percent: float = Field(..., description="LV ejection fraction (%)")
    rv_edv_ml: float | None = Field(
        None, description="RV end-diastolic volume (mL) if available"
    )


class AnalysisResponse(BaseModel):
    """Complete strain analysis result returned by POST /analyze."""

    patient_id: str | None = Field(None, description="Patient identifier if available")
    global_strain: GlobalStrain
    segmental_strain: list[SegmentalStrain]
    risk_assessment: RiskAssessment
    volumes: VolumeMetrics | None = None
    timeseries: StrainTimeseries | None = None
    processing_time_sec: float = Field(
        ..., description="Total pipeline processing time in seconds"
    )


class HealthResponse(BaseModel):
    """Response from the GET /health endpoint."""

    status: str = Field(..., description='Service status: "ok" or "degraded"')
    gpu_available: bool = Field(..., description="Whether a CUDA GPU is available")
    models_loaded: dict[str, bool] = Field(
        ..., description="Load status of each model component"
    )
    version: str = Field(..., description="API version string")


class ErrorResponse(BaseModel):
    """Structured error response."""

    error: str = Field(..., description="Error type or short summary")
    detail: str | None = Field(None, description="Detailed error description")
