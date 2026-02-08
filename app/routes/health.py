"""Health check endpoint."""

from __future__ import annotations

import logging

import torch
from fastapi import APIRouter

from app.schemas import HealthResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description=(
        "Returns the operational status of the service including GPU "
        "availability and model loading status."
    ),
)
def health_check() -> HealthResponse:
    """Report service health, GPU availability, and model load status."""
    # Lazy import to avoid circular dependency at module level
    from app.main import pipeline, settings

    gpu_available = torch.cuda.is_available()
    models_status = pipeline.models_loaded_status

    all_loaded = all(models_status.values())
    status = "ok" if all_loaded else "degraded"

    if not all_loaded:
        logger.warning("Health check: service degraded — models_loaded=%s", models_status)

    return HealthResponse(
        status=status,
        gpu_available=gpu_available,
        models_loaded=models_status,
        version=settings.api_version,
    )
