"""FastAPI application for myocardial strain quantification."""

from fastapi import FastAPI

from app.routes.analysis import router as analysis_router
from app.routes.health import router as health_router

app = FastAPI(
    title="Myocardial Strain Quantification",
    description="AI-based myocardial strain quantification for risk stratification following acute myocardial infarction.",
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(analysis_router, prefix="/analyze", tags=["analysis"])
