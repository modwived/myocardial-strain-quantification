"""FastAPI application for myocardial strain quantification."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import Settings
from app.schemas import ErrorResponse
from app.services import StrainPipeline

# ---------------------------------------------------------------------------
# Configuration & global objects
# ---------------------------------------------------------------------------

settings = Settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

pipeline = StrainPipeline(settings)

# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup; clean up on shutdown."""
    logger.info("Starting up — loading models …")
    try:
        pipeline.load_models()
        logger.info("All models loaded successfully")
    except Exception:
        logger.exception("Error loading models — service will start in degraded mode")
    yield
    logger.info("Shutting down — releasing resources")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Myocardial Strain Quantification",
    description=(
        "AI-based myocardial strain quantification for risk stratification "
        "following acute myocardial infarction."
    ),
    version=settings.api_version,
    lifespan=lifespan,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    logger.warning("ValueError on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=400,
        content={"error": "Bad request", "detail": str(exc)},
    )


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError) -> JSONResponse:
    logger.warning("FileNotFoundError on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=404,
        content={"error": "File not found", "detail": str(exc)},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please check the server logs.",
        },
    )


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

from app.routes.analysis import router as analysis_router  # noqa: E402
from app.routes.health import router as health_router  # noqa: E402

app.include_router(health_router)
app.include_router(analysis_router, prefix="/analyze", tags=["analysis"])
