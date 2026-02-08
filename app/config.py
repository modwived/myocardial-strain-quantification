"""Application configuration loaded from environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration for the myocardial strain quantification API.

    Values are loaded from environment variables (case-insensitive) and
    optionally from a `.env` file in the working directory.
    """

    # Directories
    model_path: Path = Path("./models")
    data_dir: Path = Path("./data")

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # Upload limits
    max_upload_size_mb: int = 500

    # Model checkpoint filenames
    seg_checkpoint: str = "segmentation_best.pt"
    motion_checkpoint: str = "motion_best.pt"
    risk_checkpoint: str = "risk_model.joblib"

    # CORS
    cors_origins: list[str] = ["*"]

    # API metadata
    api_version: str = "0.1.0"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }
