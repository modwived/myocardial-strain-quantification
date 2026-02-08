# Agent: Phase 6 — API & Deployment

## Mission
Wire together all pipeline stages into a production REST API. Handle file uploads, orchestrate the full analysis pipeline, return structured results, and ensure robust error handling, logging, and monitoring.

## Status: IN PROGRESS

## Files You Own
- `app/main.py` — FastAPI application (skeleton exists)
- `app/routes/analysis.py` — /analyze endpoint (skeleton exists)
- `app/routes/health.py` — /health endpoint (skeleton exists)
- `app/schemas.py` — Pydantic request/response models (CREATE)
- `app/services.py` — pipeline orchestration (CREATE)
- `app/config.py` — application configuration (CREATE)
- `Dockerfile` — container image (exists, may need updates)
- `docker-compose.yml` — orchestration (exists, may need updates)
- `tests/test_api.py` — API integration tests (CREATE)

## Detailed Requirements

### 1. schemas.py — CREATE
```python
from pydantic import BaseModel

class GlobalStrain(BaseModel):
    GLS: float | None = None
    GCS: float
    GRS: float

class SegmentalStrain(BaseModel):
    segment: int
    name: str
    GCS: float
    GRS: float

class RiskAssessment(BaseModel):
    risk_score: float
    risk_category: str  # "low", "intermediate", "high"
    interpretation: str

class StrainTimeseries(BaseModel):
    time_ms: list[float]
    GCS: list[float]
    GRS: list[float]
    GLS: list[float] | None = None

class VolumeMetrics(BaseModel):
    lv_edv_ml: float
    lv_esv_ml: float
    lvef_percent: float
    rv_edv_ml: float | None = None

class AnalysisResponse(BaseModel):
    patient_id: str | None = None
    global_strain: GlobalStrain
    segmental_strain: list[SegmentalStrain]
    risk_assessment: RiskAssessment
    volumes: VolumeMetrics | None = None
    timeseries: StrainTimeseries | None = None
    processing_time_sec: float

class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    models_loaded: dict[str, bool]
    version: str

class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
```

### 2. services.py — CREATE
```python
import time
from pathlib import Path

class StrainPipeline:
    """Orchestrates the full analysis pipeline.

    Loads models once at startup, reuses for all requests.
    """

    def __init__(self, config):
        self.config = config
        self.seg_model = None
        self.motion_model = None
        self.risk_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_models(self):
        """Load all trained models from disk. Called once at startup."""
        # Load segmentation model
        # Load motion estimation model
        # Load risk prediction model

    def analyze(self, file_path: Path) -> dict:
        """Run full pipeline on uploaded study.

        Steps:
        1. Load data (DICOM or NIfTI)
        2. Preprocess (normalize, resample, crop)
        3. Segment all cardiac phases
        4. Estimate motion (displacement fields)
        5. Compute strain (GCS, GRS, GLS)
        6. Predict risk
        7. Generate report

        Returns structured result dict.
        """
        start = time.time()

        # Step 1: Load
        data, metadata = load_study(file_path)

        # Step 2: Preprocess
        preprocessed = preprocess_pipeline(data, metadata)

        # Step 3: Segment
        segmentations = predict_volume(self.seg_model, preprocessed, ...)
        volumes = compute_volumes(segmentations, metadata['spacing'])

        # Step 4: Motion
        displacements = predict_cardiac_cycle(self.motion_model, preprocessed, ...)
        cumulative = compute_cumulative_displacement(displacements)

        # Step 5: Strain
        myocardial_masks = (segmentations == 2)
        strain_results = compute_global_strain_timeseries(cumulative, myocardial_masks)
        segmental = compute_segmental_strain(...)

        # Step 6: Risk
        features = extract_strain_features(strain_results, volumes)
        risk = self.risk_model.predict(features)

        elapsed = time.time() - start

        return {
            "global_strain": strain_results["global"],
            "segmental_strain": segmental,
            "risk_assessment": risk,
            "volumes": volumes,
            "timeseries": strain_results["timeseries"],
            "processing_time_sec": round(elapsed, 2),
        }
```

### 3. config.py — CREATE
```python
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_path: Path = Path("./models")
    data_dir: Path = Path("./data")
    log_level: str = "info"
    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_size_mb: int = 500

    # Model checkpoint filenames
    seg_checkpoint: str = "segmentation_best.pt"
    motion_checkpoint: str = "motion_best.pt"
    risk_checkpoint: str = "risk_model.joblib"

    class Config:
        env_file = ".env"
```

### 4. main.py — IMPROVE existing
```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.config import Settings
from app.services import StrainPipeline

settings = Settings()
pipeline = StrainPipeline(settings)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load all models
    pipeline.load_models()
    yield
    # Shutdown: cleanup

app = FastAPI(
    title="Myocardial Strain Quantification",
    version="0.1.0",
    lifespan=lifespan,
)
```

### 5. routes/analysis.py — IMPROVE existing
```python
@router.post("", response_model=AnalysisResponse)
async def analyze(file: UploadFile):
    """Upload a cardiac MRI study (DICOM .zip or .nii.gz) for strain analysis."""
    # 1. Validate file type
    # 2. Save to temp directory
    # 3. If zip, extract DICOM files
    # 4. Run pipeline.analyze()
    # 5. Clean up temp files
    # 6. Return structured response
```

### 6. routes/health.py — IMPROVE existing
```python
@router.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "ok",
        "gpu_available": torch.cuda.is_available(),
        "models_loaded": {
            "segmentation": pipeline.seg_model is not None,
            "motion": pipeline.motion_model is not None,
            "risk": pipeline.risk_model is not None,
        },
        "version": "0.1.0",
    }
```

### 7. Update Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 8. Update docker-compose.yml
```yaml
services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models
      - DATA_DIR=/app/data
      - LOG_LEVEL=info
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### 9. tests/test_api.py
```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_analyze_no_file():
    response = client.post("/analyze")
    assert response.status_code == 422  # validation error

def test_analyze_invalid_file():
    # Upload a non-medical file
    response = client.post("/analyze", files={"file": ("test.txt", b"hello", "text/plain")})
    assert response.status_code == 400  # or appropriate error
```

## Interface Contract
```python
# Receives from ALL other phases:
# - Phase 1: data loading + preprocessing functions
# - Phase 2: segmentation model + volume computation
# - Phase 3: motion estimation model
# - Phase 4: strain computation functions
# - Phase 5: risk model + report generation

# Exposes to external consumers:
# POST /analyze → AnalysisResponse (JSON)
# GET /health → HealthResponse (JSON)
```

## If You Get Stuck
- FastAPI file upload: `from fastapi import File, UploadFile`
- Temp files: `import tempfile; with tempfile.TemporaryDirectory() as tmp:`
- Zip extraction: `import zipfile; zipfile.ZipFile(path).extractall(dest)`
- For GPU in Docker: install nvidia-container-toolkit on host
- FastAPI lifespan is the modern way to handle startup/shutdown (replaces on_event)
- If models aren't loaded yet (no checkpoints), return graceful error, not crash
- pydantic-settings for env var config: `pip install pydantic-settings`
- Add CORS middleware if frontend will call API: `from fastapi.middleware.cors import CORSMiddleware`
