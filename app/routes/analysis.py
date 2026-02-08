"""Analysis endpoint for strain quantification."""

from __future__ import annotations

import logging
import shutil
import tempfile
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from app.schemas import AnalysisResponse, ErrorResponse

router = APIRouter()
logger = logging.getLogger(__name__)

# Allowed upload extensions
_ALLOWED_EXTENSIONS = {".zip", ".nii", ".gz", ".nii.gz", ".dcm"}
_MAX_UPLOAD_BYTES: int | None = None  # set from settings on first call


def _validate_filename(filename: str | None) -> str:
    """Return the lower-cased suffix(es) and raise on unsupported files."""
    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file has no filename.",
        )
    suffix = "".join(Path(filename).suffixes).lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Accepted formats: {', '.join(sorted(_ALLOWED_EXTENSIONS))}."
            ),
        )
    return suffix


def _find_study_path(base_dir: Path) -> Path:
    """Find the actual study data inside an extracted directory.

    If the directory contains NIfTI files, return the first one.
    Otherwise return the directory itself (assumed to be DICOM).
    """
    nifti_files = list(base_dir.rglob("*.nii.gz")) + list(base_dir.rglob("*.nii"))
    if nifti_files:
        return nifti_files[0]

    # Check for DICOM files (files with .dcm or without extension)
    dcm_files = list(base_dir.rglob("*.dcm"))
    if dcm_files:
        return dcm_files[0].parent

    # Fallback: look for any directory with files (may be extensionless DICOM)
    for child in base_dir.iterdir():
        if child.is_dir():
            files = list(child.iterdir())
            if files and all(f.is_file() for f in files):
                return child

    # Last resort: return the base directory
    return base_dir


@router.post(
    "",
    response_model=AnalysisResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Analyze a cardiac MRI study",
    description=(
        "Upload a cardiac MRI study as a DICOM `.zip` archive or a NIfTI "
        "`.nii`/`.nii.gz` file. The endpoint runs the full strain "
        "quantification pipeline and returns structured results."
    ),
)
async def analyze(file: UploadFile):
    """Upload a cardiac MRI study and return strain analysis results.

    Accepts DICOM zip or NIfTI files.
    """
    # Lazy import to avoid circular imports at module level
    from app.main import pipeline, settings

    # ------------------------------------------------------------------
    # 1. Validate file type
    # ------------------------------------------------------------------
    suffix = _validate_filename(file.filename)

    # ------------------------------------------------------------------
    # 2. Save upload to a temporary directory
    # ------------------------------------------------------------------
    tmp_dir = tempfile.mkdtemp(prefix="strain_upload_")
    tmp_path = Path(tmp_dir)
    try:
        upload_path = tmp_path / file.filename
        logger.info("Saving upload to %s", upload_path)

        # Stream file to disk to handle large uploads
        max_bytes = settings.max_upload_size_mb * 1024 * 1024
        total_written = 0
        with open(upload_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)  # 1 MB chunks
                if not chunk:
                    break
                total_written += len(chunk)
                if total_written > max_bytes:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"File exceeds maximum upload size of "
                            f"{settings.max_upload_size_mb} MB."
                        ),
                    )
                f.write(chunk)

        logger.info("Upload saved (%d bytes)", total_written)

        # ------------------------------------------------------------------
        # 3. Extract zip archives
        # ------------------------------------------------------------------
        study_path: Path
        if suffix == ".zip":
            extract_dir = tmp_path / "extracted"
            extract_dir.mkdir()
            try:
                with zipfile.ZipFile(upload_path, "r") as zf:
                    zf.extractall(extract_dir)
            except zipfile.BadZipFile:
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded .zip file is corrupt or not a valid zip archive.",
                )
            study_path = _find_study_path(extract_dir)
        elif suffix in (".nii", ".nii.gz"):
            study_path = upload_path
        else:
            # Single DICOM file — place in its own directory
            study_path = upload_path.parent

        logger.info("Study path resolved to %s", study_path)

        # ------------------------------------------------------------------
        # 4. Run pipeline
        # ------------------------------------------------------------------
        result = pipeline.analyze(study_path)

        # ------------------------------------------------------------------
        # 5. Return structured response
        # ------------------------------------------------------------------
        return AnalysisResponse(
            patient_id=None,
            global_strain=result["global_strain"],
            segmental_strain=result["segmental_strain"],
            risk_assessment=result["risk_assessment"],
            volumes=result.get("volumes"),
            timeseries=result.get("timeseries"),
            processing_time_sec=result["processing_time_sec"],
        )

    except HTTPException:
        # Re-raise FastAPI HTTP exceptions as-is
        raise
    except Exception as exc:
        logger.exception("Pipeline failed for upload %s", file.filename)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis pipeline error: {exc}",
        )
    finally:
        # ------------------------------------------------------------------
        # 6. Clean up temp files
        # ------------------------------------------------------------------
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info("Cleaned up temp directory %s", tmp_dir)
