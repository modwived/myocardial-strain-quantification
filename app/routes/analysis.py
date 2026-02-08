"""Analysis endpoint for strain quantification."""

from fastapi import APIRouter, UploadFile

router = APIRouter()


@router.post("")
async def analyze(file: UploadFile):
    """Upload a cardiac MRI study and return strain analysis results.

    Accepts DICOM zip or NIfTI files.
    """
    # TODO: implement full pipeline orchestration
    return {
        "filename": file.filename,
        "status": "not_implemented",
        "message": "Pipeline integration pending. See PLAN.md for implementation details.",
    }
