"""Load cardiac MRI data from DICOM series and NIfTI volumes.

Supports auto-detection of file format, multi-slice DICOM series grouped
by slice location and sorted by trigger time, and extraction of cardiac
metadata from DICOM headers.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk

try:
    import pydicom
except ImportError:  # pragma: no cover
    pydicom = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NIfTI loading
# ---------------------------------------------------------------------------

def load_nifti(path: str | Path) -> tuple[np.ndarray, dict]:
    """Load a NIfTI file and return the image array and metadata.

    Args:
        path: Path to .nii or .nii.gz file.

    Returns:
        Tuple of (image_array, metadata_dict).
        image_array has shape (D, H, W) for 3D or (T, D, H, W) for 4D.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the file cannot be read.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")
    if not (path.suffix == ".nii" or path.name.endswith(".nii.gz")):
        raise ValueError(f"Unsupported file format (expected .nii or .nii.gz): {path}")

    try:
        img = sitk.ReadImage(str(path))
    except Exception as exc:
        raise RuntimeError(f"Failed to read NIfTI file {path}: {exc}") from exc

    array = sitk.GetArrayFromImage(img)
    metadata = {
        "spacing": img.GetSpacing(),
        "origin": img.GetOrigin(),
        "direction": img.GetDirection(),
        "size": img.GetSize(),
    }
    return array, metadata


# ---------------------------------------------------------------------------
# DICOM loading
# ---------------------------------------------------------------------------

def load_dicom_series(directory: str | Path) -> tuple[np.ndarray, dict]:
    """Load a DICOM series from a directory.

    For a simple single-series directory this delegates to SimpleITK.  For
    multi-slice cardiac cine series, use :func:`load_dicom_multislice` instead.

    Args:
        directory: Path to directory containing DICOM files.

    Returns:
        Tuple of (image_array, metadata_dict).

    Raises:
        FileNotFoundError: If no DICOM files are found in the directory.
        RuntimeError: If the DICOM series cannot be read.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(str(directory))
    if not dicom_files:
        raise FileNotFoundError(f"No DICOM files found in {directory}")

    try:
        reader.SetFileNames(dicom_files)
        img = reader.Execute()
    except Exception as exc:
        raise RuntimeError(f"Failed to read DICOM series in {directory}: {exc}") from exc

    array = sitk.GetArrayFromImage(img)
    metadata = {
        "spacing": img.GetSpacing(),
        "origin": img.GetOrigin(),
        "direction": img.GetDirection(),
        "size": img.GetSize(),
    }
    return array, metadata


def load_dicom_multislice(directory: str | Path) -> tuple[np.ndarray, dict]:
    """Load a multi-slice cardiac cine DICOM series.

    Groups DICOM files by slice location and sorts within each group by
    trigger time, producing a 4-D array (T, Slices, H, W) or a 3-D array
    (T, H, W) for a single slice.

    Requires ``pydicom``.

    Args:
        directory: Path to directory containing DICOM files.

    Returns:
        Tuple of (image_array, metadata_dict).

    Raises:
        FileNotFoundError: If no DICOM files are found.
        ImportError: If ``pydicom`` is not installed.
        RuntimeError: If DICOM files cannot be read or are inconsistent.
    """
    if pydicom is None:
        raise ImportError("pydicom is required for multi-slice DICOM loading")

    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Collect all DICOM files
    dcm_files: list[Path] = []
    for ext in ("*.dcm", "*.DCM", "*.ima", "*.IMA"):
        dcm_files.extend(directory.glob(ext))
    # Also try files without extension (common in DICOM)
    if not dcm_files:
        dcm_files = [
            f for f in directory.iterdir()
            if f.is_file() and not f.name.startswith(".")
        ]

    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files found in {directory}")

    # Group by slice location
    slice_groups: dict[float, list[tuple[float, np.ndarray, pydicom.Dataset]]] = defaultdict(list)

    for fpath in dcm_files:
        try:
            ds = pydicom.dcmread(str(fpath))
        except Exception as exc:
            logger.warning("Skipping corrupt DICOM %s: %s", fpath, exc)
            continue

        if not hasattr(ds, "pixel_array"):
            logger.warning("Skipping DICOM without pixel data: %s", fpath)
            continue

        slice_loc = float(getattr(ds, "SliceLocation", 0.0))
        trigger_time = float(getattr(ds, "TriggerTime", 0.0))
        slice_groups[slice_loc].append((trigger_time, ds.pixel_array.astype(np.float32), ds))

    if not slice_groups:
        raise RuntimeError(f"No valid DICOM images found in {directory}")

    # Sort slices by location, frames within each slice by trigger time
    sorted_locations = sorted(slice_groups.keys())
    sorted_groups = []
    for loc in sorted_locations:
        frames = sorted(slice_groups[loc], key=lambda x: x[0])
        sorted_groups.append([f[1] for f in frames])

    # Validate consistent frame counts
    n_frames = len(sorted_groups[0])
    for i, group in enumerate(sorted_groups):
        if len(group) != n_frames:
            logger.warning(
                "Slice %d has %d frames (expected %d); padding/truncating",
                i, len(group), n_frames,
            )

    n_slices = len(sorted_groups)

    # Build 4D volume: (T, Slices, H, W) if multi-slice, or (T, H, W) for single slice
    if n_slices == 1:
        volume = np.stack(sorted_groups[0], axis=0)  # (T, H, W)
    else:
        # Stack as (Slices, T, H, W) then transpose to (T, Slices, H, W)
        slices = []
        for group in sorted_groups:
            slices.append(np.stack(group[:n_frames], axis=0))  # (T, H, W)
        volume = np.stack(slices, axis=1)  # (T, Slices, H, W)

    # Extract metadata from the first valid dataset
    first_ds = slice_groups[sorted_locations[0]][0][2]
    pixel_spacing = [float(x) for x in getattr(first_ds, "PixelSpacing", [1.0, 1.0])]

    metadata = {
        "spacing": tuple(pixel_spacing),
        "n_slices": n_slices,
        "n_frames": n_frames,
        "slice_locations": sorted_locations,
    }
    return volume, metadata


# ---------------------------------------------------------------------------
# Cardiac metadata extraction
# ---------------------------------------------------------------------------

def extract_cardiac_metadata(path: str | Path) -> dict:
    """Extract cardiac-specific metadata from DICOM headers.

    Reads the first valid DICOM file in *path* (file or directory) and
    returns heart rate, trigger time, slice location, number of phases,
    and other cardiac-relevant fields.

    Args:
        path: Path to a DICOM file or a directory containing DICOM files.

    Returns:
        Dictionary with cardiac metadata fields.  Missing fields are set to
        ``None``.

    Raises:
        ImportError: If ``pydicom`` is not installed.
        FileNotFoundError: If no DICOM files can be found.
    """
    if pydicom is None:
        raise ImportError("pydicom is required for cardiac metadata extraction")

    path = Path(path)

    # Collect DICOM paths
    if path.is_file():
        dcm_paths = [path]
    elif path.is_dir():
        dcm_paths = sorted(
            f for f in path.iterdir()
            if f.is_file() and not f.name.startswith(".")
        )
    else:
        raise FileNotFoundError(f"Path not found: {path}")

    # Try to read the first valid file
    ds = None
    for fpath in dcm_paths:
        try:
            ds = pydicom.dcmread(str(fpath))
            break
        except Exception:
            continue

    if ds is None:
        raise FileNotFoundError(f"No readable DICOM files found at {path}")

    def _get(attr: str, default=None):
        val = getattr(ds, attr, default)
        if val is not None and val != "":
            try:
                return float(val)
            except (TypeError, ValueError):
                return val
        return default

    # Count unique slice locations and trigger times across all files
    slice_locations: set[float] = set()
    trigger_times: set[float] = set()
    for fpath in dcm_paths:
        try:
            d = pydicom.dcmread(str(fpath), stop_before_pixels=True)
            sl = getattr(d, "SliceLocation", None)
            tt = getattr(d, "TriggerTime", None)
            if sl is not None:
                slice_locations.add(float(sl))
            if tt is not None:
                trigger_times.add(float(tt))
        except Exception:
            continue

    n_phases = len(trigger_times) if trigger_times else None
    n_slices = len(slice_locations) if slice_locations else None

    return {
        "heart_rate": _get("HeartRate"),
        "trigger_time": _get("TriggerTime"),
        "slice_location": _get("SliceLocation"),
        "slice_thickness": _get("SliceThickness"),
        "repetition_time": _get("RepetitionTime"),
        "echo_time": _get("EchoTime"),
        "pixel_spacing": list(getattr(ds, "PixelSpacing", [])),
        "rows": _get("Rows"),
        "columns": _get("Columns"),
        "n_phases": n_phases,
        "n_slices": n_slices,
        "patient_id": getattr(ds, "PatientID", None),
        "study_description": getattr(ds, "StudyDescription", None),
        "series_description": getattr(ds, "SeriesDescription", None),
    }


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def load_study(path: str | Path) -> tuple[np.ndarray, dict]:
    """Auto-detect format and load a cardiac MRI study.

    If *path* is a NIfTI file (.nii / .nii.gz), loads via :func:`load_nifti`.
    If *path* is a directory, attempts to load as DICOM series.  When
    ``pydicom`` is available the multi-slice loader is tried first;
    otherwise falls back to the SimpleITK series reader.

    Args:
        path: Path to a NIfTI file or a directory of DICOM files.

    Returns:
        Tuple of (image_array, metadata_dict).

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the format cannot be determined.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    # NIfTI file
    if path.is_file():
        if path.suffix == ".nii" or path.name.endswith(".nii.gz"):
            return load_nifti(path)
        raise ValueError(f"Unsupported file format: {path}")

    # Directory -> DICOM
    if path.is_dir():
        if pydicom is not None:
            try:
                return load_dicom_multislice(path)
            except Exception as exc:
                logger.info(
                    "Multi-slice DICOM loader failed (%s); falling back to SimpleITK", exc
                )
        return load_dicom_series(path)

    raise ValueError(f"Cannot determine format for: {path}")
