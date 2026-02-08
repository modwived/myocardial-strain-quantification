"""Pipeline orchestration service for myocardial strain quantification."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch

from app.config import Settings
from strain.data.loader import load_dicom_series, load_nifti
from strain.data.preprocessing import center_crop, normalize_intensity, resample_image
from strain.mechanics.global_strain import compute_global_strain
from strain.mechanics.segmental_strain import AHA_SEGMENT_NAMES, assign_aha_segments
from strain.models.motion.carmen import CarMEN
from strain.models.segmentation.unet import UNet
from strain.risk.features import extract_strain_features
from strain.risk.model import StrainRiskModel

logger = logging.getLogger(__name__)


class StrainPipeline:
    """Orchestrates the full myocardial strain analysis pipeline.

    Loads all trained models once at startup and reuses them for every
    incoming request.  The pipeline accepts either a NIfTI file or a
    directory of DICOM files and returns a structured result dictionary.
    """

    def __init__(self, config: Settings) -> None:
        self.config = config
        self.seg_model: UNet | None = None
        self.motion_model: CarMEN | None = None
        self.risk_model: StrainRiskModel | None = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._models_loaded = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        """Load all trained model checkpoints from disk.

        Called once during application startup.  If a checkpoint file is
        missing the corresponding model attribute stays ``None`` and the
        pipeline degrades gracefully (e.g. uses a threshold-based risk
        classifier instead of the ML model).
        """
        model_dir = self.config.model_path

        # --- Segmentation model ---
        seg_path = model_dir / self.config.seg_checkpoint
        if seg_path.exists():
            logger.info("Loading segmentation model from %s", seg_path)
            self.seg_model = UNet(in_channels=1, num_classes=4)
            state = torch.load(str(seg_path), map_location=self.device, weights_only=False)
            self.seg_model.load_state_dict(state)
            self.seg_model.to(self.device).eval()
            logger.info("Segmentation model loaded on %s", self.device)
        else:
            logger.warning(
                "Segmentation checkpoint not found at %s — "
                "instantiating default model (untrained)",
                seg_path,
            )
            self.seg_model = UNet(in_channels=1, num_classes=4)
            self.seg_model.to(self.device).eval()

        # --- Motion estimation model ---
        motion_path = model_dir / self.config.motion_checkpoint
        if motion_path.exists():
            logger.info("Loading motion model from %s", motion_path)
            self.motion_model = CarMEN(in_channels=1)
            state = torch.load(str(motion_path), map_location=self.device, weights_only=False)
            self.motion_model.load_state_dict(state)
            self.motion_model.to(self.device).eval()
            logger.info("Motion model loaded on %s", self.device)
        else:
            logger.warning(
                "Motion checkpoint not found at %s — "
                "instantiating default model (untrained)",
                motion_path,
            )
            self.motion_model = CarMEN(in_channels=1)
            self.motion_model.to(self.device).eval()

        # --- Risk prediction model ---
        risk_path = model_dir / self.config.risk_checkpoint
        if risk_path.exists():
            logger.info("Loading risk model from %s", risk_path)
            self.risk_model = StrainRiskModel(model_path=risk_path)
        else:
            logger.warning(
                "Risk checkpoint not found at %s — using threshold classifier",
                risk_path,
            )
            self.risk_model = StrainRiskModel()

        self._models_loaded = True
        logger.info("All models initialized (device=%s)", self.device)

    @property
    def models_loaded_status(self) -> dict[str, bool]:
        """Return per-model load status for the health endpoint."""
        return {
            "segmentation": self.seg_model is not None,
            "motion": self.motion_model is not None,
            "risk": self.risk_model is not None,
        }

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_study(file_path: Path) -> tuple[np.ndarray, dict]:
        """Load a cardiac MRI study from a file or directory.

        Supports NIfTI (``.nii``, ``.nii.gz``) and DICOM directories.

        Returns:
            Tuple of (image_array, metadata_dict).
        """
        if file_path.is_dir():
            # Assume directory of DICOM files
            logger.info("Loading DICOM series from %s", file_path)
            return load_dicom_series(file_path)

        suffix = "".join(file_path.suffixes).lower()
        if suffix in (".nii", ".nii.gz"):
            logger.info("Loading NIfTI file %s", file_path)
            return load_nifti(file_path)

        # Try as DICOM directory anyway (user may have passed parent dir)
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            "Expected .nii, .nii.gz, or a DICOM directory."
        )

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(data: np.ndarray, metadata: dict) -> np.ndarray:
        """Normalize, resample, and crop the cardiac image data.

        Args:
            data: Raw image array, shape (T, H, W) or (T, D, H, W).
            metadata: Dictionary with at least ``spacing``.

        Returns:
            Pre-processed array of shape (T, H', W') ready for inference.
        """
        spacing = metadata.get("spacing", (1.25, 1.25))
        processed = normalize_intensity(data.astype(np.float32))
        processed = resample_image(processed, current_spacing=spacing[:2])
        processed = center_crop(processed, crop_size=128)
        return processed

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _segment_frames(self, frames: np.ndarray) -> np.ndarray:
        """Run segmentation on every temporal frame.

        Args:
            frames: (T, H, W) pre-processed image.

        Returns:
            (T, H, W) integer label map (0=bg, 1=LV, 2=myo, 3=RV).
        """
        T, H, W = frames.shape
        seg_maps = np.zeros((T, H, W), dtype=np.int64)

        with torch.no_grad():
            for t in range(T):
                frame_tensor = (
                    torch.from_numpy(frames[t])
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(self.device)
                )  # (1, 1, H, W)
                logits = self.seg_model(frame_tensor)  # (1, C, H, W)
                seg_maps[t] = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        return seg_maps

    def _estimate_motion(self, frames: np.ndarray) -> np.ndarray:
        """Estimate displacement fields for the full cardiac cycle.

        Computes the displacement from the first (ED) frame to every
        subsequent frame, yielding cumulative displacements.

        Args:
            frames: (T, H, W) pre-processed images.

        Returns:
            (T, 2, H, W) cumulative displacement fields.
        """
        T, H, W = frames.shape
        displacements = np.zeros((T, 2, H, W), dtype=np.float32)

        reference = (
            torch.from_numpy(frames[0])
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            for t in range(1, T):
                target = (
                    torch.from_numpy(frames[t])
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(self.device)
                )
                disp = self.motion_model(reference, target)  # (1, 2, H, W)
                displacements[t] = disp.squeeze(0).cpu().numpy()

        return displacements

    # ------------------------------------------------------------------
    # Volume computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_volumes(
        segmentations: np.ndarray, spacing: tuple[float, ...]
    ) -> dict[str, float]:
        """Compute LV and RV volumes from segmentation maps.

        Uses a simple voxel-counting approach multiplied by voxel volume.

        Args:
            segmentations: (T, H, W) label maps.
            spacing: Pixel spacing (sx, sy) in mm.

        Returns:
            Dictionary with EDV, ESV, LVEF, and optionally RV-EDV.
        """
        pixel_area_mm2 = float(spacing[0]) * float(spacing[1])

        # LV cavity = label 1
        lv_areas = np.array(
            [(seg == 1).sum() * pixel_area_mm2 for seg in segmentations]
        )
        # Approximate volume by assuming 8 mm slice thickness
        slice_thickness = 8.0  # mm
        lv_volumes = lv_areas * slice_thickness / 1000.0  # mL

        lv_edv = float(lv_volumes.max())
        lv_esv = float(lv_volumes.min())
        lvef = ((lv_edv - lv_esv) / lv_edv * 100.0) if lv_edv > 0 else 0.0

        # RV cavity = label 3
        rv_areas = np.array(
            [(seg == 3).sum() * pixel_area_mm2 for seg in segmentations]
        )
        rv_volumes = rv_areas * slice_thickness / 1000.0
        rv_edv = float(rv_volumes.max()) if rv_volumes.max() > 0 else None

        return {
            "lv_edv_ml": round(lv_edv, 2),
            "lv_esv_ml": round(lv_esv, 2),
            "lvef_percent": round(lvef, 1),
            "rv_edv_ml": round(rv_edv, 2) if rv_edv is not None else None,
        }

    # ------------------------------------------------------------------
    # Strain computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_strain_timeseries(
        displacements: np.ndarray, myocardial_masks: np.ndarray
    ) -> dict:
        """Compute global strain curves over the cardiac cycle.

        Args:
            displacements: (T, 2, H, W) cumulative displacement fields.
            myocardial_masks: (T, H, W) binary myocardial masks.

        Returns:
            Dictionary with ``global`` (peak GCS/GRS) and ``timeseries``.
        """
        T = displacements.shape[0]
        gcs_curve: list[float] = []
        grs_curve: list[float] = []

        for t in range(T):
            mask_t = myocardial_masks[t]
            if mask_t.sum() == 0:
                gcs_curve.append(0.0)
                grs_curve.append(0.0)
                continue
            strain_t = compute_global_strain(displacements[t], mask_t)
            gcs_curve.append(strain_t["GCS"])
            grs_curve.append(strain_t["GRS"])

        # Peak strain (most negative GCS, most positive GRS)
        peak_gcs = float(min(gcs_curve)) if gcs_curve else 0.0
        peak_grs = float(max(grs_curve)) if grs_curve else 0.0

        # Approximate time axis assuming 30 ms temporal resolution
        time_ms = [t * 30.0 for t in range(T)]

        return {
            "global": {"GCS": round(peak_gcs, 2), "GRS": round(peak_grs, 2)},
            "timeseries": {
                "time_ms": time_ms,
                "GCS": [round(v, 2) for v in gcs_curve],
                "GRS": [round(v, 2) for v in grs_curve],
            },
        }

    @staticmethod
    def _compute_segmental(
        displacement: np.ndarray, myocardial_mask: np.ndarray
    ) -> list[dict]:
        """Compute per-segment strain from the peak-systolic frame.

        Args:
            displacement: (2, H, W) displacement field at peak systole.
            myocardial_mask: (H, W) binary myocardial mask.

        Returns:
            List of segment dictionaries.
        """
        segments_map = assign_aha_segments(myocardial_mask, slice_level="mid")
        segment_list: list[dict] = []

        for seg_id in sorted(AHA_SEGMENT_NAMES.keys()):
            seg_mask = (segments_map == seg_id).astype(np.float64)
            if seg_mask.sum() == 0:
                continue
            strain = compute_global_strain(displacement, seg_mask)
            segment_list.append(
                {
                    "segment": seg_id,
                    "name": AHA_SEGMENT_NAMES[seg_id],
                    "GCS": round(strain["GCS"], 2),
                    "GRS": round(strain["GRS"], 2),
                }
            )

        return segment_list

    # ------------------------------------------------------------------
    # Risk assessment
    # ------------------------------------------------------------------

    def _assess_risk(
        self,
        global_strain: dict[str, float],
        segmental: list[dict],
        volumes: dict[str, float],
    ) -> dict[str, float | str]:
        """Run the risk prediction model and generate interpretation text.

        Args:
            global_strain: Peak global strain values.
            segmental: Per-segment strain results.
            volumes: LV/RV volume metrics.

        Returns:
            Risk assessment dictionary with score, category, and interpretation.
        """
        seg_dict = {
            s["segment"]: {"GCS": s["GCS"], "GRS": s["GRS"]} for s in segmental
        }
        lv_vols = {
            "EDV": volumes.get("lv_edv_ml", 0.0),
            "ESV": volumes.get("lv_esv_ml", 0.0),
        }
        features = extract_strain_features(global_strain, seg_dict, lv_vols)
        risk = self.risk_model.predict(features)

        # Add human-readable interpretation
        category = risk.get("risk_category", "unknown")
        gcs = global_strain.get("GCS", 0.0)
        lvef = volumes.get("lvef_percent", 0.0)
        interpretation_map = {
            "low": (
                f"Low risk of MACE. GCS={gcs:.1f}%, LVEF={lvef:.1f}%. "
                "Strain values are within normal limits."
            ),
            "intermediate": (
                f"Intermediate risk of MACE. GCS={gcs:.1f}%, LVEF={lvef:.1f}%. "
                "Some strain parameters are mildly abnormal; closer follow-up is recommended."
            ),
            "high": (
                f"High risk of MACE. GCS={gcs:.1f}%, LVEF={lvef:.1f}%. "
                "Significantly impaired myocardial function detected. "
                "Urgent clinical evaluation is recommended."
            ),
        }
        risk["interpretation"] = interpretation_map.get(
            category,
            f"Risk category: {category}. GCS={gcs:.1f}%, LVEF={lvef:.1f}%.",
        )
        return risk

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def analyze(self, file_path: Path) -> dict:
        """Run the full strain quantification pipeline on an uploaded study.

        Steps:
            1. Load cardiac MRI data (DICOM or NIfTI).
            2. Pre-process (normalize, resample, crop).
            3. Segment all cardiac phases.
            4. Estimate inter-frame motion (displacement fields).
            5. Compute global and segmental strain curves.
            6. Compute LV/RV volumes.
            7. Assess clinical risk.

        Args:
            file_path: Path to a NIfTI file or a directory of DICOM files.

        Returns:
            Structured result dictionary suitable for ``AnalysisResponse``.
        """
        start = time.time()

        # 1. Load
        logger.info("Loading study from %s", file_path)
        data, metadata = self._load_study(file_path)
        logger.info("Loaded data with shape %s", data.shape)

        # 2. Pre-process
        preprocessed = self._preprocess(data, metadata)
        logger.info("Pre-processed to shape %s", preprocessed.shape)

        # Ensure 3-D (T, H, W) — squeeze extra dims if needed
        if preprocessed.ndim == 4:
            # (T, D, H, W) — take middle slice
            mid = preprocessed.shape[1] // 2
            preprocessed = preprocessed[:, mid, :, :]
        if preprocessed.ndim == 2:
            preprocessed = preprocessed[np.newaxis, :, :]

        T = preprocessed.shape[0]

        # 3. Segment
        logger.info("Running segmentation on %d frames", T)
        segmentations = self._segment_frames(preprocessed)

        # 4. Motion estimation
        logger.info("Estimating motion for %d frames", T)
        displacements = self._estimate_motion(preprocessed)

        # 5. Strain computation
        myocardial_masks = (segmentations == 2).astype(np.float64)
        logger.info("Computing strain curves")
        strain_results = self._compute_strain_timeseries(displacements, myocardial_masks)

        # Peak systolic frame (frame with most negative GCS)
        gcs_values = strain_results["timeseries"]["GCS"]
        peak_frame = int(np.argmin(gcs_values)) if len(gcs_values) > 0 else 0
        segmental = self._compute_segmental(
            displacements[peak_frame], myocardial_masks[peak_frame]
        )

        # 6. Volumes
        spacing = metadata.get("spacing", (1.25, 1.25))
        volumes = self._compute_volumes(segmentations, spacing)

        # 7. Risk
        logger.info("Assessing clinical risk")
        risk = self._assess_risk(strain_results["global"], segmental, volumes)

        elapsed = round(time.time() - start, 2)
        logger.info("Pipeline completed in %.2f s", elapsed)

        return {
            "global_strain": strain_results["global"],
            "segmental_strain": segmental,
            "risk_assessment": risk,
            "volumes": volumes,
            "timeseries": strain_results["timeseries"],
            "processing_time_sec": elapsed,
        }
