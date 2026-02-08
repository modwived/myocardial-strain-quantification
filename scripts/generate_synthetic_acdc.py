"""Generate a synthetic ACDC-format dataset for development and testing.

Creates fake cardiac MRI volumes with synthetic segmentation masks that
mimic the directory structure and file naming conventions of the ACDC
challenge dataset. This allows full pipeline testing without downloading
real data.

Usage:
    python scripts/generate_synthetic_acdc.py [--output-dir data/ACDC_synthetic] [--num-patients 20]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def _make_cardiac_phantom(
    nx: int = 128,
    ny: int = 128,
    nz: int = 10,
    nt: int = 20,
    spacing: tuple[float, float, float] = (1.5, 1.5, 5.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Create a synthetic 4D cardiac cine volume with segmentation masks.

    Returns:
        cine_4d: (T, Z, Y, X) float32 cine volume
        seg_ed: (Z, Y, X) uint8 segmentation at end-diastole (frame 0)
        seg_es: (Z, Y, X) uint8 segmentation at end-systole
        ed_frame: ED frame index (always 0)
        es_frame: ES frame index
    """
    rng = np.random.default_rng()

    # Center of the LV
    cx, cy = nx // 2, ny // 2

    # LV cavity radius, myocardial thickness, RV params
    lv_inner_ed = rng.uniform(12, 18)
    myo_thickness = rng.uniform(6, 10)
    lv_outer_ed = lv_inner_ed + myo_thickness
    rv_offset_x = rng.uniform(-20, -15)  # RV is to the right of LV
    rv_inner_ed = rng.uniform(10, 15)
    rv_thickness = rng.uniform(3, 5)

    # Contraction: LV shrinks at ES
    contraction = rng.uniform(0.6, 0.8)
    es_frame = nt // 2

    cine_4d = np.zeros((nt, nz, ny, nx), dtype=np.float32)
    seg_ed = np.zeros((nz, ny, nx), dtype=np.uint8)
    seg_es = np.zeros((nz, ny, nx), dtype=np.uint8)

    yy, xx = np.mgrid[:ny, :nx]

    for z in range(nz):
        # Taper radii near apex (last slices)
        taper = max(0.3, 1.0 - 0.07 * z)

        for t in range(nt):
            # Smooth contraction curve (sinusoidal)
            phase = np.sin(np.pi * t / nt) ** 2
            scale = 1.0 - (1.0 - contraction) * phase

            lv_inner = lv_inner_ed * scale * taper
            lv_outer = lv_outer_ed * taper  # epicardium doesn't contract much
            rv_inner = rv_inner_ed * taper
            rv_outer = (rv_inner_ed + rv_thickness) * taper

            # Distance maps
            dist_lv = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            dist_rv = np.sqrt((xx - (cx + rv_offset_x)) ** 2 + (yy - cy) ** 2)

            # Create image: blood pool bright, myocardium medium, background dark
            frame = rng.normal(50, 10, (ny, nx)).astype(np.float32)

            # LV cavity (bright blood)
            lv_mask = dist_lv < lv_inner
            frame[lv_mask] = rng.normal(200, 15, lv_mask.sum())

            # Myocardium (intermediate signal)
            myo_mask = (dist_lv >= lv_inner) & (dist_lv < lv_outer)
            frame[myo_mask] = rng.normal(120, 10, myo_mask.sum())

            # RV cavity (bright blood)
            rv_cav_mask = dist_rv < rv_inner
            frame[rv_cav_mask] = rng.normal(190, 15, rv_cav_mask.sum())

            # RV wall
            rv_wall_mask = (dist_rv >= rv_inner) & (dist_rv < rv_outer)
            frame[rv_wall_mask] = rng.normal(110, 10, rv_wall_mask.sum())

            cine_4d[t, z] = frame

            # Segmentation at ED (t=0) and ES
            if t == 0:
                seg_ed[z][rv_cav_mask] = 1  # RV cavity
                seg_ed[z][myo_mask] = 2  # Myocardium
                seg_ed[z][lv_mask] = 3  # LV cavity
            if t == es_frame:
                seg_es[z][rv_cav_mask] = 1
                seg_es[z][myo_mask] = 2
                seg_es[z][lv_mask] = 3

    return cine_4d, seg_ed, seg_es, 0, es_frame


def _save_nifti(array: np.ndarray, path: Path, spacing: tuple[float, ...]) -> None:
    """Save a numpy array as NIfTI with given spacing."""
    img = sitk.GetImageFromArray(array)
    if len(spacing) >= 3:
        img.SetSpacing(spacing[:3])
    elif len(spacing) == 2:
        img.SetSpacing((*spacing, 1.0))
    sitk.WriteImage(img, str(path))


def generate_patient(
    patient_id: str,
    output_dir: Path,
    group: str,
) -> None:
    """Generate one synthetic patient in ACDC format."""
    patient_dir = output_dir / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)

    spacing = (1.5, 1.5, 5.0)
    cine_4d, seg_ed, seg_es, ed_idx, es_idx = _make_cardiac_phantom(spacing=spacing)

    ed_frame_str = f"{ed_idx + 1:02d}"
    es_frame_str = f"{es_idx + 1:02d}"

    # Save 4D cine
    _save_nifti(cine_4d, patient_dir / f"{patient_id}_4d.nii.gz", spacing)

    # Save ED frame + ground truth
    _save_nifti(cine_4d[ed_idx], patient_dir / f"{patient_id}_frame{ed_frame_str}.nii.gz", spacing)
    _save_nifti(seg_ed, patient_dir / f"{patient_id}_frame{ed_frame_str}_gt.nii.gz", spacing)

    # Save ES frame + ground truth
    _save_nifti(cine_4d[es_idx], patient_dir / f"{patient_id}_frame{es_frame_str}.nii.gz", spacing)
    _save_nifti(seg_es, patient_dir / f"{patient_id}_frame{es_frame_str}_gt.nii.gz", spacing)

    # Info file (ACDC format)
    info = (
        f"ED: {ed_idx}\n"
        f"ES: {es_idx}\n"
        f"Group: {group}\n"
        f"Height: 170\n"
        f"NbFrame: {cine_4d.shape[0]}\n"
        f"Weight: 75\n"
    )
    (patient_dir / "Info.cfg").write_text(info)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic ACDC dataset")
    parser.add_argument("--output-dir", type=str, default="data/ACDC_synthetic")
    parser.add_argument("--num-patients", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    groups = ["NOR", "MINF", "DCM", "HCM", "RV"]

    print(f"Generating {args.num_patients} synthetic patients in {output_dir}")

    for i in range(args.num_patients):
        patient_id = f"patient{i + 1:03d}"
        group = groups[i % len(groups)]
        print(f"  {patient_id} ({group})...", end=" ", flush=True)
        generate_patient(patient_id, output_dir, group)
        print("done")

    print(f"\nDataset generated: {args.num_patients} patients in {output_dir}")
    print("Groups: " + ", ".join(f"{g}={args.num_patients // len(groups)}" for g in groups))


if __name__ == "__main__":
    main()
