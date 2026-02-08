"""Feature extraction for risk stratification."""

import numpy as np


def extract_strain_features(
    global_strain: dict[str, float],
    segmental_strain: dict[int, dict[str, float]] | None = None,
    lv_volumes: dict[str, float] | None = None,
) -> dict[str, float]:
    """Extract features for risk prediction from strain analysis results.

    Args:
        global_strain: Dictionary with GLS, GCS, GRS values.
        segmental_strain: Optional per-segment strain values.
        lv_volumes: Optional LV volumes (EDV, ESV in mL).

    Returns:
        Feature dictionary for the risk model.
    """
    features = {}

    # Global strain features
    features["gls"] = global_strain.get("GLS", 0.0)
    features["gcs"] = global_strain.get("GCS", 0.0)
    features["grs"] = global_strain.get("GRS", 0.0)

    # LV function
    if lv_volumes:
        edv = lv_volumes.get("EDV", 0.0)
        esv = lv_volumes.get("ESV", 0.0)
        features["edv"] = edv
        features["esv"] = esv
        features["lvef"] = ((edv - esv) / edv * 100) if edv > 0 else 0.0

    # Segmental strain heterogeneity
    if segmental_strain:
        seg_gcs = [v.get("GCS", 0.0) for v in segmental_strain.values()]
        features["gcs_std"] = float(np.std(seg_gcs))
        features["gcs_min"] = float(np.min(seg_gcs))
        features["gcs_max"] = float(np.max(seg_gcs))

    return features
