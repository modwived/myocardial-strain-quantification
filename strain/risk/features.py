"""Feature extraction for risk stratification.

Extracts strain, functional, and clinical features from cardiac strain
analysis results for use in MACE risk prediction models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Strain-rate helpers
# ---------------------------------------------------------------------------

def _compute_strain_rate(strain_timeseries: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Compute strain rate as the first derivative of strain w.r.t. time.

    Args:
        strain_timeseries: 1-D array of strain values over one cardiac cycle.
        time: 1-D array of time points (ms) aligned with *strain_timeseries*.

    Returns:
        Strain rate array (1/s).  Length = len(strain_timeseries) - 1.
    """
    dt = np.diff(time) / 1000.0  # ms -> s
    dt = np.where(dt == 0, 1e-6, dt)  # guard against zero intervals
    ds = np.diff(strain_timeseries)
    return ds / dt


def _extract_strain_rate_features(strain_results: dict) -> dict[str, float]:
    """Peak systolic and diastolic strain rates.

    If pre-computed values exist in *strain_results["strain_rate"]*, use them.
    Otherwise attempt to derive them from the timeseries data.
    """
    features: dict[str, float] = {}

    # Try pre-computed values first
    sr = strain_results.get("strain_rate", {})
    if sr.get("peak_systolic_sr") is not None:
        features["peak_systolic_sr"] = float(sr["peak_systolic_sr"])
        features["peak_diastolic_sr"] = float(sr.get("peak_diastolic_sr", 0.0))
        return features

    # Derive from timeseries if available
    ts = strain_results.get("timeseries", {})
    gcs_ts = ts.get("GCS")
    time = ts.get("time")
    if gcs_ts is not None and time is not None:
        gcs_ts = np.asarray(gcs_ts, dtype=float)
        time = np.asarray(time, dtype=float)
        if len(gcs_ts) >= 2 and len(time) >= 2:
            sr_curve = _compute_strain_rate(gcs_ts, time)
            # Systolic SR is the most negative rate (shortening)
            features["peak_systolic_sr"] = float(np.min(sr_curve))
            # Diastolic SR is the most positive rate (relaxation)
            features["peak_diastolic_sr"] = float(np.max(sr_curve))
            return features

    # Fallback: not available
    features["peak_systolic_sr"] = 0.0
    features["peak_diastolic_sr"] = 0.0
    return features


# ---------------------------------------------------------------------------
# Heterogeneity features
# ---------------------------------------------------------------------------

def _extract_heterogeneity_features(
    segmental_strain: dict[int, dict[str, float]] | None,
) -> dict[str, float]:
    """Coefficient of variation and spread of segmental GCS values."""
    features: dict[str, float] = {}
    if not segmental_strain:
        return features

    seg_gcs = [v.get("GCS", 0.0) for v in segmental_strain.values()]
    if not seg_gcs:
        return features

    arr = np.array(seg_gcs, dtype=float)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))

    features["gcs_std"] = std_val
    features["gcs_min"] = float(np.min(arr))
    features["gcs_max"] = float(np.max(arr))
    features["gcs_cv"] = abs(std_val / mean_val) if abs(mean_val) > 1e-9 else 0.0

    # GRS heterogeneity
    seg_grs = [v.get("GRS", 0.0) for v in segmental_strain.values() if "GRS" in v]
    if seg_grs:
        grs_arr = np.array(seg_grs, dtype=float)
        features["grs_std"] = float(np.std(grs_arr))

    return features


# ---------------------------------------------------------------------------
# Wall motion abnormality score
# ---------------------------------------------------------------------------

def compute_wall_motion_score(
    segmental_strain: dict[int, dict[str, float]] | None,
    threshold_gcs: float = -10.0,
) -> float:
    """Count the number of segments with |GCS| below *threshold_gcs*.

    A segment with GCS *greater* than *threshold_gcs* (i.e. less negative,
    closer to zero) is considered hypokinetic.

    Args:
        segmental_strain: Per-segment strain dictionary.
        threshold_gcs: GCS threshold for wall motion abnormality (default -10%).

    Returns:
        Wall motion score (count of abnormal segments).
    """
    if not segmental_strain:
        return 0.0
    count = 0
    for seg_data in segmental_strain.values():
        gcs = seg_data.get("GCS", 0.0)
        if gcs > threshold_gcs:  # GCS is negative; closer to 0 = worse
            count += 1
    return float(count)


# ---------------------------------------------------------------------------
# Temporal features
# ---------------------------------------------------------------------------

def _extract_temporal_features(strain_results: dict) -> dict[str, float]:
    """Time-to-peak strain and strain recovery ratio.

    Time to peak: time in ms from cycle start to peak (most negative) GCS.
    Recovery ratio: strain at end-diastole divided by peak strain.  A value
    close to 0 indicates full recovery; close to 1 indicates no recovery.
    """
    features: dict[str, float] = {}
    ts = strain_results.get("timeseries", {})
    gcs_ts = ts.get("GCS")
    time = ts.get("time")

    if gcs_ts is not None and time is not None:
        gcs_ts = np.asarray(gcs_ts, dtype=float)
        time = np.asarray(time, dtype=float)
        if len(gcs_ts) >= 2:
            peak_idx = int(np.argmin(gcs_ts))  # most negative
            features["time_to_peak_strain"] = float(time[peak_idx] - time[0])
            peak_val = gcs_ts[peak_idx]
            end_val = gcs_ts[-1]
            if abs(peak_val) > 1e-9:
                features["strain_recovery_ratio"] = float(abs(end_val / peak_val))
            else:
                features["strain_recovery_ratio"] = 0.0
            return features

    features["time_to_peak_strain"] = 0.0
    features["strain_recovery_ratio"] = 0.0
    return features


# ---------------------------------------------------------------------------
# Main feature extraction
# ---------------------------------------------------------------------------

def extract_strain_features(
    strain_results: dict,
    clinical_data: dict | None = None,
) -> dict[str, float]:
    """Full feature extraction for risk prediction.

    Extracts the following feature groups:

    **Strain features** (from ``strain_results``):
      - ``gls``, ``gcs``, ``grs`` -- global peak values
      - ``gcs_std``, ``gcs_min``, ``gcs_max``, ``gcs_cv`` -- segmental heterogeneity
      - ``grs_std`` -- radial strain heterogeneity
      - ``peak_systolic_sr``, ``peak_diastolic_sr`` -- strain rate
      - ``time_to_peak_strain`` (ms)
      - ``strain_recovery_ratio`` (end-diastolic / peak strain)
      - ``wall_motion_score`` (count of segments with |GCS| < 10%)

    **Functional features** (from ``strain_results`` volumes or direct keys):
      - ``lvef``, ``edv``, ``esv``, ``sv``

    **Clinical features** (if *clinical_data* provided):
      - ``age``, ``sex``, ``diabetes``, ``hypertension``, ``killip_class``,
        ``timi_flow``, ``troponin``

    Args:
        strain_results: Dictionary with keys ``global``, ``segmental``,
            ``timeseries``, ``strain_rate``, and optionally ``volumes``.
        clinical_data: Optional dictionary of clinical variables.

    Returns:
        Flat feature dictionary ``{name: value}``.
    """
    features: dict[str, float] = {}

    # -- Global strain -------------------------------------------------------
    global_strain = strain_results.get("global", {})
    features["gls"] = float(global_strain.get("GLS", 0.0))
    features["gcs"] = float(global_strain.get("GCS", 0.0))
    features["grs"] = float(global_strain.get("GRS", 0.0))

    # -- Segmental heterogeneity --------------------------------------------
    segmental_strain = strain_results.get("segmental")
    features.update(_extract_heterogeneity_features(segmental_strain))

    # -- Wall motion score ---------------------------------------------------
    features["wall_motion_score"] = compute_wall_motion_score(segmental_strain)

    # -- Strain rate features ------------------------------------------------
    features.update(_extract_strain_rate_features(strain_results))

    # -- Temporal features ---------------------------------------------------
    features.update(_extract_temporal_features(strain_results))

    # -- LV function (volumes) -----------------------------------------------
    volumes = strain_results.get("volumes", {})
    if volumes:
        edv = float(volumes.get("LV_EDV", volumes.get("EDV", 0.0)))
        esv = float(volumes.get("LV_ESV", volumes.get("ESV", 0.0)))
        features["edv"] = edv
        features["esv"] = esv
        sv = edv - esv
        features["sv"] = sv
        features["lvef"] = (sv / edv * 100.0) if edv > 0 else 0.0
        if "RV_EDV" in volumes:
            features["rv_edv"] = float(volumes["RV_EDV"])

    # -- Clinical features ---------------------------------------------------
    if clinical_data:
        clinical_keys = [
            "age", "sex", "diabetes", "hypertension",
            "killip_class", "timi_flow", "troponin",
        ]
        for key in clinical_keys:
            if key in clinical_data:
                features[key] = float(clinical_data[key])

    return features


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------

def create_feature_matrix(
    patients_data: list[dict],
    feature_names: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Convert a list of per-patient result dictionaries to a feature matrix.

    Each element of *patients_data* must contain at least a ``strain_results``
    key (and optionally ``clinical_data``).

    Args:
        patients_data: List of dicts, each with ``strain_results`` and
            optionally ``clinical_data``.
        feature_names: If given, restrict columns to this ordered list.  Any
            missing feature is filled with ``NaN``.

    Returns:
        Tuple of ``(feature_df, feature_names)`` where *feature_df* has shape
        ``(N, D)`` and *feature_names* is the ordered list of column names.
    """
    rows: list[dict[str, float]] = []
    for patient in patients_data:
        strain_results = patient.get("strain_results", patient)
        clinical_data = patient.get("clinical_data")
        feats = extract_strain_features(strain_results, clinical_data)
        rows.append(feats)

    df = pd.DataFrame(rows)

    if feature_names is not None:
        # Ensure requested columns exist (fill with NaN if missing)
        for col in feature_names:
            if col not in df.columns:
                df[col] = np.nan
        df = df[feature_names]
    else:
        feature_names = sorted(df.columns.tolist())
        df = df[feature_names]

    return df, feature_names
