"""Compute strain rate as the temporal derivative of strain.

Strain rate (SR) is the rate of myocardial deformation and is an important
clinical parameter:

* **Peak systolic SR** -- the most negative rate during systole for
  circumferential/longitudinal strain, indicating the speed of contraction.
* **Peak early-diastolic SR** -- the most positive rate during early
  diastole, indicating the speed of relaxation.

Normal reference values (circumferential):
  Peak systolic SR:   -1.1 +/- 0.3  (1/s)
  Peak diastolic SR:  +1.5 +/- 0.5  (1/s)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_strain_rate(
    strain_timeseries: np.ndarray,
    time_points: np.ndarray,
) -> np.ndarray:
    """Compute strain rate as the temporal derivative of strain.

    Central differences are used for interior time points and one-sided
    (forward/backward) differences at the first and last time points.

    Args:
        strain_timeseries: (T,) strain values over time (in percent).
        time_points: (T,) time values in **milliseconds**.

    Returns:
        strain_rate: (T,) strain rate in **1/s** (percent per second, then
        divided by 100 to yield fractional strain rate).
    """
    strain = np.asarray(strain_timeseries, dtype=np.float64)
    t = np.asarray(time_points, dtype=np.float64)
    T = len(strain)

    if T < 2:
        return np.zeros_like(strain)

    sr = np.empty(T, dtype=np.float64)

    # Forward difference at t=0
    dt0 = t[1] - t[0]
    if abs(dt0) < 1e-12:
        sr[0] = 0.0
    else:
        sr[0] = (strain[1] - strain[0]) / dt0

    # Central differences for interior points
    for i in range(1, T - 1):
        dt = t[i + 1] - t[i - 1]
        if abs(dt) < 1e-12:
            sr[i] = 0.0
        else:
            sr[i] = (strain[i + 1] - strain[i - 1]) / dt

    # Backward difference at t=T-1
    dt_end = t[T - 1] - t[T - 2]
    if abs(dt_end) < 1e-12:
        sr[T - 1] = 0.0
    else:
        sr[T - 1] = (strain[T - 1] - strain[T - 2]) / dt_end

    # strain is in percent; time in ms.  SR = (d strain%) / (dt ms)
    # Convert to fractional strain rate per second:
    #   (percent / ms) -> (1/s):  multiply by 1000 / 100 = 10
    sr *= 10.0

    return sr


# ---------------------------------------------------------------------------
# Peak detection helpers
# ---------------------------------------------------------------------------

def compute_peak_systolic_strain_rate(
    strain_rate: np.ndarray,
    time_points: np.ndarray | None = None,
) -> float:
    """Find peak systolic strain rate (most negative value).

    For circumferential and longitudinal strain, systolic contraction is
    characterised by the most negative strain rate.

    Args:
        strain_rate: (T,) strain rate in 1/s.
        time_points: (T,) time in ms (unused but kept for API symmetry).

    Returns:
        Peak systolic strain rate (negative value, 1/s).
    """
    if len(strain_rate) == 0:
        return 0.0
    return float(np.min(strain_rate))


def compute_peak_diastolic_strain_rate(
    strain_rate: np.ndarray,
    time_points: np.ndarray | None = None,
) -> float:
    """Find peak early-diastolic strain rate (most positive value).

    During early diastole the myocardium relaxes, producing a positive
    strain rate for circumferential/longitudinal directions.

    Args:
        strain_rate: (T,) strain rate in 1/s.
        time_points: (T,) time in ms (unused but kept for API symmetry).

    Returns:
        Peak diastolic strain rate (positive value, 1/s).
    """
    if len(strain_rate) == 0:
        return 0.0
    return float(np.max(strain_rate))


def compute_strain_rate_metrics(
    strain_timeseries: np.ndarray,
    time_points: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """Convenience function: compute strain rate and peak values in one call.

    Args:
        strain_timeseries: (T,) strain values (%) over time.
        time_points: (T,) time in milliseconds.

    Returns:
        Dictionary with:
          - ``'strain_rate'``: (T,) strain rate array (1/s).
          - ``'peak_systolic_sr'``: peak systolic strain rate (1/s).
          - ``'peak_diastolic_sr'``: peak diastolic strain rate (1/s).
    """
    sr = compute_strain_rate(strain_timeseries, time_points)
    return {
        "strain_rate": sr,
        "peak_systolic_sr": compute_peak_systolic_strain_rate(sr, time_points),
        "peak_diastolic_sr": compute_peak_diastolic_strain_rate(sr, time_points),
    }
