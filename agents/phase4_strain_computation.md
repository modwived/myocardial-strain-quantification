# Agent: Phase 4 — Strain Computation

## Mission
Implement the biomechanical engine that converts displacement fields into clinically meaningful myocardial strain values (GLS, GCS, GRS), both globally and per AHA segment. This is the core scientific contribution of the pipeline.

## Status: IN PROGRESS

## Files You Own
- `strain/mechanics/deformation.py` — deformation gradient tensor (skeleton exists)
- `strain/mechanics/coordinate_system.py` — cardiac coordinate axes (skeleton exists)
- `strain/mechanics/global_strain.py` — GCS, GRS computation (skeleton exists)
- `strain/mechanics/segmental_strain.py` — AHA 16-segment model (skeleton exists)
- `strain/mechanics/strain_rate.py` — temporal derivative of strain (CREATE)
- `strain/mechanics/longitudinal_strain.py` — GLS from long-axis views (CREATE)
- `strain/mechanics/visualization.py` — bull's-eye plots, strain curves (CREATE)
- `tests/test_strain_computation.py` — unit tests (CREATE)

## Detailed Requirements

### 1. deformation.py — Verify and improve
- [x] `compute_deformation_gradient(displacement)` — F = I + ∇u
- [x] `compute_green_lagrange_strain(F)` — E = 0.5(F^T F - I)
- [ ] Add Jacobian determinant check: det(F) > 0 everywhere (physically valid)
- [ ] Add smoothing option: Gaussian-smooth displacement before gradient (reduces noise)
- [ ] Add `compute_principal_strains(E)` — eigenvalue decomposition for principal strain directions
- [ ] Handle edge pixels: use one-sided differences at boundaries instead of np.gradient

### 2. coordinate_system.py — Verify and improve
- [x] `compute_cardiac_axes(myocardial_mask)` — e_circ, e_rad from LV geometry
- [ ] Improve center detection: use LV cavity mask (label=3) centroid, not eroded myocardium
- [ ] Add `compute_longitudinal_axis(basal_mask, apical_mask)` — for long-axis strain
- [ ] Add option to input RV insertion points for standardized angular reference (AHA convention)
- [ ] Validate: e_circ ⊥ e_rad everywhere (dot product ≈ 0)
- [ ] Handle cases where mask has disconnected components

### 3. global_strain.py — Verify and improve
- [x] `compute_global_strain(displacement, myocardial_mask)` — returns GCS, GRS
- [ ] Add temporal strain curves: compute strain at each time point across the cardiac cycle
- [ ] Add peak strain detection: find the time of maximum strain (typically end-systole)
- [ ] Return strain-time curves as arrays, not just peak values
- [ ] Add `compute_global_strain_timeseries(displacements, myocardial_masks)`:
  ```python
  def compute_global_strain_timeseries(
      displacements: np.ndarray,    # (T, 2, H, W) cumulative from ED
      myocardial_masks: np.ndarray, # (T, H, W) myocardial mask at each phase
  ) -> dict[str, np.ndarray]:
      """Returns {'GCS': array(T,), 'GRS': array(T,), 'time': array(T,)}"""
  ```

### 4. segmental_strain.py — Verify and improve
- [x] `AHA_SEGMENT_NAMES` — 16-segment labels
- [x] `assign_aha_segments(myocardial_mask, slice_level)` — angular segmentation
- [ ] Fix angular reference: AHA standard starts at RV anterior insertion point, not at 0°
- [ ] Add `compute_segmental_strain(displacement, myocardial_mask, segments)`:
  ```python
  def compute_segmental_strain(
      displacement: np.ndarray,
      myocardial_mask: np.ndarray,
      segments: np.ndarray,
  ) -> dict[int, dict[str, float]]:
      """Returns {1: {'GCS': -22.1, 'GRS': 35.2}, 2: {...}, ...}"""
  ```
- [ ] Map short-axis slices to basal/mid/apical based on slice position

### 5. strain_rate.py — CREATE
```python
def compute_strain_rate(
    strain_timeseries: np.ndarray,
    time_points: np.ndarray,
) -> np.ndarray:
    """Compute strain rate as temporal derivative of strain.

    Args:
        strain_timeseries: (T,) strain values over time
        time_points: (T,) time in milliseconds

    Returns:
        strain_rate: (T,) strain rate in 1/s
    """
    # Use central differences for interior, one-sided at boundaries
    # Convert from per-ms to per-second

def compute_peak_systolic_strain_rate(strain_rate, time_points) -> float:
    """Find peak systolic strain rate (most negative for circumferential/longitudinal)."""

def compute_peak_diastolic_strain_rate(strain_rate, time_points) -> float:
    """Find peak early diastolic strain rate (most positive)."""
```

### 6. longitudinal_strain.py — CREATE
```python
def compute_gls_from_long_axis(
    displacements_2ch: np.ndarray | None,  # 2-chamber long axis
    displacements_4ch: np.ndarray | None,  # 4-chamber long axis
    myocardial_mask_2ch: np.ndarray | None,
    myocardial_mask_4ch: np.ndarray | None,
) -> dict[str, float]:
    """Compute GLS from long-axis views.

    GLS measures base-to-apex shortening (negative in normal hearts).
    If both views available, average for global GLS.

    Returns:
        {'GLS': float, 'GLS_2ch': float, 'GLS_4ch': float}
    """
```
- Longitudinal direction is along the LV long axis (base to apex)
- Project strain tensor onto longitudinal direction
- If only short-axis data available, estimate from through-plane motion (less accurate)

### 7. visualization.py — CREATE
```python
def plot_bulls_eye(segmental_values: dict[int, float], title: str = "") -> Figure:
    """Create AHA 16-segment bull's-eye plot.

    Standard cardiac visualization: concentric rings for basal/mid/apical,
    color-coded by strain value.
    """

def plot_strain_curves(
    strain_timeseries: dict[str, np.ndarray],
    time_points: np.ndarray,
    title: str = "",
) -> Figure:
    """Plot strain-time curves for GCS, GRS, GLS."""

def plot_displacement_field(
    image: np.ndarray,
    displacement: np.ndarray,
    step: int = 4,
) -> Figure:
    """Overlay displacement vectors (quiver plot) on cardiac image."""
```

### 8. tests/test_strain_computation.py
- Test deformation gradient with known displacement (e.g., uniform stretch)
- Test Green-Lagrange with identity deformation → E = 0
- Test with uniform radial expansion → positive GRS, negative GCS
- Test AHA segments cover full myocardium (no gaps, no overlap)
- Test segmental strain: uniform deformation → all segments equal
- Test strain rate: linear strain → constant rate
- Test coordinate system: e_circ ⊥ e_rad
- Test Jacobian: identity → det(F) = 1 everywhere

## Normal Reference Values (for validation)
| Parameter | Normal Range | Units |
|-----------|-------------|-------|
| GLS | -20.1 ± 3.2 | % |
| GCS | -23.0 ± 4.8 | % |
| GRS | +34.1 ± 12.6 | % |
| Peak systolic SR (circ) | -1.1 ± 0.3 | 1/s |
| Peak diastolic SR (circ) | +1.5 ± 0.5 | 1/s |

## Interface Contract
```python
# Input from Phase 3:
displacements: np.ndarray   # (T, 2, H, W) cumulative displacement from ED
# Input from Phase 2:
myocardial_masks: np.ndarray # (T, H, W) with label 2 = myocardium
lv_cavity_masks: np.ndarray  # (T, H, W) with label 3 = LV cavity

# Output to Phase 5 (Risk):
strain_results = {
    "global": {"GLS": -18.2, "GCS": -21.5, "GRS": 30.1},
    "segmental": {1: {"GCS": -22.1, "GRS": 35.2}, ...},
    "timeseries": {"GCS": np.array(...), "GRS": np.array(...), "time": np.array(...)},
    "strain_rate": {"peak_systolic_sr": -1.05, "peak_diastolic_sr": 1.42},
}

# Output to Phase 6 (API):
# Same dict serialized as JSON
```

## If You Get Stuck
- Strain convention: negative = shortening (GCS, GLS), positive = thickening (GRS)
- If strain values are unreasonable, check displacement field units (should be in pixels)
- np.gradient uses central differences by default — appropriate for interior points
- For AHA segments, the anterior RV insertion point is typically at ~45° in standard short-axis orientation
- Bull's-eye plot: use matplotlib polar projection with custom radius mapping
- Green-Lagrange vs Engineering strain: E_GL = ε + ε²/2 (they differ for large strains)
- For noisy strain maps, apply Gaussian smoothing to displacement field BEFORE computing gradient
