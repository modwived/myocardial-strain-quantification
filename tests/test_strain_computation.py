"""Tests for the mechanics modules: deformation, coordinate system, global/segmental strain, and strain rate.

Uses synthetic numpy arrays exclusively -- no real medical data required.
"""

from __future__ import annotations

import numpy as np
import pytest

from strain.mechanics.deformation import (
    check_jacobian,
    compute_deformation_gradient,
    compute_green_lagrange_strain,
    compute_jacobian,
    compute_principal_strains,
)
from strain.mechanics.coordinate_system import (
    _robust_centroid,
    compute_cardiac_axes,
    compute_longitudinal_axis,
    compute_rv_insertion_angle,
    get_lv_center,
    validate_orthogonality,
)
from strain.mechanics.global_strain import (
    compute_global_strain,
    compute_global_strain_timeseries,
    detect_peak_strain,
)
from strain.mechanics.segmental_strain import (
    AHA_SEGMENT_NAMES,
    assign_aha_segments,
    classify_slice_level,
    compute_segmental_strain,
)
from strain.mechanics.strain_rate import (
    compute_peak_diastolic_strain_rate,
    compute_peak_systolic_strain_rate,
    compute_strain_rate,
    compute_strain_rate_metrics,
)


# ---------------------------------------------------------------------------
# Deformation gradient tests
# ---------------------------------------------------------------------------


class TestComputeDeformationGradient:
    """Tests for compute_deformation_gradient."""

    def test_zero_displacement_is_identity(self):
        """F = I + grad(u); when u=0 everywhere, F should be the identity."""
        H, W = 32, 32
        displacement = np.zeros((2, H, W), dtype=np.float64)
        F = compute_deformation_gradient(displacement)
        assert F.shape == (2, 2, H, W)
        # F[0,0] and F[1,1] should be 1.0; F[0,1] and F[1,0] should be 0.0
        np.testing.assert_allclose(F[0, 0], 1.0, atol=1e-10)
        np.testing.assert_allclose(F[1, 1], 1.0, atol=1e-10)
        np.testing.assert_allclose(F[0, 1], 0.0, atol=1e-10)
        np.testing.assert_allclose(F[1, 0], 0.0, atol=1e-10)

    def test_uniform_translation(self):
        """Constant displacement (translation) should also give F = I."""
        H, W = 32, 32
        displacement = np.full((2, H, W), 5.0, dtype=np.float64)
        F = compute_deformation_gradient(displacement)
        np.testing.assert_allclose(F[0, 0], 1.0, atol=1e-10)
        np.testing.assert_allclose(F[1, 1], 1.0, atol=1e-10)
        np.testing.assert_allclose(F[0, 1], 0.0, atol=1e-10)
        np.testing.assert_allclose(F[1, 0], 0.0, atol=1e-10)

    def test_output_shape(self):
        """F should have shape (2, 2, H, W)."""
        displacement = np.zeros((2, 50, 60), dtype=np.float64)
        F = compute_deformation_gradient(displacement)
        assert F.shape == (2, 2, 50, 60)

    def test_with_smoothing(self):
        """Smoothing should not crash and should still produce correct shape."""
        displacement = np.random.randn(2, 32, 32).astype(np.float64) * 0.1
        F = compute_deformation_gradient(displacement, smooth_sigma=1.0)
        assert F.shape == (2, 2, 32, 32)

    def test_linear_displacement(self):
        """For u_x = a*x, grad_x(u_x) = a, so F[0,0] = 1+a in the interior."""
        H, W = 32, 32
        a = 0.1
        displacement = np.zeros((2, H, W), dtype=np.float64)
        # u_x varies linearly along axis=1 (x/column direction)
        for col in range(W):
            displacement[0, :, col] = a * col
        F = compute_deformation_gradient(displacement)
        # In the interior (away from boundaries), F[0,0] ~ 1 + a
        interior = F[0, 0, 1:-1, 1:-1]
        np.testing.assert_allclose(interior, 1.0 + a, atol=0.01)


class TestComputeGreenLagrangeStrain:
    """Tests for compute_green_lagrange_strain."""

    def test_identity_F_gives_zero_E(self):
        """E = 0.5*(F^T F - I); when F = I, E should be 0."""
        H, W = 32, 32
        F = np.zeros((2, 2, H, W), dtype=np.float64)
        F[0, 0] = 1.0
        F[1, 1] = 1.0
        E = compute_green_lagrange_strain(F)
        np.testing.assert_allclose(E, 0.0, atol=1e-10)

    def test_output_shape(self):
        """E should have shape (2, 2, H, W)."""
        F = np.zeros((2, 2, 40, 50), dtype=np.float64)
        F[0, 0] = 1.0
        F[1, 1] = 1.0
        E = compute_green_lagrange_strain(F)
        assert E.shape == (2, 2, 40, 50)

    def test_symmetry(self):
        """Green-Lagrange tensor should be symmetric: E[0,1] == E[1,0]."""
        F = np.random.randn(2, 2, 16, 16).astype(np.float64)
        F[0, 0] += 1.0
        F[1, 1] += 1.0
        E = compute_green_lagrange_strain(F)
        np.testing.assert_allclose(E[0, 1], E[1, 0], atol=1e-12)

    def test_known_stretching(self):
        """Pure x-stretch by factor s: F = [[s,0],[0,1]], E_xx = 0.5*(s^2 - 1)."""
        H, W = 16, 16
        s = 1.2
        F = np.zeros((2, 2, H, W), dtype=np.float64)
        F[0, 0] = s
        F[1, 1] = 1.0
        E = compute_green_lagrange_strain(F)
        expected_Exx = 0.5 * (s ** 2 - 1.0)
        np.testing.assert_allclose(E[0, 0], expected_Exx, atol=1e-10)
        np.testing.assert_allclose(E[1, 1], 0.0, atol=1e-10)


class TestComputeJacobian:
    """Tests for compute_jacobian and check_jacobian."""

    def test_identity_gives_one(self):
        """det(I) = 1."""
        H, W = 16, 16
        F = np.zeros((2, 2, H, W), dtype=np.float64)
        F[0, 0] = 1.0
        F[1, 1] = 1.0
        J = compute_jacobian(F)
        np.testing.assert_allclose(J, 1.0, atol=1e-10)

    def test_stretching_jacobian(self):
        """For F = [[2,0],[0,3]], det(F) = 6."""
        H, W = 8, 8
        F = np.zeros((2, 2, H, W), dtype=np.float64)
        F[0, 0] = 2.0
        F[1, 1] = 3.0
        J = compute_jacobian(F)
        np.testing.assert_allclose(J, 6.0, atol=1e-10)

    def test_check_jacobian_valid(self):
        """Identity F should pass the validity check."""
        H, W = 16, 16
        F = np.zeros((2, 2, H, W), dtype=np.float64)
        F[0, 0] = 1.0
        F[1, 1] = 1.0
        is_valid, J = check_jacobian(F)
        assert is_valid is True
        np.testing.assert_allclose(J, 1.0)

    def test_check_jacobian_invalid(self):
        """Negative Jacobian should fail the validity check."""
        H, W = 8, 8
        F = np.zeros((2, 2, H, W), dtype=np.float64)
        F[0, 0] = 1.0
        F[1, 1] = -1.0  # Makes det(F) = -1
        is_valid, J = check_jacobian(F)
        assert is_valid is False


class TestComputePrincipalStrains:
    """Tests for eigenvalue decomposition of the strain tensor."""

    def test_zero_strain(self):
        """Zero strain tensor should give zero principal strains."""
        H, W = 16, 16
        E = np.zeros((2, 2, H, W), dtype=np.float64)
        e1, e2, directions = compute_principal_strains(E)
        np.testing.assert_allclose(e1, 0.0, atol=1e-10)
        np.testing.assert_allclose(e2, 0.0, atol=1e-10)

    def test_ordering(self):
        """e1 should be >= e2 (descending order)."""
        E = np.random.randn(2, 2, 8, 8).astype(np.float64)
        # Make E symmetric
        E[0, 1] = E[1, 0] = (E[0, 1] + E[1, 0]) / 2.0
        e1, e2, _ = compute_principal_strains(E)
        assert np.all(e1 >= e2 - 1e-10), "e1 should be >= e2 everywhere"

    def test_output_shapes(self):
        """e1, e2 should be (H,W); directions should be (2,2,H,W)."""
        E = np.zeros((2, 2, 10, 12), dtype=np.float64)
        e1, e2, directions = compute_principal_strains(E)
        assert e1.shape == (10, 12)
        assert e2.shape == (10, 12)
        assert directions.shape == (2, 2, 10, 12)


# ---------------------------------------------------------------------------
# Cardiac coordinate system tests
# ---------------------------------------------------------------------------


class TestComputeCardiacAxes:
    """Tests for compute_cardiac_axes."""

    def _make_ring_mask(self, H=64, W=64, inner_r=10, outer_r=20, center=(32, 32)):
        """Create a synthetic annular myocardial mask."""
        mask = np.zeros((H, W), dtype=np.float64)
        y, x = np.ogrid[:H, :W]
        dist = np.sqrt((y - center[0]) ** 2 + (x - center[1]) ** 2)
        mask[(dist >= inner_r) & (dist <= outer_r)] = 1.0
        return mask

    def test_orthogonality(self):
        """e_circ and e_rad should be orthogonal (dot product ~ 0) at every pixel."""
        mask = self._make_ring_mask()
        e_circ, e_rad = compute_cardiac_axes(mask)
        # Dot product at each pixel
        dot = e_circ[0] * e_rad[0] + e_circ[1] * e_rad[1]
        myo_pixels = mask > 0
        assert np.all(np.abs(dot[myo_pixels]) < 1e-6), (
            f"Max dot product = {np.abs(dot[myo_pixels]).max()}, should be ~0"
        )

    def test_unit_vectors(self):
        """e_circ and e_rad should have unit norm at myocardial pixels."""
        mask = self._make_ring_mask()
        e_circ, e_rad = compute_cardiac_axes(mask)
        myo_pixels = mask > 0

        circ_norm = np.sqrt(e_circ[0] ** 2 + e_circ[1] ** 2)
        rad_norm = np.sqrt(e_rad[0] ** 2 + e_rad[1] ** 2)

        np.testing.assert_allclose(circ_norm[myo_pixels], 1.0, atol=1e-6)
        np.testing.assert_allclose(rad_norm[myo_pixels], 1.0, atol=1e-6)

    def test_with_lv_cavity_mask(self):
        """Should use LV cavity centroid when provided."""
        mask = self._make_ring_mask(center=(32, 32))
        lv_cavity = np.zeros((64, 64), dtype=np.float64)
        lv_cavity[28:36, 28:36] = 1.0  # centered at ~32,32
        e_circ, e_rad = compute_cardiac_axes(mask, lv_cavity_mask=lv_cavity)
        assert e_circ.shape == (2, 64, 64)
        assert e_rad.shape == (2, 64, 64)

    def test_empty_mask_raises(self):
        """Empty mask should raise ValueError (cannot compute centroid)."""
        mask = np.zeros((64, 64), dtype=np.float64)
        with pytest.raises(ValueError, match="empty mask"):
            compute_cardiac_axes(mask)

    def test_validate_orthogonality_helper(self):
        """validate_orthogonality should return True for properly computed axes."""
        mask = self._make_ring_mask()
        e_circ, e_rad = compute_cardiac_axes(mask)
        assert validate_orthogonality(e_circ, e_rad, mask) is True


class TestComputeLongitudinalAxis:
    """Tests for compute_longitudinal_axis."""

    def test_direction(self):
        """Should return a unit vector from base to apex."""
        basal = (10.0, 32.0)
        apical = (50.0, 32.0)
        axis = compute_longitudinal_axis(basal, apical)
        assert axis.shape == (2,)
        norm = np.linalg.norm(axis)
        assert abs(norm - 1.0) < 1e-8, "Should be a unit vector"

    def test_degenerate_case(self):
        """Same base and apex should return a default direction."""
        axis = compute_longitudinal_axis((32, 32), (32, 32))
        assert axis.shape == (2,)
        norm = np.linalg.norm(axis)
        assert abs(norm - 1.0) < 1e-8


class TestRvInsertionAngle:
    """Tests for compute_rv_insertion_angle."""

    def test_known_angle(self):
        """RV insertion directly to the right of LV center should be 0 degrees."""
        lv_center = (50.0, 50.0)
        rv_pt = (50.0, 70.0)  # directly to the right
        angle = compute_rv_insertion_angle(lv_center, rv_pt)
        assert abs(angle) < 1e-5 or abs(angle - 360) < 1e-5

    def test_above_center(self):
        """RV insertion directly above should be ~270 degrees (or -90 -> 270)."""
        lv_center = (50.0, 50.0)
        rv_pt = (30.0, 50.0)  # directly above
        angle = compute_rv_insertion_angle(lv_center, rv_pt)
        # atan2(-20, 0) = -90 deg -> 270 deg
        assert abs(angle - 270.0) < 1.0


class TestRobustCentroid:
    """Tests for _robust_centroid."""

    def test_single_component(self):
        """Should return centroid of the single component."""
        mask = np.zeros((64, 64), dtype=np.float64)
        mask[20:40, 20:40] = 1.0
        cy, cx = _robust_centroid(mask)
        assert abs(cy - 29.5) < 1.0
        assert abs(cx - 29.5) < 1.0

    def test_empty_mask_raises(self):
        """Empty mask should raise ValueError."""
        mask = np.zeros((32, 32), dtype=np.float64)
        with pytest.raises(ValueError, match="empty mask"):
            _robust_centroid(mask)


class TestGetLvCenter:
    """Tests for get_lv_center."""

    def test_uses_cavity_when_available(self):
        """Should use LV cavity mask centroid when provided."""
        myo_mask = np.zeros((64, 64), dtype=np.float64)
        myo_mask[10:50, 10:50] = 1.0
        lv_cavity = np.zeros((64, 64), dtype=np.float64)
        lv_cavity[25:35, 25:35] = 1.0

        cy, cx = get_lv_center(myo_mask, lv_cavity)
        assert abs(cy - 29.5) < 1.0
        assert abs(cx - 29.5) < 1.0


# ---------------------------------------------------------------------------
# Global strain tests
# ---------------------------------------------------------------------------


class TestComputeGlobalStrain:
    """Tests for compute_global_strain."""

    def test_zero_displacement_zero_strain(self):
        """Zero displacement should yield zero GCS and GRS."""
        H, W = 64, 64
        displacement = np.zeros((2, H, W), dtype=np.float64)
        # Create a ring-shaped myocardial mask
        mask = np.zeros((H, W), dtype=np.float64)
        y, x = np.ogrid[:H, :W]
        dist = np.sqrt((y - 32) ** 2 + (x - 32) ** 2)
        mask[(dist >= 10) & (dist <= 20)] = 1.0

        result = compute_global_strain(displacement, mask)
        assert "GCS" in result
        assert "GRS" in result
        assert abs(result["GCS"]) < 1e-6
        assert abs(result["GRS"]) < 1e-6

    def test_empty_mask_raises(self):
        """Empty myocardial mask should raise ValueError from centroid computation."""
        displacement = np.zeros((2, 32, 32), dtype=np.float64)
        mask = np.zeros((32, 32), dtype=np.float64)
        with pytest.raises(ValueError, match="empty mask"):
            compute_global_strain(displacement, mask)

    def test_returns_dict_with_expected_keys(self):
        """Should return a dict with GCS and GRS keys."""
        displacement = np.zeros((2, 32, 32), dtype=np.float64)
        mask = np.ones((32, 32), dtype=np.float64)
        result = compute_global_strain(displacement, mask)
        assert isinstance(result, dict)
        assert "GCS" in result
        assert "GRS" in result


class TestComputeGlobalStrainTimeseries:
    """Tests for compute_global_strain_timeseries."""

    def test_output_keys(self):
        """Should return GCS, GRS, and time arrays."""
        T, H, W = 5, 32, 32
        displacements = np.zeros((T, 2, H, W), dtype=np.float64)
        masks = np.ones((T, H, W), dtype=np.float64)
        result = compute_global_strain_timeseries(displacements, masks)
        assert "GCS" in result
        assert "GRS" in result
        assert "time" in result
        assert len(result["GCS"]) == T
        assert len(result["GRS"]) == T
        assert len(result["time"]) == T

    def test_default_time_points(self):
        """Without explicit time_points, should use 0,1,2,...,T-1."""
        T = 5
        displacements = np.zeros((T, 2, 32, 32), dtype=np.float64)
        masks = np.ones((T, 32, 32), dtype=np.float64)
        result = compute_global_strain_timeseries(displacements, masks)
        np.testing.assert_array_equal(result["time"], np.arange(T, dtype=np.float64))


class TestDetectPeakStrain:
    """Tests for detect_peak_strain."""

    def test_circumferential_peak(self):
        """For circumferential strain, peak is the most negative value."""
        curve = np.array([0.0, -5.0, -20.0, -15.0, -3.0])
        idx, val = detect_peak_strain(curve, strain_type="circumferential")
        assert idx == 2
        assert val == -20.0

    def test_radial_peak(self):
        """For radial strain, peak is the most positive value."""
        curve = np.array([0.0, 10.0, 40.0, 30.0, 5.0])
        idx, val = detect_peak_strain(curve, strain_type="radial")
        assert idx == 2
        assert val == 40.0

    def test_empty_curve(self):
        """Empty curve should return (0, 0.0)."""
        idx, val = detect_peak_strain(np.array([]), strain_type="circumferential")
        assert idx == 0
        assert val == 0.0

    def test_invalid_type_raises(self):
        """Unknown strain type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown strain_type"):
            detect_peak_strain(np.array([1.0, 2.0]), strain_type="invalid")


# ---------------------------------------------------------------------------
# AHA segmental strain tests
# ---------------------------------------------------------------------------


class TestAssignAhaSegments:
    """Tests for assign_aha_segments."""

    def _make_ring_mask(self, H=64, W=64, inner_r=10, outer_r=20, center=(32, 32)):
        """Create a synthetic annular myocardial mask."""
        mask = np.zeros((H, W), dtype=np.float64)
        y, x = np.ogrid[:H, :W]
        dist = np.sqrt((y - center[0]) ** 2 + (x - center[1]) ** 2)
        mask[(dist >= inner_r) & (dist <= outer_r)] = 1.0
        return mask

    def test_full_coverage(self):
        """Every myocardial pixel should be assigned to a segment (no zero-label)."""
        mask = self._make_ring_mask()
        segments = assign_aha_segments(mask, slice_level="mid")
        myo_pixels = mask > 0
        # All myocardial pixels should have a non-zero segment label
        assert np.all(segments[myo_pixels] > 0), "All myocardial pixels should be assigned"

    def test_mid_has_6_segments(self):
        """Mid slice should produce segments 7--12 (6 segments)."""
        mask = self._make_ring_mask()
        segments = assign_aha_segments(mask, slice_level="mid")
        unique_labels = set(np.unique(segments)) - {0}
        assert all(7 <= s <= 12 for s in unique_labels), f"Mid segments should be 7-12, got {unique_labels}"

    def test_basal_has_6_segments(self):
        """Basal slice should produce segments 1--6."""
        mask = self._make_ring_mask()
        segments = assign_aha_segments(mask, slice_level="basal")
        unique_labels = set(np.unique(segments)) - {0}
        assert all(1 <= s <= 6 for s in unique_labels), f"Basal segments should be 1-6, got {unique_labels}"

    def test_apical_has_4_segments(self):
        """Apical slice should produce segments 13--16 (4 segments)."""
        mask = self._make_ring_mask()
        segments = assign_aha_segments(mask, slice_level="apical")
        unique_labels = set(np.unique(segments)) - {0}
        assert all(13 <= s <= 16 for s in unique_labels), f"Apical segments should be 13-16, got {unique_labels}"

    def test_empty_mask(self):
        """Empty mask should return all zeros."""
        mask = np.zeros((64, 64), dtype=np.float64)
        segments = assign_aha_segments(mask, slice_level="mid")
        assert np.all(segments == 0)

    def test_invalid_slice_level(self):
        """Unknown slice level should raise ValueError."""
        mask = self._make_ring_mask()
        with pytest.raises(ValueError, match="Unknown slice level"):
            assign_aha_segments(mask, slice_level="invalid")

    def test_background_is_zero(self):
        """Non-myocardial pixels should have segment label 0."""
        mask = self._make_ring_mask()
        segments = assign_aha_segments(mask, slice_level="mid")
        bg_pixels = mask == 0
        assert np.all(segments[bg_pixels] == 0)


class TestClassifySliceLevel:
    """Tests for classify_slice_level."""

    def test_basal_slice(self):
        """First third of slices should be basal."""
        assert classify_slice_level(0, 9) == "basal"
        assert classify_slice_level(2, 9) == "basal"

    def test_mid_slice(self):
        """Middle third should be mid."""
        assert classify_slice_level(3, 9) == "mid"
        assert classify_slice_level(5, 9) == "mid"

    def test_apical_slice(self):
        """Last third should be apical."""
        assert classify_slice_level(6, 9) == "apical"
        assert classify_slice_level(8, 9) == "apical"

    def test_invalid_total_slices(self):
        """total_slices <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="total_slices must be > 0"):
            classify_slice_level(0, 0)


class TestComputeSegmentalStrain:
    """Tests for compute_segmental_strain."""

    def test_zero_displacement(self):
        """Zero displacement should give zero strain for all segments."""
        H, W = 64, 64
        displacement = np.zeros((2, H, W), dtype=np.float64)
        mask = np.zeros((H, W), dtype=np.float64)
        y, x = np.ogrid[:H, :W]
        dist = np.sqrt((y - 32) ** 2 + (x - 32) ** 2)
        mask[(dist >= 10) & (dist <= 20)] = 1.0
        segments = assign_aha_segments(mask, slice_level="mid")

        result = compute_segmental_strain(displacement, mask, segments)
        for seg_id, vals in result.items():
            assert abs(vals["GCS"]) < 1e-4, f"Segment {seg_id} GCS should be ~0"
            assert abs(vals["GRS"]) < 1e-4, f"Segment {seg_id} GRS should be ~0"


class TestAhaSegmentNames:
    """Tests for the AHA_SEGMENT_NAMES constant."""

    def test_has_16_segments(self):
        """Should have 16 named segments."""
        assert len(AHA_SEGMENT_NAMES) == 16

    def test_keys_1_to_16(self):
        """Keys should be integers 1 through 16."""
        assert set(AHA_SEGMENT_NAMES.keys()) == set(range(1, 17))


# ---------------------------------------------------------------------------
# Strain rate tests
# ---------------------------------------------------------------------------


class TestComputeStrainRate:
    """Tests for compute_strain_rate."""

    def test_constant_strain_zero_rate(self):
        """Constant strain over time should give zero strain rate."""
        strain = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        time = np.array([0.0, 50.0, 100.0, 150.0, 200.0])
        sr = compute_strain_rate(strain, time)
        np.testing.assert_allclose(sr, 0.0, atol=1e-10)

    def test_linear_strain_constant_rate(self):
        """Linear strain: strain = a*t should give constant rate.

        strain = 2.0 * t (% per ms)
        d(strain)/dt = 2.0 %/ms
        SR = 2.0 %/ms * 10 = 20.0 (1/s)
        """
        time = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        strain = 2.0 * time  # 0, 20, 40, 60, 80  (% units)
        sr = compute_strain_rate(strain, time)
        # Interior points use central differences: (strain[i+1] - strain[i-1]) / (t[i+1] - t[i-1])
        # = (2.0 * (t[i+1] - t[i-1])) / (t[i+1] - t[i-1]) = 2.0  (%/ms) * 10 = 20.0 (1/s)
        np.testing.assert_allclose(sr, 20.0, atol=1e-6)

    def test_output_length(self):
        """Output should have the same length as input."""
        strain = np.array([0.0, -5.0, -15.0, -10.0, -2.0])
        time = np.array([0.0, 30.0, 60.0, 90.0, 120.0])
        sr = compute_strain_rate(strain, time)
        assert len(sr) == len(strain)

    def test_single_point(self):
        """Single time point should return zero."""
        sr = compute_strain_rate(np.array([5.0]), np.array([100.0]))
        assert len(sr) == 1
        assert sr[0] == 0.0


class TestPeakStrainRates:
    """Tests for peak systolic and diastolic strain rate detection."""

    def test_peak_systolic(self):
        """Peak systolic SR is the most negative value."""
        sr = np.array([0.0, -1.0, -2.5, -1.5, 0.5, 1.0])
        peak = compute_peak_systolic_strain_rate(sr)
        assert peak == -2.5

    def test_peak_diastolic(self):
        """Peak diastolic SR is the most positive value."""
        sr = np.array([0.0, -1.0, -2.5, -1.5, 0.5, 2.0])
        peak = compute_peak_diastolic_strain_rate(sr)
        assert peak == 2.0

    def test_empty_array(self):
        """Empty array should return 0.0."""
        assert compute_peak_systolic_strain_rate(np.array([])) == 0.0
        assert compute_peak_diastolic_strain_rate(np.array([])) == 0.0


class TestComputeStrainRateMetrics:
    """Tests for the convenience function compute_strain_rate_metrics."""

    def test_returns_expected_keys(self):
        """Should return strain_rate, peak_systolic_sr, peak_diastolic_sr."""
        strain = np.array([0.0, -5.0, -15.0, -10.0, -2.0])
        time = np.array([0.0, 30.0, 60.0, 90.0, 120.0])
        result = compute_strain_rate_metrics(strain, time)
        assert "strain_rate" in result
        assert "peak_systolic_sr" in result
        assert "peak_diastolic_sr" in result

    def test_strain_rate_shape(self):
        """strain_rate array should have the same length as input."""
        strain = np.array([0.0, -10.0, -20.0, -15.0, -5.0])
        time = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
        result = compute_strain_rate_metrics(strain, time)
        assert len(result["strain_rate"]) == len(strain)
