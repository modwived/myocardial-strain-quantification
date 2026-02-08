"""Tests for the data pipeline: preprocessing, augmentation, and loader modules.

Uses synthetic numpy arrays exclusively -- no real medical imaging data required.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

from strain.data.preprocessing import (
    build_preprocessing_pipeline,
    center_crop,
    crop_around_lv,
    detect_lv_center,
    normalize_intensity,
    resample_image,
)


class TestNormalizeIntensity:
    """Tests for normalize_intensity."""

    def test_output_zero_mean_unit_variance(self):
        """Normalized image should have approximately zero mean and unit std."""
        rng = np.random.RandomState(42)
        image = rng.rand(64, 64).astype(np.float32) * 200 + 50
        result = normalize_intensity(image)

        assert abs(result.mean()) < 1e-5, f"Mean should be ~0, got {result.mean()}"
        assert abs(result.std() - 1.0) < 1e-5, f"Std should be ~1, got {result.std()}"

    def test_preserves_shape(self):
        """Output shape should match input shape."""
        image = np.random.rand(3, 64, 64).astype(np.float32)
        result = normalize_intensity(image)
        assert result.shape == image.shape

    def test_constant_image(self):
        """Constant image (std=0) should not produce NaN/Inf."""
        image = np.ones((32, 32), dtype=np.float32) * 5.0
        result = normalize_intensity(image)
        assert not np.isnan(result).any(), "Constant image should not produce NaN"
        assert not np.isinf(result).any(), "Constant image should not produce Inf"
        # With zero std, result should be image - mean = 0
        np.testing.assert_allclose(result, 0.0, atol=1e-7)

    def test_works_4d(self):
        """Should handle 4-D arrays (T, D, H, W)."""
        image = np.random.rand(5, 3, 64, 64).astype(np.float32)
        result = normalize_intensity(image)
        assert result.shape == (5, 3, 64, 64)
        assert abs(result.mean()) < 1e-5


class TestCenterCrop:
    """Tests for center_crop."""

    def test_output_shape_matches_crop_size(self):
        """Cropped image should have shape (..., crop_size, crop_size)."""
        image = np.random.rand(200, 200).astype(np.float32)
        result = center_crop(image, crop_size=128)
        assert result.shape == (128, 128)

    def test_output_shape_3d(self):
        """Should work with leading dimensions: (D, H, W) -> (D, crop, crop)."""
        image = np.random.rand(5, 200, 200).astype(np.float32)
        result = center_crop(image, crop_size=64)
        assert result.shape == (5, 64, 64)

    def test_output_shape_4d(self):
        """Should work with (T, D, H, W)."""
        image = np.random.rand(10, 3, 256, 256).astype(np.float32)
        result = center_crop(image, crop_size=128)
        assert result.shape == (10, 3, 128, 128)

    def test_crop_larger_than_image(self):
        """When crop_size > image size, the output should be at most image size."""
        image = np.random.rand(32, 32).astype(np.float32)
        result = center_crop(image, crop_size=64)
        # start = max(0, (32-64)//2) = 0, so result is image[0:64, 0:64]
        # which is image[0:32, 0:32] since numpy clips
        assert result.shape[0] <= 64 and result.shape[1] <= 64

    def test_default_crop_size_128(self):
        """Default crop_size should be 128."""
        image = np.random.rand(256, 256).astype(np.float32)
        result = center_crop(image)
        assert result.shape == (128, 128)


class TestResampleImage:
    """Tests for resample_image."""

    def test_changes_spatial_dimensions(self):
        """Resampling to different spacing should change spatial dimensions."""
        image = np.random.rand(100, 100).astype(np.float32)
        # Current spacing 2mm, target 1mm -> image should double in size
        result = resample_image(image, current_spacing=(2.0, 2.0), target_spacing=(1.0, 1.0))
        assert result.shape[0] > image.shape[0], "Resampled H should be larger"
        assert result.shape[1] > image.shape[1], "Resampled W should be larger"

    def test_same_spacing_preserves_size(self):
        """Resampling with same current and target spacing should keep dimensions."""
        image = np.random.rand(64, 64).astype(np.float32)
        result = resample_image(image, current_spacing=(1.25, 1.25), target_spacing=(1.25, 1.25))
        assert result.shape == (64, 64)

    def test_preserves_leading_dimensions(self):
        """Leading dimensions (e.g. T, D) should be preserved."""
        image = np.random.rand(5, 3, 64, 64).astype(np.float32)
        result = resample_image(image, current_spacing=(1.25, 1.25), target_spacing=(1.25, 1.25))
        assert result.shape[0] == 5
        assert result.shape[1] == 3

    def test_downsampling(self):
        """Larger target spacing should reduce spatial dimensions."""
        image = np.random.rand(128, 128).astype(np.float32)
        result = resample_image(image, current_spacing=(1.0, 1.0), target_spacing=(2.0, 2.0))
        assert result.shape[0] < image.shape[0]
        assert result.shape[1] < image.shape[1]


class TestDetectLvCenter:
    """Tests for detect_lv_center."""

    def test_returns_center_for_bright_spot(self):
        """Should detect center of a bright region."""
        image = np.zeros((128, 128), dtype=np.float32)
        # Place a bright circle near center
        y, x = np.ogrid[:128, :128]
        mask = ((y - 64) ** 2 + (x - 64) ** 2) < 15 ** 2
        image[mask] = 1.0
        cy, cx = detect_lv_center(image)
        assert abs(cy - 64) < 5, f"Row center should be ~64, got {cy}"
        assert abs(cx - 64) < 5, f"Col center should be ~64, got {cx}"

    def test_3d_input(self):
        """Should work with 3-D input (D, H, W)."""
        image = np.zeros((5, 128, 128), dtype=np.float32)
        y, x = np.ogrid[:128, :128]
        mask = ((y - 60) ** 2 + (x - 70) ** 2) < 10 ** 2
        image[:, mask] = 1.0
        cy, cx = detect_lv_center(image)
        assert isinstance(cy, int)
        assert isinstance(cx, int)

    def test_4d_input(self):
        """Should work with 4-D input (T, D, H, W)."""
        image = np.zeros((3, 5, 128, 128), dtype=np.float32)
        y, x = np.ogrid[:128, :128]
        mask = ((y - 64) ** 2 + (x - 64) ** 2) < 12 ** 2
        image[:, :, mask] = 1.0
        cy, cx = detect_lv_center(image)
        assert isinstance(cy, int) and isinstance(cx, int)


class TestCropAroundLv:
    """Tests for crop_around_lv."""

    def test_output_shape(self):
        """Should produce (crop_size, crop_size) spatial dims."""
        image = np.random.rand(256, 256).astype(np.float32)
        result = crop_around_lv(image, center=(128, 128), crop_size=64)
        assert result.shape == (64, 64)

    def test_boundary_clamping(self):
        """Center near the edge should still produce a valid crop."""
        image = np.random.rand(128, 128).astype(np.float32)
        # Center is at (5, 5), which is near the top-left corner
        result = crop_around_lv(image, center=(5, 5), crop_size=64)
        assert result.shape == (64, 64)


class TestBuildPreprocessingPipeline:
    """Tests for build_preprocessing_pipeline."""

    def test_empty_config_returns_identity(self):
        """Empty config dict should produce a no-op pipeline."""
        pipeline = build_preprocessing_pipeline({})
        image = np.random.rand(64, 64).astype(np.float32)
        result = pipeline(image)
        np.testing.assert_array_equal(result, image)

    def test_zscore_normalization(self):
        """Pipeline with zscore normalize should produce ~0 mean, ~1 std."""
        pipeline = build_preprocessing_pipeline({"normalize": "zscore"})
        image = np.random.rand(100, 100).astype(np.float32) * 255
        result = pipeline(image)
        assert abs(result.mean()) < 1e-4
        assert abs(result.std() - 1.0) < 1e-4

    def test_crop_center(self):
        """Pipeline with crop should produce correct spatial dimensions."""
        pipeline = build_preprocessing_pipeline({"crop_size": 64, "crop_mode": "center"})
        image = np.random.rand(200, 200).astype(np.float32)
        result = pipeline(image)
        assert result.shape == (64, 64)

    def test_combined_pipeline(self):
        """Pipeline with multiple steps should apply them in order."""
        config = {
            "normalize": "zscore",
            "crop_size": 64,
            "crop_mode": "center",
        }
        pipeline = build_preprocessing_pipeline(config)
        image = np.random.rand(256, 256).astype(np.float32) * 100
        result = pipeline(image)
        assert result.shape == (64, 64)


# ---------------------------------------------------------------------------
# Augmentation tests
# ---------------------------------------------------------------------------

from strain.data.augmentation import (
    Compose,
    random_elastic_deformation,
    random_flip,
    random_gamma,
    random_noise,
    random_rotation,
    random_scale,
)


class TestRandomRotation:
    """Tests for random_rotation."""

    def test_preserves_shape_image_only(self):
        """Rotated image should have the same shape (reshape=False)."""
        np.random.seed(0)
        image = np.random.rand(128, 128).astype(np.float32)
        result = random_rotation(image, max_angle=15.0)
        assert result.shape == image.shape

    def test_preserves_shape_with_mask(self):
        """Image and mask should both preserve shape."""
        np.random.seed(0)
        image = np.random.rand(128, 128).astype(np.float32)
        mask = np.zeros((128, 128), dtype=np.int32)
        mask[50:80, 50:80] = 1
        rotated_img, rotated_mask = random_rotation(image, mask, max_angle=15.0)
        assert rotated_img.shape == image.shape
        assert rotated_mask.shape == mask.shape

    def test_3d_input(self):
        """Should handle (D, H, W) arrays."""
        np.random.seed(0)
        image = np.random.rand(3, 64, 64).astype(np.float32)
        result = random_rotation(image, max_angle=10.0)
        assert result.shape == (3, 64, 64)


class TestRandomScale:
    """Tests for random_scale."""

    def test_no_crash(self):
        """Should run without errors."""
        np.random.seed(0)
        image = np.random.rand(64, 64).astype(np.float32)
        result = random_scale(image, scale_range=(0.9, 1.1))
        assert result.ndim == 2

    def test_with_mask(self):
        """Should return image and mask when mask is provided."""
        np.random.seed(0)
        image = np.random.rand(64, 64).astype(np.float32)
        mask = np.zeros((64, 64), dtype=np.int32)
        result = random_scale(image, mask, scale_range=(1.0, 1.0))
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestRandomGamma:
    """Tests for random_gamma."""

    def test_preserves_shape(self):
        """Gamma-corrected image should have the same shape."""
        np.random.seed(0)
        image = np.random.rand(64, 64).astype(np.float32)
        result = random_gamma(image, gamma_range=(0.7, 1.5))
        assert result.shape == image.shape

    def test_constant_image(self):
        """Constant image should be returned unchanged."""
        image = np.ones((32, 32), dtype=np.float32) * 5.0
        result = random_gamma(image, gamma_range=(0.5, 2.0))
        np.testing.assert_allclose(result, image, atol=1e-5)

    def test_output_range_nonnegative(self):
        """Output should stay within the input range for non-negative inputs."""
        np.random.seed(42)
        image = np.random.rand(64, 64).astype(np.float32) * 100
        result = random_gamma(image, gamma_range=(0.7, 1.5))
        assert result.min() >= image.min() - 1e-5
        assert result.max() <= image.max() + 1e-5


class TestRandomElasticDeformation:
    """Tests for random_elastic_deformation."""

    def test_preserves_shape(self):
        """Deformed image should have the same shape as input."""
        np.random.seed(0)
        image = np.random.rand(64, 64).astype(np.float32)
        result = random_elastic_deformation(image, alpha=50.0, sigma=5.0)
        assert result.shape == image.shape

    def test_with_mask(self):
        """Should return (image, mask) tuple."""
        np.random.seed(0)
        image = np.random.rand(64, 64).astype(np.float32)
        mask = np.zeros((64, 64), dtype=np.int32)
        mask[20:40, 20:40] = 1
        result_img, result_mask = random_elastic_deformation(image, mask, alpha=50.0, sigma=5.0)
        assert result_img.shape == image.shape
        assert result_mask.shape == mask.shape

    def test_3d_input(self):
        """Should handle (D, H, W) input."""
        np.random.seed(0)
        image = np.random.rand(3, 64, 64).astype(np.float32)
        result = random_elastic_deformation(image, alpha=50.0, sigma=5.0)
        assert result.shape == (3, 64, 64)


class TestRandomFlip:
    """Tests for random_flip."""

    def test_preserves_shape(self):
        """Flipped image should retain its shape."""
        np.random.seed(0)
        image = np.random.rand(64, 64).astype(np.float32)
        result = random_flip(image, prob=1.0)
        assert result.shape == image.shape

    def test_prob_zero_no_flip(self):
        """With prob=0.0, image should be unchanged."""
        image = np.arange(16).reshape(4, 4).astype(np.float32)
        result = random_flip(image, prob=0.0)
        np.testing.assert_array_equal(result, image)

    def test_horizontal_flip(self):
        """With prob=1.0 and axis=-1, columns should be reversed."""
        image = np.arange(16).reshape(4, 4).astype(np.float32)
        result = random_flip(image, axis=-1, prob=1.0)
        np.testing.assert_array_equal(result, np.flip(image, axis=-1))


class TestRandomNoise:
    """Tests for random_noise."""

    def test_preserves_shape(self):
        """Noisy image should have the same shape."""
        image = np.random.rand(64, 64).astype(np.float32)
        result = random_noise(image, std=0.02)
        assert result.shape == image.shape

    def test_noise_is_additive(self):
        """Noise should change pixel values."""
        np.random.seed(0)
        image = np.zeros((64, 64), dtype=np.float32)
        result = random_noise(image, std=1.0)
        assert not np.allclose(result, image)


class TestCompose:
    """Tests for the Compose augmentation chain."""

    def test_chains_transforms(self):
        """Compose should apply multiple transforms."""
        np.random.seed(42)
        image = np.random.rand(64, 64).astype(np.float32)
        mask = np.ones((64, 64), dtype=np.int32)

        aug = Compose([
            (lambda img, msk: random_rotation(img, msk, 15.0), 1.0),
            (lambda img, msk: (random_noise(img, 0.02), msk), 1.0),
        ])
        result_img, result_mask = aug(image, mask)
        assert result_img.shape == image.shape
        assert result_mask.shape == mask.shape

    def test_no_mask(self):
        """Compose should work without a mask."""
        np.random.seed(42)
        image = np.random.rand(64, 64).astype(np.float32)
        aug = Compose([
            (lambda img, msk: (random_noise(img, 0.01), msk), 1.0),
        ])
        # Pass mask=None - returns just the image
        result = aug(image, mask=None)
        # Without mask, returns image directly
        assert isinstance(result, np.ndarray)
        assert result.shape == image.shape

    def test_probability_zero_skips(self):
        """Transforms with probability 0 should be skipped."""
        image = np.random.rand(64, 64).astype(np.float32)
        original = image.copy()
        aug = Compose([
            (lambda img, msk: (img * 0, msk), 0.0),  # Would zero out image
        ])
        result = aug(image)
        np.testing.assert_array_equal(result, original)

    def test_repr(self):
        """Compose repr should not crash."""
        aug = Compose([
            (lambda img, msk: (img, msk), 0.5),
        ])
        r = repr(aug)
        assert "Compose" in r
