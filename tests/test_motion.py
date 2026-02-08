"""Tests for the motion estimation module: CarMEN network, warping, and losses.

Uses synthetic torch tensors -- no real medical data or trained models required.
"""

from __future__ import annotations

import pytest

try:
    import torch
except (ImportError, OSError):
    pytest.skip("PyTorch is not available or broken in this environment", allow_module_level=True)

import numpy as np

from strain.models.motion.carmen import (
    CarMEN,
    MotionDecoder,
    MotionEncoder,
    SegmentationGuidedAttention,
)
from strain.models.motion.losses import (
    BendingEnergy,
    CyclicConsistencyLoss,
    DiffusionRegularizer,
    NCC,
)
from strain.models.motion.warp import (
    _make_identity_grid,
    compose_displacements,
    spatial_transform,
)


# ---------------------------------------------------------------------------
# MotionEncoder tests
# ---------------------------------------------------------------------------


class TestMotionEncoder:
    """Tests for the shared MotionEncoder."""

    def test_output_feature_count(self):
        """Encoder should return one feature map per stage."""
        enc = MotionEncoder(in_ch=1, features=(16, 32, 64, 128))
        x = torch.randn(1, 1, 128, 128)
        features = enc(x)
        assert len(features) == 4

    def test_progressive_downsampling(self):
        """Each feature map should be spatially smaller than the previous."""
        enc = MotionEncoder(in_ch=1, features=(16, 32, 64, 128))
        x = torch.randn(1, 1, 128, 128)
        features = enc(x)
        for i in range(1, len(features)):
            assert features[i].shape[2] < features[i - 1].shape[2], (
                f"Feature {i} spatial dim should be smaller than feature {i - 1}"
            )

    def test_channel_counts(self):
        """Feature maps should have the correct number of channels."""
        enc = MotionEncoder(in_ch=1, features=(16, 32, 64, 128))
        x = torch.randn(1, 1, 128, 128)
        features = enc(x)
        expected_channels = [16, 32, 64, 128]
        for feat, expected_ch in zip(features, expected_channels):
            assert feat.shape[1] == expected_ch


# ---------------------------------------------------------------------------
# CarMEN full model tests
# ---------------------------------------------------------------------------


class TestCarMEN:
    """Tests for the CarMEN motion estimation network."""

    def test_forward_output_shape(self):
        """CarMEN: two (B,1,128,128) inputs -> (B,2,128,128) displacement."""
        model = CarMEN(in_channels=1, features=(16, 32, 64, 128))
        model.eval()
        source = torch.randn(2, 1, 128, 128)
        target = torch.randn(2, 1, 128, 128)
        with torch.no_grad():
            output = model(source, target)
        assert output.shape == (2, 2, 128, 128), f"Expected (2,2,128,128), got {output.shape}"

    def test_single_sample(self):
        """Should work with batch size 1."""
        model = CarMEN(in_channels=1)
        model.eval()
        source = torch.randn(1, 1, 128, 128)
        target = torch.randn(1, 1, 128, 128)
        with torch.no_grad():
            output = model(source, target)
        assert output.shape == (1, 2, 128, 128)

    def test_zero_initialized_output(self):
        """Newly initialized CarMEN should produce near-zero displacements."""
        model = CarMEN(in_channels=1)
        model.eval()
        source = torch.randn(1, 1, 64, 64)
        target = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            output = model(source, target)
        max_disp = output.abs().max().item()
        assert max_disp < 10.0, f"Zero-init model should produce small displacements, got max {max_disp}"

    def test_multi_scale_output(self):
        """With multi_scale=True, should return a list of displacement fields."""
        model = CarMEN(in_channels=1, features=(16, 32, 64, 128), multi_scale=True)
        model.eval()
        source = torch.randn(1, 1, 128, 128)
        target = torch.randn(1, 1, 128, 128)
        with torch.no_grad():
            output = model(source, target)
        assert isinstance(output, list), "Multi-scale output should be a list"
        assert len(output) > 0
        assert output[-1].shape[1] == 2
        assert output[-1].shape[2:] == (128, 128)

    def test_seg_attention(self):
        """With use_seg_attention=True, should accept a segmentation mask."""
        model = CarMEN(in_channels=1, use_seg_attention=True)
        model.eval()
        source = torch.randn(1, 1, 128, 128)
        target = torch.randn(1, 1, 128, 128)
        seg_mask = torch.ones(1, 1, 128, 128)
        with torch.no_grad():
            output = model(source, target, seg_mask=seg_mask)
        assert output.shape == (1, 2, 128, 128)

    def test_gradient_flow(self):
        """Gradients should flow from the output back to the source."""
        model = CarMEN(in_channels=1, features=(16, 32, 64, 128))
        source = torch.randn(1, 1, 64, 64, requires_grad=True)
        target = torch.randn(1, 1, 64, 64)
        output = model(source, target)
        loss = output.mean()
        loss.backward()
        assert source.grad is not None, "Gradients should flow to source"

    def test_encoder_instantiation(self):
        """CarMEN encoder should instantiate and produce feature list."""
        model = CarMEN(in_channels=1, features=(16, 32, 64, 128))
        source = torch.randn(1, 1, 128, 128)
        features = model.encoder(source)
        assert len(features) == 4
        assert features[0].shape[1] == 16
        assert features[-1].shape[1] == 128

    @pytest.mark.skip(reason="requires trained model")
    def test_trained_model_registration(self):
        """Placeholder for testing with a trained model checkpoint."""
        pass


# ---------------------------------------------------------------------------
# SegmentationGuidedAttention tests
# ---------------------------------------------------------------------------


class TestSegmentationGuidedAttention:
    """Tests for the SegmentationGuidedAttention module."""

    def test_output_shape(self):
        """Output should match displacement shape."""
        attn = SegmentationGuidedAttention()
        disp = torch.randn(2, 2, 64, 64)
        mask = torch.ones(2, 1, 64, 64)
        result = attn(disp, mask)
        assert result.shape == disp.shape

    def test_mask_interpolation(self):
        """Should handle masks of different spatial size via interpolation."""
        attn = SegmentationGuidedAttention()
        disp = torch.randn(1, 2, 64, 64)
        mask = torch.ones(1, 1, 32, 32)  # different size
        result = attn(disp, mask)
        assert result.shape == disp.shape


# ---------------------------------------------------------------------------
# Spatial transform (warp) tests
# ---------------------------------------------------------------------------


class TestSpatialTransform:
    """Tests for the spatial_transform function."""

    def test_zero_displacement_identity(self):
        """Zero displacement should return (approximately) the source image."""
        source = torch.randn(1, 1, 64, 64)
        displacement = torch.zeros(1, 2, 64, 64)
        warped = spatial_transform(source, displacement)
        # With zero displacement, warped should equal source
        diff = (warped - source).abs().max().item()
        assert diff < 1e-4, f"Zero displacement should produce identity warp, max diff = {diff}"

    def test_output_shape(self):
        """Warped image should match source shape."""
        source = torch.randn(2, 1, 64, 64)
        displacement = torch.randn(2, 2, 64, 64) * 0.01
        warped = spatial_transform(source, displacement)
        assert warped.shape == source.shape

    def test_multichannel_source(self):
        """Should work with multi-channel source (B, C, H, W) where C > 1."""
        source = torch.randn(1, 3, 64, 64)
        displacement = torch.zeros(1, 2, 64, 64)
        warped = spatial_transform(source, displacement)
        assert warped.shape == source.shape

    def test_differentiable(self):
        """Warping should be differentiable w.r.t. displacement.

        We create the displacement as a leaf Parameter so that .grad is
        populated after backward().  The multiplication ``* 0.1`` in the
        original version creates a non-leaf tensor whose .grad is not
        retained by default.
        """
        source = torch.randn(1, 1, 32, 32)
        displacement = torch.nn.Parameter(torch.randn(1, 2, 32, 32) * 0.1)
        warped = spatial_transform(source, displacement)
        loss = warped.sum()
        loss.backward()
        assert displacement.grad is not None, "Gradients should flow through spatial_transform"


class TestMakeIdentityGrid:
    """Tests for the identity grid helper."""

    def test_shape(self):
        """Identity grid should have shape (1, 2, H, W)."""
        grid = _make_identity_grid(64, 64, torch.device("cpu"), torch.float32)
        assert grid.shape == (1, 2, 64, 64)

    def test_x_channel_values(self):
        """Channel 0 (x) should contain column indices."""
        grid = _make_identity_grid(32, 48, torch.device("cpu"), torch.float32)
        # Check that x values at row 0 are 0, 1, 2, ..., 47
        x_row0 = grid[0, 0, 0, :]
        expected = torch.arange(48, dtype=torch.float32)
        assert torch.allclose(x_row0, expected)


class TestComposeDisplacements:
    """Tests for displacement field composition."""

    def test_zero_fields_identity(self):
        """Composing two zero displacements should yield zero."""
        d1 = torch.zeros(1, 2, 32, 32)
        d2 = torch.zeros(1, 2, 32, 32)
        composed = compose_displacements(d1, d2)
        assert composed.abs().max().item() < 1e-5

    def test_output_shape(self):
        """Composed displacement should have the same shape."""
        d1 = torch.randn(2, 2, 32, 32) * 0.1
        d2 = torch.randn(2, 2, 32, 32) * 0.1
        composed = compose_displacements(d1, d2)
        assert composed.shape == d1.shape


# ---------------------------------------------------------------------------
# Motion loss tests
# ---------------------------------------------------------------------------


class TestNCC:
    """Tests for the Normalized Cross-Correlation loss."""

    def test_identical_images_low_loss(self):
        """NCC loss for identical images should be near 0."""
        ncc = NCC(window_size=9)
        image = torch.randn(1, 1, 64, 64)
        loss = ncc(image, image)
        assert loss.item() < 0.05, f"NCC loss for identical images should be ~0, got {loss.item()}"

    def test_returns_scalar(self):
        """NCC should return a scalar."""
        ncc = NCC()
        a = torch.randn(2, 1, 32, 32)
        b = torch.randn(2, 1, 32, 32)
        loss = ncc(a, b)
        assert loss.ndim == 0

    def test_loss_nonnegative(self):
        """NCC loss (1 - NCC) should be non-negative for reasonable inputs."""
        ncc = NCC()
        a = torch.randn(1, 1, 64, 64)
        # NCC can technically exceed 1 due to the epsilon in denominator,
        # so we just check it's bounded.
        loss = ncc(a, a)
        assert loss.item() >= -0.1, "NCC loss should be approximately non-negative"

    def test_gradient_flows(self):
        """Gradients should flow through NCC."""
        ncc = NCC()
        a = torch.randn(1, 1, 32, 32, requires_grad=True)
        b = torch.randn(1, 1, 32, 32)
        loss = ncc(a, b)
        loss.backward()
        assert a.grad is not None


class TestDiffusionRegularizer:
    """Tests for the DiffusionRegularizer."""

    def test_constant_field_zero_loss(self):
        """Constant displacement field should have zero gradient -> loss ~ 0."""
        reg = DiffusionRegularizer()
        displacement = torch.ones(1, 2, 64, 64) * 5.0  # constant everywhere
        loss = reg(displacement)
        assert loss.item() < 1e-10, f"Constant field should give ~0 loss, got {loss.item()}"

    def test_zero_field_zero_loss(self):
        """Zero displacement should give zero loss."""
        reg = DiffusionRegularizer()
        displacement = torch.zeros(1, 2, 32, 32)
        loss = reg(displacement)
        assert loss.item() < 1e-10

    def test_nonzero_for_varying_field(self):
        """A varying field should produce non-zero regularization loss."""
        reg = DiffusionRegularizer()
        displacement = torch.randn(1, 2, 32, 32) * 5.0
        loss = reg(displacement)
        assert loss.item() > 0.0

    def test_returns_scalar(self):
        """Should return a scalar tensor."""
        reg = DiffusionRegularizer()
        displacement = torch.randn(2, 2, 32, 32)
        loss = reg(displacement)
        assert loss.ndim == 0


class TestBendingEnergy:
    """Tests for the BendingEnergy regularizer."""

    def test_constant_field_zero_loss(self):
        """Constant field has zero second derivatives -> loss ~ 0."""
        be = BendingEnergy()
        displacement = torch.ones(1, 2, 32, 32) * 3.0
        loss = be(displacement)
        assert loss.item() < 1e-10

    def test_linear_field_zero_loss(self):
        """Linear displacement field has zero second derivatives -> loss ~ 0."""
        be = BendingEnergy()
        # Linear field: u(x,y) = ax + by
        H, W = 32, 32
        x = torch.arange(W, dtype=torch.float32).unsqueeze(0).expand(H, -1)
        y = torch.arange(H, dtype=torch.float32).unsqueeze(1).expand(-1, W)
        dx = 0.5 * x + 0.3 * y
        dy = -0.2 * x + 0.7 * y
        displacement = torch.stack([dx, dy], dim=0).unsqueeze(0)  # (1, 2, H, W)
        loss = be(displacement)
        assert loss.item() < 1e-4, f"Linear field should give ~0 bending energy, got {loss.item()}"

    def test_returns_scalar(self):
        """Should return a scalar."""
        be = BendingEnergy()
        displacement = torch.randn(1, 2, 32, 32)
        loss = be(displacement)
        assert loss.ndim == 0


class TestCyclicConsistencyLoss:
    """Tests for the CyclicConsistencyLoss."""

    def test_zero_fields_zero_loss(self):
        """Zero forward and backward displacements should give zero loss."""
        cc = CyclicConsistencyLoss()
        fwd = torch.zeros(1, 2, 32, 32)
        bwd = torch.zeros(1, 2, 32, 32)
        loss = cc(fwd, bwd)
        assert loss.item() < 1e-10

    def test_returns_scalar(self):
        """Should return a scalar."""
        cc = CyclicConsistencyLoss()
        fwd = torch.randn(1, 2, 32, 32)
        bwd = torch.randn(1, 2, 32, 32)
        loss = cc(fwd, bwd)
        assert loss.ndim == 0

    def test_opposite_fields_low_loss(self):
        """Opposite displacements should produce low (but not necessarily zero) loss.

        For small displacements, fwd + bwd ~ 0 when bwd = -fwd.
        For large displacements the warping makes this inexact.
        """
        cc = CyclicConsistencyLoss()
        fwd = torch.randn(1, 2, 32, 32) * 0.01  # very small
        bwd = -fwd.clone()
        loss = cc(fwd, bwd)
        assert loss.item() < 0.01, f"Small opposite displacements should give low loss, got {loss.item()}"
