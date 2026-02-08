"""Tests for the segmentation module: UNet model, losses, and metrics.

Uses synthetic torch tensors and numpy arrays -- no real medical data required.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
except (ImportError, OSError):
    pytest.skip("PyTorch is not available or broken in this environment", allow_module_level=True)

from strain.models.segmentation.losses import DiceCELoss, DiceLoss
from strain.models.segmentation.metrics import (
    LABEL_MAP,
    _extract_surface_points,
    compute_volumes,
    dice_score,
    hausdorff_distance_95,
)
from strain.models.segmentation.unet import ConvBlock, ResBlock, UNet


# ---------------------------------------------------------------------------
# UNet architecture tests
# ---------------------------------------------------------------------------


class TestConvBlock:
    """Tests for the ConvBlock building block."""

    def test_output_shape(self):
        """ConvBlock should map (B, in_ch, H, W) -> (B, out_ch, H, W)."""
        block = ConvBlock(1, 32)
        x = torch.randn(2, 1, 64, 64)
        out = block(x)
        assert out.shape == (2, 32, 64, 64)

    def test_different_channels(self):
        """Should work with various in/out channel combinations."""
        block = ConvBlock(16, 64)
        x = torch.randn(1, 16, 32, 32)
        out = block(x)
        assert out.shape == (1, 64, 32, 32)


class TestResBlock:
    """Tests for the ResBlock building block."""

    def test_output_shape(self):
        """ResBlock should preserve spatial dims and produce correct channels."""
        block = ResBlock(1, 32)
        x = torch.randn(2, 1, 64, 64)
        out = block(x)
        assert out.shape == (2, 32, 64, 64)

    def test_skip_identity_when_same_channels(self):
        """When in_ch == out_ch, skip connection should be Identity."""
        block = ResBlock(32, 32)
        assert isinstance(block.skip, torch.nn.Identity)

    def test_skip_conv_when_different_channels(self):
        """When in_ch != out_ch, skip connection should be a Conv2d."""
        block = ResBlock(1, 32)
        assert isinstance(block.skip, torch.nn.Conv2d)


class TestUNet:
    """Tests for the UNet segmentation model."""

    def test_forward_pass_shape(self):
        """UNet with default params: (B,1,128,128) -> (B,4,128,128)."""
        model = UNet(in_channels=1, num_classes=4)
        model.eval()
        x = torch.randn(2, 1, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 4, 128, 128), f"Expected (2,4,128,128), got {out.shape}"

    def test_single_sample(self):
        """Should work with batch size 1."""
        model = UNet(in_channels=1, num_classes=4)
        model.eval()
        x = torch.randn(1, 1, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 4, 128, 128)

    def test_custom_features(self):
        """Should work with custom feature counts."""
        model = UNet(in_channels=1, num_classes=2, features=(32, 64, 128))
        model.eval()
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2, 64, 64)

    def test_deep_supervision_training(self):
        """In training mode with deep_supervision, should return (main, [aux])."""
        model = UNet(in_channels=1, num_classes=4, deep_supervision=True)
        model.train()
        x = torch.randn(2, 1, 128, 128)
        out = model(x)
        assert isinstance(out, tuple), "Training with deep supervision should return a tuple"
        main, aux_list = out
        assert main.shape == (2, 4, 128, 128)
        assert isinstance(aux_list, list)
        assert len(aux_list) > 0
        # All auxiliary outputs should be upsampled to input resolution
        for aux in aux_list:
            assert aux.shape[2:] == (128, 128), f"Aux output shape {aux.shape} should match input spatial dims"

    def test_deep_supervision_eval(self):
        """In eval mode with deep_supervision, should return only the main output."""
        model = UNet(in_channels=1, num_classes=4, deep_supervision=True)
        model.eval()
        x = torch.randn(1, 1, 128, 128)
        with torch.no_grad():
            out = model(x)
        # In eval mode, should return just a tensor, not a tuple
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 4, 128, 128)

    def test_dropout(self):
        """Model with dropout should still produce correct shape."""
        model = UNet(in_channels=1, num_classes=4, dropout=0.2)
        model.eval()
        x = torch.randn(1, 1, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 4, 128, 128)

    @pytest.mark.skip(reason="requires trained model and GPU")
    def test_trained_model_inference(self):
        """Placeholder for testing with a trained model checkpoint."""
        pass


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------


class TestDiceLoss:
    """Tests for the DiceLoss."""

    def test_returns_scalar(self):
        """DiceLoss should return a scalar tensor."""
        loss_fn = DiceLoss()
        logits = torch.randn(2, 4, 32, 32)
        targets = torch.randint(0, 4, (2, 32, 32))
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0, "Loss should be a scalar"

    def test_loss_in_range(self):
        """DiceLoss should be in [0, 1]."""
        loss_fn = DiceLoss()
        logits = torch.randn(4, 4, 32, 32)
        targets = torch.randint(0, 4, (4, 32, 32))
        loss = loss_fn(logits, targets)
        assert 0.0 <= loss.item() <= 1.0 + 1e-5, f"Dice loss {loss.item()} out of [0,1]"

    def test_perfect_prediction_low_loss(self):
        """When logits strongly match targets, Dice loss should be low."""
        loss_fn = DiceLoss()
        # Create perfect logits: one-hot targets as logits (with high confidence)
        targets = torch.zeros(1, 32, 32, dtype=torch.long)
        targets[0, 10:20, 10:20] = 1
        targets[0, 20:30, 10:20] = 2
        targets[0, 10:20, 20:30] = 3

        # Build logits that strongly predict the correct class
        logits = torch.full((1, 4, 32, 32), -10.0)
        for cls in range(4):
            logits[0, cls][targets[0] == cls] = 10.0

        loss = loss_fn(logits, targets)
        assert loss.item() < 0.05, f"Loss for perfect prediction should be ~0, got {loss.item()}"

    def test_gradient_flows(self):
        """DiceLoss should allow gradients to flow to logits."""
        loss_fn = DiceLoss()
        logits = torch.randn(2, 4, 32, 32, requires_grad=True)
        targets = torch.randint(0, 4, (2, 32, 32))
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None, "Gradients should flow through DiceLoss"


class TestDiceCELoss:
    """Tests for the combined DiceCELoss."""

    def test_computes_without_error(self):
        """DiceCELoss should compute a finite scalar loss."""
        loss_fn = DiceCELoss()
        logits = torch.randn(2, 4, 32, 32)
        targets = torch.randint(0, 4, (2, 32, 32))
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0
        assert torch.isfinite(loss), f"DiceCELoss returned non-finite value: {loss.item()}"

    def test_gradient_flows(self):
        """Gradients should flow through the combined loss."""
        loss_fn = DiceCELoss()
        logits = torch.randn(2, 4, 32, 32, requires_grad=True)
        targets = torch.randint(0, 4, (2, 32, 32))
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None

    def test_custom_weights(self):
        """Should accept custom Dice/CE weight ratios."""
        loss_fn = DiceCELoss(dice_weight=0.8, ce_weight=0.2)
        logits = torch.randn(2, 4, 32, 32)
        targets = torch.randint(0, 4, (2, 32, 32))
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


class TestDiceScore:
    """Tests for the dice_score metric."""

    def test_perfect_prediction(self):
        """Dice score should be 1.0 when pred == target."""
        pred = np.zeros((32, 32), dtype=np.int64)
        pred[5:15, 5:15] = 1
        pred[15:25, 5:15] = 2
        pred[5:15, 15:25] = 3
        target = pred.copy()

        scores = dice_score(pred, target, num_classes=4)
        for label_name, score in scores.items():
            assert abs(score - 1.0) < 1e-7, f"Dice for {label_name} should be 1.0, got {score}"

    def test_completely_wrong(self):
        """Dice should be 0 when pred and target have no overlap."""
        pred = np.zeros((32, 32), dtype=np.int64)
        pred[0:10, 0:10] = 1
        target = np.zeros((32, 32), dtype=np.int64)
        target[20:32, 20:32] = 1

        scores = dice_score(pred, target, num_classes=4)
        assert scores["RV"] == 0.0, f"Dice for class 1 with no overlap should be 0, got {scores['RV']}"

    def test_both_empty(self):
        """When both pred and target are all background, Dice should be 1.0."""
        pred = np.zeros((32, 32), dtype=np.int64)
        target = np.zeros((32, 32), dtype=np.int64)
        scores = dice_score(pred, target, num_classes=4)
        for name, score in scores.items():
            assert score == 1.0, f"Both-empty Dice for {name} should be 1.0"

    def test_returns_expected_keys(self):
        """Should return keys for RV, Myo, LV."""
        pred = np.zeros((32, 32), dtype=np.int64)
        target = np.zeros((32, 32), dtype=np.int64)
        scores = dice_score(pred, target, num_classes=4)
        assert "RV" in scores
        assert "Myo" in scores
        assert "LV" in scores

    def test_3d_input(self):
        """Dice should work with (D, H, W) arrays."""
        pred = np.zeros((3, 32, 32), dtype=np.int64)
        pred[:, 5:15, 5:15] = 1
        target = pred.copy()
        scores = dice_score(pred, target, num_classes=4)
        assert scores["RV"] == 1.0


class TestHausdorffDistance95:
    """Tests for the hausdorff_distance_95 metric."""

    def test_perfect_overlap(self):
        """HD95 should be 0.0 for identical masks."""
        pred = np.zeros((32, 32), dtype=np.int64)
        pred[10:20, 10:20] = 1
        target = pred.copy()
        results = hausdorff_distance_95(pred, target, num_classes=4)
        assert results["RV"] == 0.0

    def test_both_empty(self):
        """HD95 should be 0.0 when both masks are empty."""
        pred = np.zeros((32, 32), dtype=np.int64)
        target = np.zeros((32, 32), dtype=np.int64)
        results = hausdorff_distance_95(pred, target, num_classes=4)
        assert results["RV"] == 0.0

    def test_one_empty_returns_inf(self):
        """HD95 should be inf when one mask is empty and the other is not."""
        pred = np.zeros((32, 32), dtype=np.int64)
        pred[10:20, 10:20] = 1
        target = np.zeros((32, 32), dtype=np.int64)
        results = hausdorff_distance_95(pred, target, num_classes=4)
        assert results["RV"] == float("inf")


class TestExtractSurfacePoints:
    """Tests for the surface point extraction helper."""

    def test_empty_mask(self):
        """Empty mask should return an empty array."""
        mask = np.zeros((32, 32), dtype=np.uint8)
        pts = _extract_surface_points(mask)
        assert pts.shape == (0, 2)

    def test_filled_mask(self):
        """Non-empty mask should return surface point coordinates."""
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10:20, 10:20] = 1
        pts = _extract_surface_points(mask)
        assert pts.shape[0] > 0
        assert pts.shape[1] == 2


class TestComputeVolumes:
    """Tests for the compute_volumes function."""

    def test_single_frame(self):
        """Single 3-D frame (not time-resolved) should return LV_vol, RV_vol."""
        seg = np.zeros((1, 32, 32), dtype=np.int64)
        seg[0, 5:15, 5:15] = 1  # RV
        seg[0, 5:15, 15:25] = 3  # LV
        result = compute_volumes(seg, spacing=(1.0, 1.0))
        # With ndim==3 and shape[0]==1, _has_time_axis returns False
        # so we get 'LV_vol' and 'RV_vol'
        assert "LV_vol" in result or "LV_EDV" in result

    def test_time_resolved(self):
        """Time-resolved segmentation should return EDV, ESV, LVEF."""
        # (T, H, W) with T >= 2
        seg = np.zeros((5, 32, 32), dtype=np.int64)
        # Frame 0: large LV (ED)
        seg[0, 5:25, 5:25] = 3
        # Frame 2: small LV (ES)
        seg[2, 10:15, 10:15] = 3
        # Other frames: medium
        seg[1, 8:20, 8:20] = 3
        seg[3, 8:20, 8:20] = 3
        seg[4, 8:20, 8:20] = 3

        result = compute_volumes(seg, spacing=(1.0, 1.0))
        assert "LV_EDV" in result
        assert "LV_ESV" in result
        assert "LVEF" in result
        assert result["LV_EDV"] >= result["LV_ESV"]
        assert 0.0 <= result["LVEF"] <= 100.0
