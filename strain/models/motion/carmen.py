"""Cardiac Motion Estimation Network (registration-based).

Implements CarMEN — an unsupervised registration network that predicts dense
displacement fields between pairs of cardiac cine MRI frames.  The architecture
follows a U-Net-style encoder-decoder with:

* Shared encoder for source and target frames
* Skip connections that concatenate source + target features
* Multi-scale displacement prediction (1/4, 1/2, full resolution)
* Optional segmentation-guided attention (weight displacement by myocardial mask)
* Zero-initialised flow heads so the network starts as an identity transform
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class MotionEncoder(nn.Module):
    """Shared encoder for source and target frames.

    Each stage: Conv2d(stride=2) -> InstanceNorm -> LeakyReLU.
    Returns a list of feature maps at progressively lower resolutions.
    """

    def __init__(self, in_ch: int = 1, features: tuple[int, ...] = (16, 32, 64, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        ch = in_ch
        for f in features:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(ch, f, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(f),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            ch = f
        self.encoder = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features: list[torch.Tensor] = []
        for layer in self.encoder:
            x = layer(x)
            features.append(x)
        return features


# ---------------------------------------------------------------------------
# Multi-scale flow heads
# ---------------------------------------------------------------------------

class _FlowHead(nn.Module):
    """A single zero-initialised 1x1 conv producing a 2-channel flow field."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 2, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------------
# Decoder with multi-scale prediction
# ---------------------------------------------------------------------------

class MotionDecoder(nn.Module):
    """Decoder that predicts dense displacement fields at multiple scales.

    The decoder upsamples the bottleneck representation with skip connections
    from the encoder.  At the final three decoder stages it outputs
    displacement predictions at 1/4, 1/2, and full decoder resolution.  During
    inference only the full-resolution field is used; during training the
    coarser fields may contribute to a coarse-to-fine loss.

    Parameters
    ----------
    features : tuple[int, ...]
        Channel counts from coarsest to finest (e.g. (128, 64, 32, 16)).
    multi_scale : bool
        If *True* return a list of displacement fields at three scales.
        If *False* return only the full-resolution field.
    """

    def __init__(
        self,
        features: tuple[int, ...] = (128, 64, 32, 16),
        multi_scale: bool = False,
    ):
        super().__init__()
        self.multi_scale = multi_scale

        up_layers: list[nn.Module] = []
        skip_projs: list[nn.Module] = []
        for i in range(len(features) - 1):
            # First stage: concatenated source+target bottleneck; subsequent: output of prior stage
            in_ch = features[i] * 2 if i == 0 else features[i]
            up_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, features[i + 1], 4, stride=2, padding=1),
                    nn.InstanceNorm2d(features[i + 1]),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            # Project concatenated skip (src + tgt) to match decoder channels
            skip_ch = features[i + 1] * 2  # src_feat + tgt_feat at this level
            skip_projs.append(
                nn.Conv2d(skip_ch, features[i + 1], kernel_size=1, bias=False)
            )
        self.decoder = nn.ModuleList(up_layers)
        self.skip_projs = nn.ModuleList(skip_projs)

        # Full-resolution flow head (always present)
        self.flow_head = _FlowHead(features[-1])

        # Multi-scale heads at intermediate decoder outputs
        if multi_scale and len(features) >= 3:
            # head at 1/4 resolution (after first up-sample)
            self.flow_head_quarter = _FlowHead(features[1])
            # head at 1/2 resolution (after second up-sample)
            self.flow_head_half = _FlowHead(features[2] if len(features) > 3 else features[-1])
        else:
            self.flow_head_quarter = None
            self.flow_head_half = None

    # ------------------------------------------------------------------ #

    def forward(
        self,
        source_features: list[torch.Tensor],
        target_features: list[torch.Tensor],
    ) -> torch.Tensor | list[torch.Tensor]:
        """Decode displacement field(s).

        Returns
        -------
        If ``multi_scale`` is *False*:
            (B, 2, H_dec, W_dec)  — single displacement field
        If ``multi_scale`` is *True*:
            list of displacement fields ordered **coarse -> fine**:
            [(B,2,H/4,W/4), (B,2,H/2,W/2), (B,2,H,W)]
        """
        x = torch.cat([source_features[-1], target_features[-1]], dim=1)

        multi_scale_flows: list[torch.Tensor] = []

        for i, layer in enumerate(self.decoder):
            x = layer(x)

            # Collect multi-scale predictions at intermediate stages
            if self.multi_scale:
                # After first decoder stage => 1/4 resolution
                if i == 0 and self.flow_head_quarter is not None:
                    multi_scale_flows.append(self.flow_head_quarter(x))
                # After second decoder stage => 1/2 resolution
                if i == 1 and self.flow_head_half is not None:
                    multi_scale_flows.append(self.flow_head_half(x))

            # Skip connection: project concatenated encoder features and add
            if i + 1 < len(self.decoder):
                skip_idx = -(i + 2)
                skip = torch.cat(
                    [source_features[skip_idx], target_features[skip_idx]], dim=1
                )
                x = x + self.skip_projs[i](skip)

        # Full-resolution flow
        full_flow = self.flow_head(x)

        if self.multi_scale:
            multi_scale_flows.append(full_flow)
            return multi_scale_flows

        return full_flow


# ---------------------------------------------------------------------------
# Segmentation-guided attention
# ---------------------------------------------------------------------------

class SegmentationGuidedAttention(nn.Module):
    """Modulate displacement field using a myocardial segmentation mask.

    The mask (B, 1, H, W) is converted to a soft attention map via a small
    learned network so that the displacement field concentrates on the
    myocardial region while still allowing small displacements elsewhere.
    """

    def __init__(self, displacement_channels: int = 2):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        displacement: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply segmentation-guided attention to a displacement field.

        Parameters
        ----------
        displacement : (B, 2, H, W)
        mask : (B, 1, H, W)  — binary or soft myocardial mask

        Returns
        -------
        (B, 2, H, W)  — modulated displacement field
        """
        if mask.shape[2:] != displacement.shape[2:]:
            mask = F.interpolate(
                mask, size=displacement.shape[2:], mode="bilinear", align_corners=True
            )
        attention = self.gate(mask)  # (B, 1, H, W) in [0, 1]
        return displacement * attention


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class CarMEN(nn.Module):
    """Cardiac Motion Estimation Network.

    Predicts a dense 2D displacement field from a pair of cardiac frames
    using an unsupervised registration approach.

    Parameters
    ----------
    in_channels : int
        Number of input channels per frame (default 1 for grayscale cine MRI).
    features : tuple[int, ...]
        Channel counts for the encoder stages.
    multi_scale : bool
        Return displacement predictions at 3 scales for coarse-to-fine loss.
    use_seg_attention : bool
        If *True*, allow a segmentation mask to gate the displacement field
        (requires passing ``seg_mask`` to :meth:`forward`).
    """

    def __init__(
        self,
        in_channels: int = 1,
        features: tuple[int, ...] = (16, 32, 64, 128),
        multi_scale: bool = False,
        use_seg_attention: bool = False,
    ):
        super().__init__()
        self.multi_scale = multi_scale
        self.use_seg_attention = use_seg_attention

        self.encoder = MotionEncoder(in_channels, features)
        self.decoder = MotionDecoder(features[::-1], multi_scale=multi_scale)

        if use_seg_attention:
            self.seg_attention = SegmentationGuidedAttention()
        else:
            self.seg_attention = None

    # ------------------------------------------------------------------ #

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        seg_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Predict displacement field(s) from source to target.

        Parameters
        ----------
        source : (B, 1, H, W)
        target : (B, 1, H, W)
        seg_mask : (B, 1, H, W) or *None*
            Optional myocardial segmentation mask for guided attention.

        Returns
        -------
        If ``multi_scale`` is *False*:
            (B, 2, H, W) displacement field in **pixels**.
        If ``multi_scale`` is *True*:
            list of (B, 2, H_s, W_s) from coarse to fine, each upsampled to
            its corresponding target resolution.
        """
        input_size = source.shape[2:]

        src_features = self.encoder(source)
        tgt_features = self.encoder(target)
        output = self.decoder(src_features, tgt_features)

        if self.multi_scale:
            # output is a list of flows at different scales
            flows: list[torch.Tensor] = output  # type: ignore[assignment]
            # Upsample each to corresponding target resolution
            n_scales = len(flows)
            upsampled: list[torch.Tensor] = []
            for i, flow in enumerate(flows):
                # Scale factor: last element is full resolution
                if i == n_scales - 1:
                    target_h, target_w = input_size
                else:
                    scale = 2 ** (n_scales - 1 - i)
                    target_h = input_size[0] // scale
                    target_w = input_size[1] // scale
                flow_up = F.interpolate(
                    flow,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=True,
                )
                # Apply seg attention to each scale
                if self.seg_attention is not None and seg_mask is not None:
                    flow_up = self.seg_attention(flow_up, seg_mask)
                upsampled.append(flow_up)
            return upsampled
        else:
            # Single-scale
            displacement: torch.Tensor = output  # type: ignore[assignment]
            displacement = F.interpolate(
                displacement, size=input_size, mode="bilinear", align_corners=True
            )
            if self.seg_attention is not None and seg_mask is not None:
                displacement = self.seg_attention(displacement, seg_mask)
            return displacement
