"""2D U-Net with residual encoder for cardiac segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Double convolution block with batch normalization and ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBlock(nn.Module):
    """Residual convolution block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.skip(x)


class UNet(nn.Module):
    """U-Net for cardiac segmentation.

    Args:
        in_channels: Number of input channels (1 for grayscale MRI).
        num_classes: Number of output classes (4: bg, RV, myo, LV).
        features: Channel counts for each encoder stage.
        dropout: Dropout probability between encoder stages (0.0 to disable).
        deep_supervision: If True, return auxiliary outputs from intermediate
            decoder stages during training for deep supervision losses.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        features: tuple[int, ...] = (64, 128, 256, 512),
        dropout: float = 0.0,
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.enc_dropouts = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(ResBlock(ch, f))
            self.pools.append(nn.MaxPool2d(2))
            self.enc_dropouts.append(
                nn.Dropout2d(p=dropout) if dropout > 0.0 else nn.Identity()
            )
            ch = f

        # Bottleneck
        self.bottleneck = ResBlock(features[-1], features[-1] * 2)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        rev_features = list(reversed(features))
        ch = features[-1] * 2
        for f in rev_features:
            self.upconvs.append(nn.ConvTranspose2d(ch, f, 2, stride=2))
            self.decoders.append(ResBlock(f * 2, f))
            ch = f

        # Output head
        self.head = nn.Conv2d(features[0], num_classes, 1)

        # Deep supervision heads: one for each decoder stage except the last
        # (the last is the main head). Decoder stages go from deepest to
        # shallowest, so index 0 is deepest (lowest resolution).
        if deep_supervision:
            self.ds_heads = nn.ModuleList()
            for i, f in enumerate(rev_features[:-1]):
                self.ds_heads.append(nn.Conv2d(f, num_classes, 1))

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            If deep_supervision is False or model is in eval mode:
                Logits tensor of shape (B, num_classes, H, W).
            If deep_supervision is True and model is in training mode:
                Tuple of (main_logits, [aux_logits_1, ...]) where each
                auxiliary output is upsampled to the input spatial size.
        """
        input_size = x.shape[2:]

        # Encoder path
        skip_connections = []
        for encoder, pool, drop in zip(
            self.encoders, self.pools, self.enc_dropouts
        ):
            x = encoder(x)
            skip_connections.append(x)
            x = drop(pool(x))

        x = self.bottleneck(x)

        # Decoder path
        aux_outputs = []
        for i, (upconv, decoder, skip) in enumerate(
            zip(self.upconvs, self.decoders, reversed(skip_connections))
        ):
            x = upconv(x)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

            # Collect auxiliary outputs for deep supervision (all but last stage)
            if self.deep_supervision and self.training and i < len(self.decoders) - 1:
                aux = self.ds_heads[i](x)
                # Upsample auxiliary output to input resolution
                if aux.shape[2:] != input_size:
                    aux = nn.functional.interpolate(
                        aux, size=input_size, mode="bilinear", align_corners=False
                    )
                aux_outputs.append(aux)

        main_out = self.head(x)

        if self.deep_supervision and self.training and aux_outputs:
            return main_out, aux_outputs

        return main_out
