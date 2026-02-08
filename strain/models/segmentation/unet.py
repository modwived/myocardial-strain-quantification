"""2D U-Net with residual encoder for cardiac segmentation."""

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
        num_classes: Number of output classes (4: bg, LV, myo, RV).
        features: Channel counts for each encoder stage.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        features: tuple[int, ...] = (64, 128, 256, 512),
    ):
        super().__init__()

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(ResBlock(ch, f))
            self.pools.append(nn.MaxPool2d(2))
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

        # Output
        self.head = nn.Conv2d(features[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        skip_connections = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        # Decoder path
        for upconv, decoder, skip in zip(
            self.upconvs, self.decoders, reversed(skip_connections)
        ):
            x = upconv(x)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        return self.head(x)
