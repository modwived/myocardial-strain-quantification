"""Cardiac Motion Estimation Network (registration-based)."""

import torch
import torch.nn as nn


class MotionEncoder(nn.Module):
    """Shared encoder for source and target frames."""

    def __init__(self, in_ch: int = 1, features: tuple[int, ...] = (16, 32, 64, 128)):
        super().__init__()
        layers = []
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
        features = []
        for layer in self.encoder:
            x = layer(x)
            features.append(x)
        return features


class MotionDecoder(nn.Module):
    """Decoder that predicts a dense displacement field."""

    def __init__(self, features: tuple[int, ...] = (128, 64, 32, 16)):
        super().__init__()
        layers = []
        for i in range(len(features) - 1):
            # *2 because we concatenate source and target features
            in_ch = features[i] * 2 if i == 0 else features[i] + features[i]
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, features[i + 1], 4, stride=2, padding=1),
                    nn.InstanceNorm2d(features[i + 1]),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
        self.decoder = nn.ModuleList(layers)
        self.flow_head = nn.Conv2d(features[-1], 2, 3, padding=1)
        # Initialize flow head to zero for identity transform
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)

    def forward(self, source_features: list[torch.Tensor], target_features: list[torch.Tensor]) -> torch.Tensor:
        x = torch.cat([source_features[-1], target_features[-1]], dim=1)
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if i + 1 < len(self.decoder):
                skip_idx = -(i + 2)
                skip = torch.cat([source_features[skip_idx], target_features[skip_idx]], dim=1)
                x = x + skip
        return self.flow_head(x)


class CarMEN(nn.Module):
    """Cardiac Motion Estimation Network.

    Predicts a dense 2D displacement field from a pair of cardiac frames
    using an unsupervised registration approach.

    Args:
        in_channels: Number of input channels per frame.
        features: Channel counts for encoder/decoder stages.
    """

    def __init__(self, in_channels: int = 1, features: tuple[int, ...] = (16, 32, 64, 128)):
        super().__init__()
        self.encoder = MotionEncoder(in_channels, features)
        self.decoder = MotionDecoder(features[::-1])

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Predict displacement field from source to target.

        Args:
            source: (B, 1, H, W) source frame.
            target: (B, 1, H, W) target frame.

        Returns:
            (B, 2, H, W) displacement field in pixels.
        """
        src_features = self.encoder(source)
        tgt_features = self.encoder(target)
        displacement = self.decoder(src_features, tgt_features)
        # Upsample to input resolution
        displacement = nn.functional.interpolate(
            displacement, size=source.shape[2:], mode="bilinear", align_corners=True
        )
        return displacement
