"""Loss functions for unsupervised motion estimation."""

import torch
import torch.nn as nn


class NCC(nn.Module):
    """Local Normalized Cross-Correlation loss."""

    def __init__(self, window_size: int = 9):
        super().__init__()
        self.window_size = window_size

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute NCC loss (1 - NCC so lower is better).

        Args:
            predicted: (B, 1, H, W) warped source image.
            target: (B, 1, H, W) target image.

        Returns:
            Scalar NCC loss.
        """
        w = self.window_size
        pad = w // 2

        # Local sums using average pooling
        I = predicted
        J = target
        I2 = I * I
        J2 = J * J
        IJ = I * J

        pool = nn.AvgPool2d(w, stride=1, padding=pad)
        I_mean = pool(I)
        J_mean = pool(J)
        I2_mean = pool(I2)
        J2_mean = pool(J2)
        IJ_mean = pool(IJ)

        cross = IJ_mean - I_mean * J_mean
        I_var = I2_mean - I_mean * I_mean
        J_var = J2_mean - J_mean * J_mean

        cc = (cross * cross + 1e-5) / (I_var * J_var + 1e-5)
        return 1.0 - cc.mean()


class DiffusionRegularizer(nn.Module):
    """Diffusion regularization on displacement field gradients."""

    def forward(self, displacement: torch.Tensor) -> torch.Tensor:
        """Compute regularization loss.

        Args:
            displacement: (B, 2, H, W) displacement field.

        Returns:
            Scalar regularization loss.
        """
        dy = displacement[:, :, 1:, :] - displacement[:, :, :-1, :]
        dx = displacement[:, :, :, 1:] - displacement[:, :, :, :-1]
        return (dx.pow(2).mean() + dy.pow(2).mean()) / 2.0
