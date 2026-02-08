"""Spatial transformer for differentiable image warping."""

import torch
import torch.nn.functional as F


def spatial_transform(source: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
    """Warp source image using a displacement field.

    Args:
        source: (B, C, H, W) source image.
        displacement: (B, 2, H, W) displacement field in pixels.

    Returns:
        (B, C, H, W) warped image.
    """
    B, _, H, W = source.shape

    # Create identity grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=source.device, dtype=source.dtype),
        torch.arange(W, device=source.device, dtype=source.dtype),
        indexing="ij",
    )
    grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)

    # Add displacement and normalize to [-1, 1]
    new_grid = grid + displacement
    new_grid[:, 0] = 2.0 * new_grid[:, 0] / (W - 1) - 1.0
    new_grid[:, 1] = 2.0 * new_grid[:, 1] / (H - 1) - 1.0

    # grid_sample expects (B, H, W, 2) with (x, y) order
    new_grid = new_grid.permute(0, 2, 3, 1)

    return F.grid_sample(source, new_grid, mode="bilinear", padding_mode="border", align_corners=True)
