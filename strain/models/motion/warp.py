"""Spatial transformer for differentiable image warping.

Provides:
* ``spatial_transform`` -- warp an image (or feature map) with a displacement field
* ``compose_displacements`` -- compose two displacement fields for multi-step tracking
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _make_identity_grid(
    H: int,
    W: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return an identity sampling grid of shape (1, 2, H, W).

    Channel 0 = x (column) coordinates, channel 1 = y (row) coordinates,
    both in **pixel** units (0-indexed).
    """
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # (1, 2, H, W)


def _pixel_to_norm(grid: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Convert pixel-unit grid (B, 2, H, W) to [-1, 1] normalised grid (B, H, W, 2)."""
    out = grid.clone()
    out[:, 0] = 2.0 * out[:, 0] / (W - 1) - 1.0
    out[:, 1] = 2.0 * out[:, 1] / (H - 1) - 1.0
    return out.permute(0, 2, 3, 1)  # (B, H, W, 2)


# ---------------------------------------------------------------------------
# Core warping
# ---------------------------------------------------------------------------

def spatial_transform(source: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
    """Warp source image using a displacement field.

    Parameters
    ----------
    source : (B, C, H, W)
        Source image or feature map.
    displacement : (B, 2, H, W)
        Displacement field in **pixels**.
        Channel 0 = dx (horizontal), channel 1 = dy (vertical).

    Returns
    -------
    (B, C, H, W)  warped image.
    """
    B, _, H, W = source.shape

    grid = _make_identity_grid(H, W, source.device, source.dtype).expand(B, -1, -1, -1)
    new_grid = grid + displacement
    new_grid_norm = _pixel_to_norm(new_grid, H, W)

    return F.grid_sample(
        source,
        new_grid_norm,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


# ---------------------------------------------------------------------------
# Displacement composition
# ---------------------------------------------------------------------------

def compose_displacements(
    d1: torch.Tensor,
    d2: torch.Tensor,
) -> torch.Tensor:
    """Compose two displacement fields: d_total(x) = d1(x) + d2(x + d1(x)).

    This gives the total displacement that first applies *d1* then *d2*:
        phi_total(x) = phi_2(phi_1(x))
    where phi(x) = x + d(x).

    The second field *d2* is sampled at the locations displaced by *d1*
    (i.e. ``spatial_transform`` applied to *d2* itself using *d1*).

    Parameters
    ----------
    d1 : (B, 2, H, W)
        First displacement field (applied first).
    d2 : (B, 2, H, W)
        Second displacement field (applied second, at the warped locations).

    Returns
    -------
    (B, 2, H, W)  composed displacement field.
    """
    # Sample d2 at the locations displaced by d1
    d2_warped = spatial_transform(d2, d1)
    return d1 + d2_warped
