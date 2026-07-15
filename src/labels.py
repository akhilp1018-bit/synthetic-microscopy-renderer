"""
Label and mask helper functions.

This module creates binary masks from rendered image volumes.

Coordinate convention:
    Input and output tensors are in ZYX order:
    [Z slices, Y pixels, X pixels].
"""

import torch


def make_object_mask_from_volume(
    vol: torch.Tensor,
    rel_threshold: float = 0.1,
) -> torch.Tensor:
    """
    Create a binary object mask from a rendered volume.

    The threshold is relative to the maximum intensity of the volume.

    Example:
        rel_threshold = 0.1 means:
        threshold = 10% of max(volume)

    Args:
        vol:
            Rendered image volume as a Torch tensor in ZYX order.
        rel_threshold:
            Relative threshold used to separate foreground from background.

    Returns:
        Binary mask as a float32 Torch tensor:
            1.0 = foreground/object
            0.0 = background
    """
    vmax = float(vol.max().item())

    if vmax <= 0:
        return torch.zeros_like(vol, dtype=torch.float32)

    threshold = rel_threshold * vmax

    return (vol > threshold).to(torch.float32)