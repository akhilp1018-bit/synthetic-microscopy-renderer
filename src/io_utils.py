"""
Input/output helper functions for the synthetic microscopy renderer.

This file contains small utilities for:
    - creating output folders
    - converting Torch volumes to uint16 TIFF stacks
    - converting binary masks to uint16 TIFF masks
    - saving ImageJ-compatible TIFF stacks
    - saving metadata files

Coordinate convention:
    All image and mask stacks are saved in ZYX order:
    [Z slices, Y pixels, X pixels].
"""

import os
import json
from pathlib import Path

import numpy as np
import tifffile
import torch


def ensure_dir(path: str | Path) -> None:
    """
    Create a directory if it does not already exist.
    """
    os.makedirs(path, exist_ok=True)


def tensor_to_u16_stack(vol: torch.Tensor) -> np.ndarray:
    """
    Convert a rendered Torch volume to a uint16 NumPy stack.

    The rendered volume is expected to be a floating-point tensor in ZYX order.
    It is normalized to the range [0, 1] using its maximum value and then scaled
    to uint16 range [0, 65535].

    Args:
        vol:
            Torch tensor containing the rendered image volume.

    Returns:
        NumPy array in uint16 format, ZYX order.
    """
    vol_np = vol.detach().cpu().numpy().astype(np.float32, copy=False)

    vmax = float(vol_np.max())
    if vmax > 0:
        vol_np = vol_np / vmax

    np.clip(vol_np, 0.0, 1.0, out=vol_np)

    return (vol_np * 65535.0).astype(np.uint16)


def binary_mask_to_u16(mask: torch.Tensor) -> np.ndarray:
    """
    Convert a binary Torch mask to a uint16 NumPy mask.

    Background is saved as 0.
    Foreground is saved as 65535.

    Args:
        mask:
            Torch tensor mask in ZYX order.

    Returns:
        NumPy uint16 mask in ZYX order.
    """
    mask_np = mask.detach().cpu().numpy().astype(np.float32, copy=False)
    return ((mask_np > 0).astype(np.uint16) * 65535)


def save_stack_imagej_zyx_u16(
    out_dir: str | Path,
    tag: str,
    stack_u16_zyx: np.ndarray,
    xy_um_per_px: float,
    z_step_um: float,
) -> str:
    """
    Save a ZYX uint16 stack as an ImageJ-compatible TIFF file.

    The TIFF metadata stores:
        - axes: ZYX
        - XY pixel size in micrometres
        - Z spacing in micrometres

    Args:
        out_dir:
            Output folder.
        tag:
            Name tag used to create the output filename.
        stack_u16_zyx:
            Image or mask stack in uint16 ZYX order.
        xy_um_per_px:
            XY pixel size in micrometres per pixel.
        z_step_um:
            Z spacing in micrometres.

    Returns:
        Path to the saved TIFF file.
    """
    os.makedirs(out_dir, exist_ok=True)

    tiff_path = os.path.join(out_dir, f"zstack_{tag}.tif")

    tifffile.imwrite(
        tiff_path,
        stack_u16_zyx,
        imagej=True,
        compression="zlib",
        resolution=(1.0 / xy_um_per_px, 1.0 / xy_um_per_px),
        metadata={"axes": "ZYX", "spacing": z_step_um, "unit": "um"},
    )

    return tiff_path


def save_run_metadata_txt(out_dir: str | Path, tag: str, lines: list[str]) -> str:
    """
    Save simple line-based metadata as a text file.

    This is kept for compatibility/simple debugging. For normal renderer runs,
    JSON metadata is preferred.
    """
    os.makedirs(out_dir, exist_ok=True)

    meta_path = os.path.join(out_dir, f"metadata_{tag}.txt")

    with open(meta_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")

    return meta_path


def save_metadata_json(out_dir: str | Path, tag: str, metadata: dict) -> str:
    """
    Save run metadata as a JSON file.

    Metadata should include enough information to understand how the image was
    produced, for example renderer type, PSF mode, output shape, paths, and
    input mode.
    """
    os.makedirs(out_dir, exist_ok=True)

    meta_path = os.path.join(out_dir, f"metadata_{tag}.json")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return meta_path