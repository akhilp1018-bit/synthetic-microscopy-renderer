"""
Input/output helper functions for the synthetic microscopy renderer.

This file contains small utilities for:
    - creating output folders
    - converting Torch volumes to uint8 or uint16 TIFF stacks
    - converting binary masks to uint8 or uint16 TIFF masks
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


# ============================================================
# Image conversion
# ============================================================

def tensor_to_u8_stack(vol: torch.Tensor) -> np.ndarray:
    """
    Convert a rendered Torch volume to a uint8 NumPy stack.

    The volume is normalized to [0, 1] using its maximum value
    and then scaled to uint8 range [0, 255].

    Args:
        vol:
            Floating-point Torch tensor in ZYX order.

    Returns:
        NumPy uint8 array in ZYX order.
    """
    vol_np = (
        vol.detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )

    vmax = float(vol_np.max())

    if vmax > 0:
        vol_np = vol_np / vmax

    np.clip(
        vol_np,
        0.0,
        1.0,
        out=vol_np,
    )

    return (
        vol_np * 255.0
    ).astype(np.uint8)


def tensor_to_u16_stack(vol: torch.Tensor) -> np.ndarray:
    """
    Convert a rendered Torch volume to a uint16 NumPy stack.

    The volume is normalized to [0, 1] using its maximum value
    and then scaled to uint16 range [0, 65535].

    Args:
        vol:
            Floating-point Torch tensor in ZYX order.

    Returns:
        NumPy uint16 array in ZYX order.
    """
    vol_np = (
        vol.detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )

    vmax = float(vol_np.max())

    if vmax > 0:
        vol_np = vol_np / vmax

    np.clip(
        vol_np,
        0.0,
        1.0,
        out=vol_np,
    )

    return (
        vol_np * 65535.0
    ).astype(np.uint16)


def tensor_to_stack(
    vol: torch.Tensor,
    bit_depth: int = 16,
) -> np.ndarray:
    """
    Convert a rendered Torch volume to the requested bit depth.

    Supported bit depths:
        8  -> uint8
        16 -> uint16
    """
    if bit_depth == 8:
        return tensor_to_u8_stack(vol)

    if bit_depth == 16:
        return tensor_to_u16_stack(vol)

    raise ValueError(
        f"Unsupported image bit depth: {bit_depth}. "
        "Supported values are 8 and 16."
    )


# ============================================================
# Mask conversion
# ============================================================

def binary_mask_to_u8(
    mask: torch.Tensor,
) -> np.ndarray:
    """
    Convert a binary Torch mask to uint8.

    Background = 0
    Foreground = 255
    """
    mask_np = (
        mask.detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )

    return (
        (mask_np > 0).astype(np.uint8)
        * 255
    )


def binary_mask_to_u16(
    mask: torch.Tensor,
) -> np.ndarray:
    """
    Convert a binary Torch mask to uint16.

    Background = 0
    Foreground = 65535
    """
    mask_np = (
        mask.detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )

    return (
        (mask_np > 0).astype(np.uint16)
        * 65535
    )


def binary_mask_to_stack(
    mask: torch.Tensor,
    bit_depth: int = 16,
) -> np.ndarray:
    """
    Convert a binary mask to the requested bit depth.

    Supported bit depths:
        8  -> 0 / 255
        16 -> 0 / 65535
    """
    if bit_depth == 8:
        return binary_mask_to_u8(mask)

    if bit_depth == 16:
        return binary_mask_to_u16(mask)

    raise ValueError(
        f"Unsupported mask bit depth: {bit_depth}. "
        "Supported values are 8 and 16."
    )


# ============================================================
# TIFF saving
# ============================================================

def save_stack_imagej_zyx(
    out_dir: str | Path,
    filename: str,
    stack_zyx: np.ndarray,
    xy_um_per_px: float,
    z_step_um: float,
) -> str:
    """
    Save a ZYX stack as an ImageJ-compatible TIFF.

    Works with both uint8 and uint16 arrays.

    Args:
        out_dir:
            Output directory.

        filename:
            Output TIFF filename.

        stack_zyx:
            NumPy image stack in ZYX order.

        xy_um_per_px:
            XY sampling in micrometres per pixel.

        z_step_um:
            Z spacing in micrometres.

    Returns:
        Path to saved TIFF file.
    """
    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    tiff_path = os.path.join(
        out_dir,
        filename,
    )

    tifffile.imwrite(
        tiff_path,
        stack_zyx,
        imagej=True,
        compression="zlib",
        resolution=(
            1.0 / xy_um_per_px,
            1.0 / xy_um_per_px,
        ),
        metadata={
            "axes": "ZYX",
            "spacing": z_step_um,
            "unit": "um",
        },
    )

    return tiff_path


def save_stack_imagej_zyx_u16(
    out_dir: str | Path,
    tag: str,
    stack_u16_zyx: np.ndarray,
    xy_um_per_px: float,
    z_step_um: float,
) -> str:
    """
    Legacy uint16 TIFF saver.

    Kept so existing render.py code continues to work.
    """
    return save_stack_imagej_zyx(
        out_dir=out_dir,
        filename=f"zstack_{tag}.tif",
        stack_zyx=stack_u16_zyx,
        xy_um_per_px=xy_um_per_px,
        z_step_um=z_step_um,
    )


def save_stack_imagej_zyx_u8(
    out_dir: str | Path,
    tag: str,
    stack_u8_zyx: np.ndarray,
    xy_um_per_px: float,
    z_step_um: float,
) -> str:
    """
    Save a uint8 ZYX TIFF stack.

    Filename follows the existing renderer naming style:
        zstack_<tag>.tif
    """
    return save_stack_imagej_zyx(
        out_dir=out_dir,
        filename=f"zstack_{tag}.tif",
        stack_zyx=stack_u8_zyx,
        xy_um_per_px=xy_um_per_px,
        z_step_um=z_step_um,
    )


# ============================================================
# Metadata
# ============================================================

def save_run_metadata_txt(
    out_dir: str | Path,
    tag: str,
    lines: list[str],
) -> str:
    """
    Save simple line-based metadata as a text file.

    Kept for compatibility and debugging.
    JSON metadata is preferred for normal runs.
    """
    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    meta_path = os.path.join(
        out_dir,
        f"metadata_{tag}.txt",
    )

    with open(
        meta_path,
        "w",
        encoding="utf-8",
    ) as f:

        for line in lines:
            f.write(
                line.rstrip()
                + "\n"
            )

    return meta_path


def save_metadata_json(
    out_dir: str | Path,
    tag: str,
    metadata: dict,
) -> str:
    """
    Save metadata as JSON.
    """
    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    meta_path = os.path.join(
        out_dir,
        f"metadata_{tag}.json",
    )

    with open(
        meta_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    return meta_path