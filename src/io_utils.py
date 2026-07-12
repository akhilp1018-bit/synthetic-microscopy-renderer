import os
import json
from pathlib import Path

import numpy as np
import tifffile
import torch


def ensure_dir(path: str | Path) -> None:
    os.makedirs(path, exist_ok=True)


def tensor_to_u16_stack(vol: torch.Tensor) -> np.ndarray:
    vol_np = vol.detach().cpu().numpy().astype(np.float32, copy=False)

    vmax = float(vol_np.max())
    if vmax > 0:
        vol_np = vol_np / vmax

    np.clip(vol_np, 0.0, 1.0, out=vol_np)

    return (vol_np * 65535.0).astype(np.uint16)


def binary_mask_to_u16(mask: torch.Tensor) -> np.ndarray:
    mask_np = mask.detach().cpu().numpy().astype(np.float32, copy=False)
    return ((mask_np > 0).astype(np.uint16) * 65535)


def save_stack_imagej_zyx_u16(
    out_dir: str | Path,
    tag: str,
    stack_u16_zyx: np.ndarray,
    xy_um_per_px: float,
    z_step_um: float,
) -> str:
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
    os.makedirs(out_dir, exist_ok=True)

    meta_path = os.path.join(out_dir, f"metadata_{tag}.txt")

    with open(meta_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")

    return meta_path


def save_metadata_json(out_dir: str | Path, tag: str, metadata: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)

    meta_path = os.path.join(out_dir, f"metadata_{tag}.json")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return meta_path