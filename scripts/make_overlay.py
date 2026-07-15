"""
Create quick preview images for rendered TIFF stacks and masks.

This script makes maximum-intensity-projection previews for:
    - rendered image
    - spine mask
    - dendrite mask
    - image + mask overlay

Run from repository root:

    PYTHONPATH=. python scripts/make_overlay.py --output-dir outputs/sample_001/gaussian_2p_voxelgrid_membrane
"""

from pathlib import Path
import argparse

import numpy as np
import tifffile
import matplotlib.pyplot as plt


def normalize01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    vmax = float(arr.max())

    if vmax <= 0:
        return np.zeros_like(arr, dtype=np.float32)

    return arr / vmax


def save_gray(path: Path, image: np.ndarray) -> None:
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_overlay(path: Path, image_mip: np.ndarray, spine_mip: np.ndarray, dendrite_mip: np.ndarray) -> None:
    image = normalize01(image_mip)
    spine = spine_mip > 0
    dendrite = dendrite_mip > 0

    rgb = np.stack([image, image, image], axis=-1)

    # Dendrite overlay: green
    rgb[dendrite, 0] = 0.0
    rgb[dendrite, 1] = 1.0
    rgb[dendrite, 2] = 0.0

    # Spine overlay: red
    rgb[spine, 0] = 1.0
    rgb[spine, 1] = 0.0
    rgb[spine, 2] = 0.0

    plt.figure(figsize=(8, 8))
    plt.imshow(rgb)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


def find_one(output_dir: Path, pattern: str) -> Path:
    matches = sorted(output_dir.glob(pattern))

    if len(matches) == 0:
        raise FileNotFoundError(f"No file found for pattern: {pattern}")

    if len(matches) > 1:
        print(f"Warning: multiple files found for {pattern}. Using:")
        print(f"  {matches[0]}")

    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, help="Renderer output folder")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    image_path = find_one(output_dir, "*image.tif")
    spine_path = find_one(output_dir, "*spine_mask.tif")
    dendrite_path = find_one(output_dir, "*dendrite_mask.tif")

    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    print("Loading:")
    print(f"  image   : {image_path}")
    print(f"  spine   : {spine_path}")
    print(f"  dendrite: {dendrite_path}")

    image = tifffile.imread(image_path)
    spine = tifffile.imread(spine_path)
    dendrite = tifffile.imread(dendrite_path)

    image_mip = image.max(axis=0)
    spine_mip = spine.max(axis=0)
    dendrite_mip = dendrite.max(axis=0)

    save_gray(preview_dir / "image_mip.png", normalize01(image_mip))
    save_gray(preview_dir / "spine_mask_mip.png", spine_mip > 0)
    save_gray(preview_dir / "dendrite_mask_mip.png", dendrite_mip > 0)
    save_overlay(preview_dir / "overlay_mip.png", image_mip, spine_mip, dendrite_mip)

    print("Saved previews:")
    print(f"  {preview_dir / 'image_mip.png'}")
    print(f"  {preview_dir / 'spine_mask_mip.png'}")
    print(f"  {preview_dir / 'dendrite_mask_mip.png'}")
    print(f"  {preview_dir / 'overlay_mip.png'}")


if __name__ == "__main__":
    main()