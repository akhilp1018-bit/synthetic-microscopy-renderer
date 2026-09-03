"""
visualize_dataset_overlay.py
----------------------------

Create contact-sheet previews for the generated synthetic microscopy dataset.

Expected dataset structure:

    synthetic_dataset_v1/
    ├── train/
    │   ├── instance_000001/
    │   └── ...
    ├── validation/
    │   ├── instance_000001/
    │   └── ...
    └── test/
        ├── instance_000001/
        └── ...

For each instance folder, the script reads:

    - noisy.tif
    - dendrite_mask.tif
    - spine_mask.tif

It creates:

    - review_image_mips.png
    - review_overlay_mips.png
    - review_mask_mips.png
    - review_index.csv

Overlay colors:

    - dendrite: cyan-blue
    - spine: orange

The microscopy image is shown in grayscale.

Run from repository root:

    PYTHONPATH=. python scripts/visualize_dataset_overlay.py \
        --output-dir outputs/synthetic_dataset_v1
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw


# ==========================================================
# DEFAULT SETTINGS
# ==========================================================

DEFAULT_DATASET_ROOT = "outputs/synthetic_dataset_v1"

IMAGE_FILENAME = "noisy.tif"
DENDRITE_MASK_FILENAME = "dendrite_mask.tif"
SPINE_MASK_FILENAME = "spine_mask.tif"

TILE_SIZE = 128
TILE_PAD = 4
LABEL_HEIGHT = 16

DENDRITE_ALPHA = 0.65
SPINE_ALPHA = 0.65


# ==========================================================
# HELPERS
# ==========================================================

def normalize_to_uint8(arr):
    """
    Normalize an array to uint8 range 0-255.
    """

    arr = arr.astype(np.float32)

    vmax = float(arr.max())

    if vmax <= 0:
        return np.zeros_like(
            arr,
            dtype=np.uint8,
        )

    arr = arr / vmax
    arr = np.clip(arr, 0.0, 1.0)

    return (
        arr * 255
    ).astype(np.uint8)


def read_instance(instance_dir):
    """
    Read one generated dataset instance.

    Returns
    -------
    image_u8
        Grayscale MIP of noisy.tif.

    dendrite_mip
        Binary MIP of dendrite mask.

    spine_mip
        Binary MIP of spine mask.
    """

    image_path = (
        instance_dir
        / IMAGE_FILENAME
    )

    dendrite_path = (
        instance_dir
        / DENDRITE_MASK_FILENAME
    )

    spine_path = (
        instance_dir
        / SPINE_MASK_FILENAME
    )

    if not (
        image_path.exists()
        and dendrite_path.exists()
        and spine_path.exists()
    ):
        return None

    image = tifffile.imread(
        image_path
    )

    dendrite = tifffile.imread(
        dendrite_path
    )

    spine = tifffile.imread(
        spine_path
    )

    if image.ndim != 3:
        raise ValueError(
            f"{image_path} is not a 3D ZYX stack. "
            f"Shape: {image.shape}"
        )

    if dendrite.shape != image.shape:
        raise ValueError(
            f"Shape mismatch in {instance_dir}: "
            f"image={image.shape}, "
            f"dendrite={dendrite.shape}"
        )

    if spine.shape != image.shape:
        raise ValueError(
            f"Shape mismatch in {instance_dir}: "
            f"image={image.shape}, "
            f"spine={spine.shape}"
        )

    # ZYX -> XY MIP
    image_mip = (
        image.max(axis=0)
    )

    dendrite_mip = (
        (dendrite > 0)
        .max(axis=0)
    )

    spine_mip = (
        (spine > 0)
        .max(axis=0)
    )

    image_u8 = normalize_to_uint8(
        image_mip
    )

    return (
        image_u8,
        dendrite_mip,
        spine_mip,
    )


def make_image_tile(image_u8):
    """
    Convert grayscale MIP to RGB.
    """

    return np.stack(
        [
            image_u8,
            image_u8,
            image_u8,
        ],
        axis=-1,
    )


def make_overlay_tile(
    image_u8,
    dendrite_mip,
    spine_mip,
):
    """
    Add cyan-blue dendrite and orange spine masks
    to the grayscale microscopy image.
    """

    base = np.stack(
        [
            image_u8,
            image_u8,
            image_u8,
        ],
        axis=-1,
    ).astype(np.float32)

    dendrite_pixels = (
        dendrite_mip.astype(bool)
    )

    spine_pixels = (
        spine_mip.astype(bool)
    )

    # Cyan-blue
    dendrite_color = np.array(
        [
            0.0,
            0.7,
            1.0,
        ],
        dtype=np.float32,
    ) * 255.0

    # Orange
    spine_color = np.array(
        [
            1.0,
            0.6,
            0.0,
        ],
        dtype=np.float32,
    ) * 255.0

    # Dendrite overlay
    base[dendrite_pixels] = (
        (1.0 - DENDRITE_ALPHA)
        * base[dendrite_pixels]
        + DENDRITE_ALPHA
        * dendrite_color
    )

    # Spine overlay applied second
    # so spine remains visible on overlap.
    base[spine_pixels] = (
        (1.0 - SPINE_ALPHA)
        * base[spine_pixels]
        + SPINE_ALPHA
        * spine_color
    )

    return np.clip(
        base,
        0,
        255,
    ).astype(np.uint8)


def make_mask_tile(
    dendrite_mip,
    spine_mip,
):
    """
    Create mask-only RGB visualization.

    dendrite = cyan-blue
    spine    = orange
    """

    height, width = (
        dendrite_mip.shape
    )

    rgb = np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    dendrite_pixels = (
        dendrite_mip.astype(bool)
    )

    spine_pixels = (
        spine_mip.astype(bool)
    )

    rgb[
        dendrite_pixels
    ] = np.array(
        [0, 179, 255],
        dtype=np.uint8,
    )

    rgb[
        spine_pixels
    ] = np.array(
        [255, 153, 0],
        dtype=np.uint8,
    )

    return rgb


def resize_tile(
    tile_rgb,
    tile_size=TILE_SIZE,
):
    """
    Resize one RGB tile.
    """

    image = Image.fromarray(
        tile_rgb
    )

    if image.size != (
        tile_size,
        tile_size,
    ):
        image = image.resize(
            (
                tile_size,
                tile_size,
            ),
            resample=Image.Resampling.NEAREST,
        )

    return np.asarray(
        image
    )


def add_label(
    tile_rgb,
    label,
):
    """
    Add split/instance label above tile.
    """

    height, width, _ = (
        tile_rgb.shape
    )

    output = np.zeros(
        (
            height + LABEL_HEIGHT,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    output[
        LABEL_HEIGHT:,
        :,
        :,
    ] = tile_rgb

    image = Image.fromarray(
        output
    )

    draw = ImageDraw.Draw(
        image
    )

    draw.text(
        (3, 2),
        label,
        fill=(
            255,
            255,
            255,
        ),
    )

    return np.asarray(
        image
    )


def build_contact_sheet(
    tile_list,
    grid_cols,
    pad=TILE_PAD,
):
    """
    Arrange all instance tiles into one image.
    """

    if not tile_list:
        raise RuntimeError(
            "No tiles available."
        )

    tile_height, tile_width, _ = (
        tile_list[0].shape
    )

    count = len(
        tile_list
    )

    grid_cols = min(
        grid_cols,
        count,
    )

    rows = math.ceil(
        count / grid_cols
    )

    sheet_height = (
        rows * tile_height
        + (rows + 1) * pad
    )

    sheet_width = (
        grid_cols * tile_width
        + (grid_cols + 1) * pad
    )

    sheet = np.zeros(
        (
            sheet_height,
            sheet_width,
            3,
        ),
        dtype=np.uint8,
    )

    for index, tile in enumerate(
        tile_list
    ):

        row = (
            index // grid_cols
        )

        column = (
            index % grid_cols
        )

        y0 = (
            pad
            + row
            * (tile_height + pad)
        )

        x0 = (
            pad
            + column
            * (tile_width + pad)
        )

        sheet[
            y0:y0 + tile_height,
            x0:x0 + tile_width,
            :,
        ] = tile

    return sheet


def save_review_csv(
    rows,
    output_path,
):
    """
    Save simple review statistics.
    """

    if not rows:
        return

    fieldnames = [
        "index",
        "split",
        "instance",
        "path",
        "image_nonzero",
        "dendrite_mip_pixels",
        "spine_mip_pixels",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ==========================================================
# MAIN
# ==========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create MIP contact-sheet previews "
            "for the synthetic training dataset."
        )
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_DATASET_ROOT,
        help=(
            "Dataset root containing train, "
            "validation and test folders."
        ),
    )

    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help=(
            "Maximum number of instances to review. "
            "Default: all."
        ),
    )

    parser.add_argument(
        "--grid-cols",
        type=int,
        default=None,
        help=(
            "Number of contact-sheet columns. "
            "Default: automatic."
        ),
    )

    args = parser.parse_args()

    dataset_root = Path(
        args.output_dir
    )

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset not found: "
            f"{dataset_root}"
        )

    output_dir = (
        dataset_root
        / "review_mips"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # Find instances inside train/validation/test
    # ------------------------------------------------------

    valid_splits = (
        "train",
        "validation",
        "test",
    )

    instance_dirs = []

    for split in valid_splits:

        split_dir = (
            dataset_root
            / split
        )

        if not split_dir.exists():
            continue

        split_instances = sorted(
            path
            for path
            in split_dir.glob(
                "instance_*"
            )
            if path.is_dir()
        )

        instance_dirs.extend(
            split_instances
        )

    if args.max_instances is not None:
        instance_dirs = (
            instance_dirs[
                :args.max_instances
            ]
        )

    if not instance_dirs:
        raise RuntimeError(
            f"No instance_* folders found "
            f"inside train/validation/test "
            f"under {dataset_root}"
        )

    # ------------------------------------------------------
    # Automatic layout
    # ------------------------------------------------------

    if args.grid_cols is None:

        if len(instance_dirs) <= 5:
            grid_cols = (
                len(instance_dirs)
            )

        elif len(instance_dirs) <= 100:
            grid_cols = 10

        else:
            grid_cols = 20

    else:
        grid_cols = (
            args.grid_cols
        )

    print("=" * 70)
    print("Synthetic Dataset Overlay Review")
    print("=" * 70)

    print(
        f"Dataset root : "
        f"{dataset_root}"
    )

    print(
        f"Instances    : "
        f"{len(instance_dirs)}"
    )

    print(
        f"Grid columns : "
        f"{grid_cols}"
    )

    print(
        f"Output       : "
        f"{output_dir}"
    )

    print("=" * 70)

    image_tiles = []
    overlay_tiles = []
    mask_tiles = []

    review_rows = []

    for index, instance_dir in enumerate(
        instance_dirs,
        start=1,
    ):

        split_name = (
            instance_dir.parent.name
        )

        instance_name = (
            instance_dir.name
        )

        display_label = (
            f"{split_name}/"
            f"{instance_name}"
        )

        data = read_instance(
            instance_dir
        )

        if data is None:
            print(
                f"Skipping incomplete instance: "
                f"{display_label}"
            )
            continue

        (
            image_u8,
            dendrite_mip,
            spine_mip,
        ) = data

        image_tile = make_image_tile(
            image_u8
        )

        overlay_tile = make_overlay_tile(
            image_u8,
            dendrite_mip,
            spine_mip,
        )

        mask_tile = make_mask_tile(
            dendrite_mip,
            spine_mip,
        )

        image_tile = resize_tile(
            image_tile
        )

        overlay_tile = resize_tile(
            overlay_tile
        )

        mask_tile = resize_tile(
            mask_tile
        )

        image_tile = add_label(
            image_tile,
            display_label,
        )

        overlay_tile = add_label(
            overlay_tile,
            display_label,
        )

        mask_tile = add_label(
            mask_tile,
            display_label,
        )

        image_tiles.append(
            image_tile
        )

        overlay_tiles.append(
            overlay_tile
        )

        mask_tiles.append(
            mask_tile
        )

        review_rows.append({
            "index":
                index,

            "split":
                split_name,

            "instance":
                instance_name,

            "path":
                str(instance_dir),

            "image_nonzero":
                int(
                    np.count_nonzero(
                        image_u8
                    )
                ),

            "dendrite_mip_pixels":
                int(
                    np.count_nonzero(
                        dendrite_mip
                    )
                ),

            "spine_mip_pixels":
                int(
                    np.count_nonzero(
                        spine_mip
                    )
                ),
        })

        if (
            index % 50 == 0
            or index == len(instance_dirs)
        ):
            print(
                f"Processed "
                f"{index}/"
                f"{len(instance_dirs)}"
            )

    if not image_tiles:
        raise RuntimeError(
            "No complete instances found."
        )

    print()
    print(
        "Building contact sheets..."
    )

    image_sheet = build_contact_sheet(
        image_tiles,
        grid_cols,
    )

    overlay_sheet = build_contact_sheet(
        overlay_tiles,
        grid_cols,
    )

    mask_sheet = build_contact_sheet(
        mask_tiles,
        grid_cols,
    )

    image_output = (
        output_dir
        / "review_image_mips.png"
    )

    overlay_output = (
        output_dir
        / "review_overlay_mips.png"
    )

    mask_output = (
        output_dir
        / "review_mask_mips.png"
    )

    csv_output = (
        output_dir
        / "review_index.csv"
    )

    Image.fromarray(
        image_sheet
    ).save(
        image_output
    )

    Image.fromarray(
        overlay_sheet
    ).save(
        overlay_output
    )

    Image.fromarray(
        mask_sheet
    ).save(
        mask_output
    )

    save_review_csv(
        review_rows,
        csv_output,
    )

    print()
    print("Saved:")

    print(
        image_output
    )

    print(
        overlay_output
    )

    print(
        mask_output
    )

    print(
        csv_output
    )

    print()
    print("Legend:")

    print(
        "Grayscale = noisy microscopy image"
    )

    print(
        "Cyan-blue = dendrite GT mask"
    )

    print(
        "Orange    = spine GT mask"
    )

    print()
    print(
        f"Reviewed "
        f"{len(image_tiles)} "
        f"complete instances."
    )

    print("Done.")


if __name__ == "__main__":
    main()