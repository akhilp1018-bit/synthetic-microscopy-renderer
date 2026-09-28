"""
Create preview images for labelled-mesh rendering experiments.

Run this script only after the rendering pipeline has completed successfully
using render.py.

Required renderer outputs:

    - rendered image TIFF ending with:
          image.tif

    - spine ground-truth mask ending with:
          spine_mask.tif

    - dendrite ground-truth mask ending with:
          dendrite_mask.tif

The script creates:

    - maximum-intensity projection of the rendered image
    - maximum-intensity projection of the spine mask
    - maximum-intensity projection of the dendrite mask
    - maximum-intensity projection with dendrite and spine GT overlay
    - full 3D RGB TIFF overlay
    - horizontal four-panel summary figure

Overlay colors:

    - dendrite: cyan-blue
    - spine: orange

Run from the repository root:

    python scripts/make_render_overlay.py \
        --output-dir outputs/sample_001/gaussian_2p_voxelgrid_membrane

Generated files:

    previews/image_mip.png
    previews/spine_mask_mip.png
    previews/dendrite_mask_mip.png
    previews/overlay_mip.png
    previews/overlay_stack.tif
    previews/labelled_render_summary.png
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import tifffile


# Overlay colors.
DENDRITE_COLOR = np.array(
    [0.0, 0.7, 1.0],
    dtype=np.float32,
)

SPINE_COLOR = np.array(
    [1.0, 0.6, 0.0],
    dtype=np.float32,
)

OVERLAY_ALPHA = 0.65


def normalize01(arr: np.ndarray) -> np.ndarray:
    """
    Normalize an array to the range [0, 1].

    If the array contains only zero or negative values, return an
    all-zero float32 array.
    """
    arr = arr.astype(np.float32)

    vmax = float(arr.max())

    if vmax <= 0:
        return np.zeros_like(
            arr,
            dtype=np.float32,
        )

    return arr / vmax


def save_gray(
    path: Path,
    image: np.ndarray,
) -> None:
    """
    Save a 2D grayscale preview as a PNG file.
    """
    fig, ax = plt.subplots(
        figsize=(8, 8),
    )

    ax.imshow(
        image,
        cmap="gray",
    )

    ax.axis("off")

    fig.tight_layout(
        pad=0,
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.close(fig)


def save_rgb(
    path: Path,
    image: np.ndarray,
) -> None:
    """
    Save a 2D RGB preview as a PNG file.
    """
    fig, ax = plt.subplots(
        figsize=(8, 8),
    )

    ax.imshow(
        np.clip(
            image,
            0.0,
            1.0,
        )
    )

    ax.axis("off")

    fig.tight_layout(
        pad=0,
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.close(fig)


def create_overlay(
    image: np.ndarray,
    spine: np.ndarray,
    dendrite: np.ndarray,
) -> np.ndarray:
    """
    Create an RGB overlay from an image and corresponding masks.

    The image may be either a 2D image or a 3D stack.

    Dendrite pixels are shown in cyan-blue and spine pixels are shown
    in orange. If the two masks overlap, the spine color is shown.
    """
    image = normalize01(
        image
    )

    spine = spine > 0
    dendrite = dendrite > 0

    # Convert grayscale data to RGB.
    rgb = np.stack(
        [
            image,
            image,
            image,
        ],
        axis=-1,
    )

    # Add dendrite overlay.
    rgb[dendrite] = (
        (1.0 - OVERLAY_ALPHA)
        * rgb[dendrite]
        + OVERLAY_ALPHA
        * DENDRITE_COLOR
    )

    # Add spine overlay.
    # Spine color is shown where the masks overlap.
    rgb[spine] = (
        (1.0 - OVERLAY_ALPHA)
        * rgb[spine]
        + OVERLAY_ALPHA
        * SPINE_COLOR
    )

    return np.clip(
        rgb,
        0.0,
        1.0,
    )


def save_overlay_stack(
    path: Path,
    image_stack: np.ndarray,
    spine_stack: np.ndarray,
    dendrite_stack: np.ndarray,
) -> None:
    """
    Save the complete 3D RGB overlay as a TIFF stack.

    Input arrays:
        Z, Y, X

    Saved RGB stack:
        Z, Y, X, 3
    """
    rgb = create_overlay(
        image_stack,
        spine_stack,
        dendrite_stack,
    )

    rgb_uint8 = (
        rgb * 255
    ).astype(np.uint8)

    tifffile.imwrite(
        path,
        rgb_uint8,
        photometric="rgb",
        compression="zlib",
        metadata={
            "axes": "ZYXS",
        },
    )


def save_combined_figure(
    path: Path,
    image_mip: np.ndarray,
    dendrite_mip: np.ndarray,
    spine_mip: np.ndarray,
    overlay_mip: np.ndarray,
) -> None:
    """
    Save a horizontal four-panel summary figure.

    Panels:
        (a) clean synthetic image
        (b) dendrite ground truth
        (c) spine ground truth
        (d) image with ground-truth overlay
    """
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 6),
    )

    axes[0].imshow(
        normalize01(image_mip),
        cmap="gray",
    )
    axes[0].set_title(
        "(a) Clean synthetic image"
    )

    axes[1].imshow(
        dendrite_mip > 0,
        cmap="gray",
    )
    axes[1].set_title(
        "(b) Dendrite GT"
    )

    axes[2].imshow(
        spine_mip > 0,
        cmap="gray",
    )
    axes[2].set_title(
        "(c) Spine GT"
    )

    axes[3].imshow(
        overlay_mip,
    )
    axes[3].set_title(
        "(d) GT overlay"
    )

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.close(fig)


def find_one(
    output_dir: Path,
    pattern: str,
) -> Path:
    """
    Find exactly one required renderer output file.
    """
    matches = sorted(
        output_dir.glob(pattern)
    )

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Required renderer output not found: {pattern}\n"
            f"Checked directory: {output_dir}\n"
            "Run the labelled-mesh experiment with render.py "
            "before running this script."
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple files matched '{pattern}'.\n"
            "Expected exactly one renderer output:\n"
            + "\n".join(
                f"  {path}"
                for path in matches
            )
        )

    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create MIP previews, RGB overlays, and a four-panel "
            "summary figure from labelled-mesh renderer outputs."
        )
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Renderer output directory from a "
            "labelled-mesh experiment"
        ),
    )

    args = parser.parse_args()

    output_dir = Path(
        args.output_dir
    )

    if not output_dir.exists():
        raise FileNotFoundError(
            f"Output directory does not exist:\n"
            f"  {output_dir}"
        )

    if not output_dir.is_dir():
        raise NotADirectoryError(
            f"The supplied output path is not a directory:\n"
            f"  {output_dir}"
        )

    # Locate renderer outputs.
    image_path = find_one(
        output_dir,
        "*image.tif",
    )

    spine_path = find_one(
        output_dir,
        "*spine_mask.tif",
    )

    dendrite_path = find_one(
        output_dir,
        "*dendrite_mask.tif",
    )

    # Create preview directory.
    preview_dir = (
        output_dir / "previews"
    )

    preview_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading renderer outputs:")
    print(f"  image   : {image_path}")
    print(f"  spine   : {spine_path}")
    print(f"  dendrite: {dendrite_path}")

    # Load renderer outputs.
    image = tifffile.imread(
        image_path
    )

    spine = tifffile.imread(
        spine_path
    )

    dendrite = tifffile.imread(
        dendrite_path
    )

    # Check dimensions.
    if image.ndim != 3:
        raise ValueError(
            "Expected rendered image with shape "
            f"(Z, Y, X), received {image.shape}."
        )

    if spine.ndim != 3:
        raise ValueError(
            "Expected spine mask with shape "
            f"(Z, Y, X), received {spine.shape}."
        )

    if dendrite.ndim != 3:
        raise ValueError(
            "Expected dendrite mask with shape "
            f"(Z, Y, X), received {dendrite.shape}."
        )

    # All outputs must use the same voxel grid.
    if image.shape != spine.shape:
        raise ValueError(
            "Rendered image and spine mask have different shapes:\n"
            f"  image: {image.shape}\n"
            f"  spine: {spine.shape}"
        )

    if image.shape != dendrite.shape:
        raise ValueError(
            "Rendered image and dendrite mask have different shapes:\n"
            f"  image   : {image.shape}\n"
            f"  dendrite: {dendrite.shape}"
        )

    print(
        f"Volume shape ZYX: {image.shape}"
    )

    # Create maximum-intensity projections along Z.
    image_mip = image.max(
        axis=0
    )

    spine_mip = spine.max(
        axis=0
    )

    dendrite_mip = dendrite.max(
        axis=0
    )

    # Create the 2D overlay from the same MIPs used in the
    # individual preview panels.
    overlay_mip = create_overlay(
        image_mip,
        spine_mip,
        dendrite_mip,
    )

    # Output paths.
    image_mip_path = (
        preview_dir
        / "image_mip.png"
    )

    spine_mip_path = (
        preview_dir
        / "spine_mask_mip.png"
    )

    dendrite_mip_path = (
        preview_dir
        / "dendrite_mask_mip.png"
    )

    overlay_mip_path = (
        preview_dir
        / "overlay_mip.png"
    )

    overlay_stack_path = (
        preview_dir
        / "overlay_stack.tif"
    )

    combined_figure_path = (
        preview_dir
        / "labelled_render_summary.png"
    )

    # Save individual MIP previews.
    save_gray(
        image_mip_path,
        normalize01(image_mip),
    )

    save_gray(
        spine_mip_path,
        spine_mip > 0,
    )

    save_gray(
        dendrite_mip_path,
        dendrite_mip > 0,
    )

    # Save MIP overlay.
    save_rgb(
        overlay_mip_path,
        overlay_mip,
    )

    # Save complete 3D RGB overlay for inspection in Fiji/ImageJ.
    save_overlay_stack(
        overlay_stack_path,
        image,
        spine,
        dendrite,
    )

    # Save the final horizontal four-panel summary figure.
    save_combined_figure(
        combined_figure_path,
        image_mip,
        dendrite_mip,
        spine_mip,
        overlay_mip,
    )

    print("\nSaved preview files:")
    print(f"  {image_mip_path}")
    print(f"  {spine_mip_path}")
    print(f"  {dendrite_mip_path}")
    print(f"  {overlay_mip_path}")
    print(f"  {overlay_stack_path}")
    print(f"  {combined_figure_path}")


if __name__ == "__main__":
    main()