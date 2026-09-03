"""
visualize_deepd3_dataset.py
---------------------------

Visualize DeepD3 predictions together with synthetic ground-truth masks.

The script is intended for qualitative inspection of DeepD3 predictions
before quantitative evaluation.

For each selected dataset instance it loads:

    noisy.tif
    dendrite_mask.tif
    spine_mask.tif

and the DeepD3 prediction files:

    deepd3_predictions/32F.prediction
    deepd3_predictions/32F_94nm.prediction

The DeepD3 prediction files contain probability volumes for:

    dendrites
    spines

The script creates XY maximum-intensity projections (MIPs) and saves
visualizations comparing the synthetic ground truth with predictions
from both pretrained DeepD3 models.

Expected project structure
--------------------------

outputs/
└── synthetic_dataset_v1/
    └── test/
        └── instance_000001/
            ├── noisy.tif
            ├── dendrite_mask.tif
            ├── spine_mask.tif
            └── deepd3_predictions/
                ├── 32F.prediction
                └── 32F_94nm.prediction

Usage
-----

Run from the repository root.

Visualize the first test instance:

    python deepd3/visualize_deepd3_dataset.py \
        --instance instance_000001

Visualize another instance:

    python deepd3/visualize_deepd3_dataset.py \
        --instance instance_000010

Use another dataset split:

    python deepd3/visualize_deepd3_dataset.py \
        --split validation \
        --instance instance_000001
"""

from __future__ import annotations

import argparse
from pathlib import Path

import flammkuchen as fl
import matplotlib.pyplot as plt
import numpy as np
import tifffile


# ==========================================================
# Default paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET = (
    PROJECT_ROOT
    / "outputs"
    / "synthetic_dataset_v1"
)


# ==========================================================
# Command-line arguments
# ==========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize DeepD3 predictions and synthetic "
            "ground-truth masks."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=(
            "Dataset root containing train/validation/test folders. "
            f"Default: {DEFAULT_DATASET}"
        ),
    )

    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
        help="Dataset split. Default: test.",
    )

    parser.add_argument(
        "--instance",
        default="instance_000001",
        help="Dataset instance to visualize.",
    )

    return parser.parse_args()


# ==========================================================
# Helpers
# ==========================================================

def require_file(path: Path, description: str):
    """
    Check that a required file exists.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"{description} not found:\n"
            f"  {path}"
        )


def normalize01(arr):
    """
    Normalize an array to the range [0, 1].
    """

    arr = np.asarray(
        arr,
        dtype=np.float32,
    )

    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    vmin = float(arr.min())
    vmax = float(arr.max())

    if vmax <= vmin:
        return np.zeros_like(
            arr,
            dtype=np.float32,
        )

    return (
        (arr - vmin)
        / (vmax - vmin)
    )


def clean_probability(arr):
    """
    Convert a DeepD3 prediction array to a finite float32 array.
    """

    arr = np.asarray(
        arr,
        dtype=np.float32,
    )

    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    return arr


def load_prediction(path: Path):
    """
    Load a DeepD3 .prediction file.

    Returns
    -------
    dendrite_probability
        Predicted dendrite probability volume.

    spine_probability
        Predicted spine probability volume.
    """

    data = fl.load(
        str(path)
    )

    if not isinstance(data, dict):
        raise TypeError(
            f"Unexpected DeepD3 prediction format:\n"
            f"  {path}\n"
            f"Expected a dictionary, got {type(data)}."
        )

    if "dendrites" not in data:
        raise KeyError(
            f"'dendrites' not found in:\n"
            f"  {path}\n"
            f"Available keys: {list(data.keys())}"
        )

    if "spines" not in data:
        raise KeyError(
            f"'spines' not found in:\n"
            f"  {path}\n"
            f"Available keys: {list(data.keys())}"
        )

    dendrites = clean_probability(
        data["dendrites"]
    )

    spines = clean_probability(
        data["spines"]
    )

    return dendrites, spines


def make_overlay(
    image,
    dendrite_mask,
    spine_mask,
):
    """
    Create an RGB overlay.

    Grayscale:
        microscopy image

    Cyan:
        dendrite mask

    Orange:
        spine mask
    """

    image = normalize01(
        image
    )

    rgb = np.stack(
        [
            image,
            image,
            image,
        ],
        axis=-1,
    )

    dendrite_pixels = (
        dendrite_mask > 0
    )

    spine_pixels = (
        spine_mask > 0
    )

    dendrite_color = np.array(
        [0.0, 0.7, 1.0],
        dtype=np.float32,
    )

    spine_color = np.array(
        [1.0, 0.6, 0.0],
        dtype=np.float32,
    )

    alpha = 0.65

    rgb[dendrite_pixels] = (
        (1.0 - alpha)
        * rgb[dendrite_pixels]
        + alpha
        * dendrite_color
    )

    rgb[spine_pixels] = (
        (1.0 - alpha)
        * rgb[spine_pixels]
        + alpha
        * spine_color
    )

    return np.clip(
        rgb,
        0.0,
        1.0,
    )


# ==========================================================
# Main
# ==========================================================

def main():
    args = parse_args()

    dataset_root = args.dataset.resolve()

    instance_dir = (
        dataset_root
        / args.split
        / args.instance
    )

    if not instance_dir.is_dir():
        raise FileNotFoundError(
            f"Instance directory not found:\n"
            f"  {instance_dir}"
        )

    # ------------------------------------------------------
    # Input files
    # ------------------------------------------------------

    noisy_path = (
        instance_dir
        / "noisy.tif"
    )

    dendrite_gt_path = (
        instance_dir
        / "dendrite_mask.tif"
    )

    spine_gt_path = (
        instance_dir
        / "spine_mask.tif"
    )

    prediction_dir = (
        instance_dir
        / "deepd3_predictions"
    )

    prediction_32f_path = (
        prediction_dir
        / "32F.prediction"
    )

    prediction_94nm_path = (
        prediction_dir
        / "32F_94nm.prediction"
    )

    require_file(
        noisy_path,
        "Noisy microscopy image",
    )

    require_file(
        dendrite_gt_path,
        "Dendrite GT mask",
    )

    require_file(
        spine_gt_path,
        "Spine GT mask",
    )

    require_file(
        prediction_32f_path,
        "DeepD3 32F prediction",
    )

    require_file(
        prediction_94nm_path,
        "DeepD3 32F 94 nm prediction",
    )

    # ------------------------------------------------------
    # Load microscopy + GT
    # ------------------------------------------------------

    noisy = tifffile.imread(
        noisy_path
    )

    dendrite_gt = tifffile.imread(
        dendrite_gt_path
    ) > 0

    spine_gt = tifffile.imread(
        spine_gt_path
    ) > 0

    # ------------------------------------------------------
    # Load DeepD3 predictions
    # ------------------------------------------------------

    (
        dendrite_32f,
        spine_32f,
    ) = load_prediction(
        prediction_32f_path
    )

    (
        dendrite_94nm,
        spine_94nm,
    ) = load_prediction(
        prediction_94nm_path
    )

    # ------------------------------------------------------
    # Print shapes and ranges
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("DeepD3 prediction visualization")
    print("=" * 70)

    print(
        f"Instance : {args.instance}"
    )

    print(
        f"Split    : {args.split}"
    )

    print()

    print(
        f"Noisy shape       : {noisy.shape}"
    )

    print(
        f"Dendrite GT shape : {dendrite_gt.shape}"
    )

    print(
        f"Spine GT shape    : {spine_gt.shape}"
    )

    print()

    print(
        f"32F dendrite      : {dendrite_32f.shape}"
    )

    print(
        f"32F spine         : {spine_32f.shape}"
    )

    print(
        f"94nm dendrite     : {dendrite_94nm.shape}"
    )

    print(
        f"94nm spine        : {spine_94nm.shape}"
    )

    print()

    print(
        "32F dendrite range : "
        f"{dendrite_32f.min():.4f} - "
        f"{dendrite_32f.max():.4f}"
    )

    print(
        "32F spine range    : "
        f"{spine_32f.min():.4f} - "
        f"{spine_32f.max():.4f}"
    )

    print(
        "94nm dendrite range: "
        f"{dendrite_94nm.min():.4f} - "
        f"{dendrite_94nm.max():.4f}"
    )

    print(
        "94nm spine range   : "
        f"{spine_94nm.min():.4f} - "
        f"{spine_94nm.max():.4f}"
    )

    print("=" * 70)

    # ------------------------------------------------------
    # Shape check
    # ------------------------------------------------------

    expected_shape = noisy.shape

    prediction_arrays = {
        "32F dendrite":
            dendrite_32f,

        "32F spine":
            spine_32f,

        "94nm dendrite":
            dendrite_94nm,

        "94nm spine":
            spine_94nm,
    }

    for name, arr in prediction_arrays.items():
        if arr.shape != expected_shape:
            raise ValueError(
                f"{name} shape does not match input.\n"
                f"Input      : {expected_shape}\n"
                f"Prediction : {arr.shape}\n\n"
                "Do not continue with evaluation until "
                "the orientation/shape difference is understood."
            )

    # ------------------------------------------------------
    # XY MIPs
    # ------------------------------------------------------

    noisy_mip = noisy.max(
        axis=0
    )

    dendrite_gt_mip = dendrite_gt.max(
        axis=0
    )

    spine_gt_mip = spine_gt.max(
        axis=0
    )

    dendrite_32f_mip = dendrite_32f.max(
        axis=0
    )

    spine_32f_mip = spine_32f.max(
        axis=0
    )

    dendrite_94nm_mip = dendrite_94nm.max(
        axis=0
    )

    spine_94nm_mip = spine_94nm.max(
        axis=0
    )

    # ------------------------------------------------------
    # Figure 1: raw probability MIPs
    # ------------------------------------------------------

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(12, 12),
    )

    axes[0, 0].imshow(
        noisy_mip,
        cmap="gray",
    )

    axes[0, 0].set_title(
        "Noisy synthetic input"
    )

    axes[0, 1].imshow(
        dendrite_gt_mip,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[0, 1].set_title(
        "GT dendrite"
    )

    axes[0, 2].imshow(
        spine_gt_mip,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[0, 2].set_title(
        "GT spines"
    )

    axes[1, 0].axis(
        "off"
    )

    axes[1, 1].imshow(
        dendrite_32f_mip,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[1, 1].set_title(
        "32F dendrite probability"
    )

    axes[1, 2].imshow(
        spine_32f_mip,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[1, 2].set_title(
        "32F spine probability"
    )

    axes[2, 0].axis(
        "off"
    )

    axes[2, 1].imshow(
        dendrite_94nm_mip,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[2, 1].set_title(
        "32F 94nm dendrite probability"
    )

    axes[2, 2].imshow(
        spine_94nm_mip,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[2, 2].set_title(
        "32F 94nm spine probability"
    )

    for ax in axes.flat:
        ax.axis(
            "off"
        )

    fig.suptitle(
        f"{args.split}/{args.instance}",
        fontsize=14,
    )

    fig.tight_layout()

    # ------------------------------------------------------
    # Figure 2: GT overlay
    # ------------------------------------------------------

    gt_overlay = make_overlay(
        noisy_mip,
        dendrite_gt_mip,
        spine_gt_mip,
    )

    fig_overlay, ax_overlay = plt.subplots(
        figsize=(7, 7)
    )

    ax_overlay.imshow(
        gt_overlay
    )

    ax_overlay.set_title(
        "Synthetic input + GT\n"
        "Cyan = dendrite, Orange = spines"
    )

    ax_overlay.axis(
        "off"
    )

    fig_overlay.tight_layout()

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    output_dir = (
        instance_dir
        / "deepd3_visualization"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    probability_output = (
        output_dir
        / "prediction_probability_mips.png"
    )

    gt_output = (
        output_dir
        / "ground_truth_overlay.png"
    )

    fig.savefig(
        probability_output,
        dpi=200,
        bbox_inches="tight",
    )

    fig_overlay.savefig(
        gt_output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    plt.close(
        fig_overlay
    )

    print()
    print("Saved:")
    print(
        f"  {probability_output}"
    )
    print(
        f"  {gt_output}"
    )

    print()
    print("Visualization complete.")


if __name__ == "__main__":
    main()