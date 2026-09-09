"""
Visualize DeepD3 Dataset
------------------------

Create qualitative DeepD3 visualizations for synthetic dataset instances.

Supported models:
    - Original pretrained DeepD3 32F
    - Original pretrained DeepD3 32F 94 nm
    - Synthetic-trained DeepD3 32F 94 nm

For each model, two figures are generated:

SPINE DETECTION
    1. Synthetic noisy image - XY maximum projection
    2. Ground-truth spine mask - XY maximum projection
    3. DeepD3 spine probability - XY maximum projection
    4. GT + predicted spine centers with matching

DENDRITE SEGMENTATION
    1. Synthetic noisy image - XY maximum projection
    2. Ground-truth dendrite mask - XY maximum projection
    3. DeepD3 dendrite probability - XY maximum projection
    4. Segmentation error overlay:
           TP = correct overlap
           FP = predicted dendrite outside GT
           FN = GT dendrite missed by prediction

All thresholds are loaded from the VALIDATION set.

Therefore TEST visualization uses frozen validation thresholds
and does not tune thresholds on the test set.


Usage
-----

Visualize one test instance for all models:

    python deepd3/visualize_deepd3_dataset.py \
        --split test \
        --instance instance_000001 \
        --model all

Visualize all test instances for all models:

    python deepd3/visualize_deepd3_dataset.py \
        --split test \
        --instance all \
        --model all

Visualize only original 32F:

    python deepd3/visualize_deepd3_dataset.py \
        --split test \
        --instance all \
        --model 32F

Visualize only original 32F 94 nm:

    python deepd3/visualize_deepd3_dataset.py \
        --split test \
        --instance all \
        --model 32F_94nm

Visualize only synthetic-trained model:

    python deepd3/visualize_deepd3_dataset.py \
        --split test \
        --instance all \
        --model synthetic_32F_94nm

Visualize one validation instance using the synthetic-trained model:

    python deepd3/visualize_deepd3_dataset.py \
        --split validation \
        --instance instance_000001 \
        --model synthetic_32F_94nm


Outputs
-------

For each instance:

instance_XXXXXX/
└── deepd3_visualization/
    ├── matching_32F.png
    ├── dendrite_32F.png
    ├── matching_32F_94nm.png
    ├── dendrite_32F_94nm.png
    ├── matching_synthetic_32F_94nm.png
    └── dendrite_synthetic_32F_94nm.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import flammkuchen as fl
import matplotlib.pyplot as plt
import numpy as np
import tifffile

from scipy.ndimage import (
    center_of_mass,
    gaussian_filter,
    label,
    maximum_filter,
)


# ==========================================================
# Paths / constants
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET = (
    PROJECT_ROOT
    / "outputs"
    / "synthetic_dataset_v1"
)

# Dataset physical spacing: Z, Y, X
SPACING_ZYX_NM = np.array(
    [500.0, 94.0, 94.0],
    dtype=np.float64,
)

# Same settings as quantitative evaluation
MATCH_DISTANCE_NM = 1000.0

NEIGHBORHOOD_ZYX = (
    5,
    9,
    9,
)

SMOOTH_SIGMA = 1.0

MODELS = (
    "32F",
    "32F_94nm",
    "synthetic_32F_94nm",
)


# ==========================================================
# Command-line arguments
# ==========================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Visualize DeepD3 spine detection and "
            "dendrite segmentation for synthetic dataset instances."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=(
            "Dataset root containing train/validation/test. "
            f"Default: {DEFAULT_DATASET}"
        ),
    )

    parser.add_argument(
        "--split",
        choices=(
            "train",
            "validation",
            "test",
        ),
        default="test",
        help="Dataset split. Default: test.",
    )

    parser.add_argument(
        "--instance",
        default="instance_000001",
        help=(
            "Dataset instance, for example instance_000001. "
            "Use 'all' to visualize every instance "
            "in the selected split."
        ),
    )

    parser.add_argument(
        "--model",
        choices=(
            "32F",
            "32F_94nm",
            "synthetic_32F_94nm",
            "all",
        ),
        default="all",
        help=(
            "DeepD3 model to visualize. "
            "Use 'all' to visualize all supported models. "
            "Default: all."
        ),
    )

    return parser.parse_args()


# ==========================================================
# Generic helpers
# ==========================================================

def require_file(
    path: Path,
    description: str,
):

    if not path.is_file():

        raise FileNotFoundError(
            f"{description} not found:\n"
            f"  {path}"
        )


def normalize01(
    arr,
):

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

    vmin = float(
        arr.min()
    )

    vmax = float(
        arr.max()
    )

    if vmax <= vmin:

        return np.zeros_like(
            arr,
            dtype=np.float32,
        )

    return (
        (arr - vmin)
        / (vmax - vmin)
    )


# ==========================================================
# Load DeepD3 prediction
# ==========================================================

def load_prediction(
    path: Path,
):

    data = fl.load(
        str(path)
    )

    if not isinstance(
        data,
        dict,
    ):

        raise TypeError(
            f"Unexpected DeepD3 prediction format:\n"
            f"  {path}\n"
            f"Expected dictionary, got {type(data)}."
        )

    if "spines" not in data:

        raise KeyError(
            f"'spines' not found in:\n"
            f"  {path}\n"
            f"Available keys: {list(data.keys())}"
        )

    if "dendrites" not in data:

        raise KeyError(
            f"'dendrites' not found in:\n"
            f"  {path}\n"
            f"Available keys: {list(data.keys())}"
        )

    spine_probability = np.asarray(
        data["spines"],
        dtype=np.float32,
    )

    dendrite_probability = np.asarray(
        data["dendrites"],
        dtype=np.float32,
    )

    spine_probability = np.nan_to_num(
        spine_probability,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    dendrite_probability = np.nan_to_num(
        dendrite_probability,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    return (
        spine_probability,
        dendrite_probability,
    )


# ==========================================================
# Load frozen validation thresholds
# ==========================================================

def extract_spine_detection_threshold(
    model_data,
):

    possible_keys = (
        "spine_detection_threshold",
        "detection_threshold",
        "spine_detection",
        "spine_detection_thr",
    )

    for key in possible_keys:

        if key in model_data:

            return float(
                model_data[key]
            )

    raise KeyError(
        "Could not find spine-detection threshold.\n"
        f"Available keys: {list(model_data.keys())}"
    )


def extract_dendrite_segmentation_threshold(
    model_data,
):

    possible_keys = (
        "dendrite_segmentation_threshold",
        "dendrite_threshold",
        "dendrite_segmentation",
        "dendrite_segmentation_thr",
    )

    for key in possible_keys:

        if key in model_data:

            return float(
                model_data[key]
            )

    raise KeyError(
        "Could not find dendrite-segmentation threshold.\n"
        f"Available keys: {list(model_data.keys())}"
    )


def load_thresholds(
    dataset_root: Path,
):

    threshold_path = (
        dataset_root
        / "deepd3_evaluation"
        / "validation"
        / "selected_thresholds.json"
    )

    require_file(
        threshold_path,
        "Validation threshold file",
    )

    with open(
        threshold_path,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(
            f
        )

    thresholds = {}

    for model in MODELS:

        if model not in data:

            raise KeyError(
                f"Model '{model}' not found in:\n"
                f"  {threshold_path}\n"
                f"Available keys: {list(data.keys())}"
            )

        thresholds[model] = {
            "spine_detection": (
                extract_spine_detection_threshold(
                    data[model]
                )
            ),
            "dendrite_segmentation": (
                extract_dendrite_segmentation_threshold(
                    data[model]
                )
            ),
        }

    return (
        thresholds,
        threshold_path,
    )


# ==========================================================
# Ground-truth spine centers
# ==========================================================

def extract_gt_centers(
    spine_mask,
):

    binary = (
        np.asarray(
            spine_mask
        )
        > 0
    )

    labeled_mask, count = label(
        binary
    )

    if count == 0:

        return np.zeros(
            (0, 3),
            dtype=np.float64,
        )

    centers = center_of_mass(
        binary,
        labeled_mask,
        range(
            1,
            count + 1,
        ),
    )

    return np.asarray(
        centers,
        dtype=np.float64,
    )


# ==========================================================
# Predicted spine centers
# ==========================================================

def detect_predicted_centers(
    spine_probability,
    threshold,
):

    smoothed = gaussian_filter(
        spine_probability,
        sigma=SMOOTH_SIGMA,
    )

    local_maximum = maximum_filter(
        smoothed,
        size=NEIGHBORHOOD_ZYX,
        mode="nearest",
    )

    peak_mask = (
        (smoothed == local_maximum)
        &
        (smoothed >= threshold)
    )

    centers = np.argwhere(
        peak_mask
    )

    return centers.astype(
        np.float64
    )


# ==========================================================
# Physical center distances
# ==========================================================

def calculate_distance_matrix_nm(
    gt_centers,
    pred_centers,
):

    n_gt = len(
        gt_centers
    )

    n_pred = len(
        pred_centers
    )

    if (
        n_gt == 0
        or n_pred == 0
    ):

        return np.zeros(
            (
                n_gt,
                n_pred,
            ),
            dtype=np.float64,
        )

    difference_voxels = (
        gt_centers[:, None, :]
        -
        pred_centers[None, :, :]
    )

    difference_nm = (
        difference_voxels
        *
        SPACING_ZYX_NM[
            None,
            None,
            :
        ]
    )

    distance_nm = np.sqrt(
        np.sum(
            difference_nm ** 2,
            axis=2,
        )
    )

    return distance_nm


# ==========================================================
# Greedy one-to-one matching
# ==========================================================

def match_centers(
    gt_centers,
    pred_centers,
):

    distances = calculate_distance_matrix_nm(
        gt_centers,
        pred_centers,
    )

    candidates = []

    for gt_index in range(
        distances.shape[0]
    ):

        for pred_index in range(
            distances.shape[1]
        ):

            distance = distances[
                gt_index,
                pred_index,
            ]

            if (
                distance
                <= MATCH_DISTANCE_NM
            ):

                candidates.append(
                    (
                        float(
                            distance
                        ),
                        gt_index,
                        pred_index,
                    )
                )

    candidates.sort(
        key=lambda item:
            item[0]
    )

    used_gt = set()
    used_pred = set()

    matches = []

    for (
        distance,
        gt_index,
        pred_index,
    ) in candidates:

        if gt_index in used_gt:
            continue

        if pred_index in used_pred:
            continue

        used_gt.add(
            gt_index
        )

        used_pred.add(
            pred_index
        )

        matches.append(
            (
                gt_index,
                pred_index,
                distance,
            )
        )

    return matches


# ==========================================================
# Spine detection metrics
# ==========================================================

def calculate_detection_metrics(
    gt_centers,
    pred_centers,
    matches,
):

    tp = len(
        matches
    )

    fp = (
        len(
            pred_centers
        )
        - tp
    )

    fn = (
        len(
            gt_centers
        )
        - tp
    )

    precision = (
        tp
        / (
            tp
            + fp
        )
        if (
            tp
            + fp
        ) > 0
        else 0.0
    )

    recall = (
        tp
        / (
            tp
            + fn
        )
        if (
            tp
            + fn
        ) > 0
        else 0.0
    )

    f1 = (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
        if (
            precision
            + recall
        ) > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ==========================================================
# Segmentation metrics
# ==========================================================

def calculate_segmentation_metrics(
    gt_mask,
    pred_mask,
):

    gt = (
        np.asarray(
            gt_mask
        )
        > 0
    )

    pred = (
        np.asarray(
            pred_mask
        )
        > 0
    )

    intersection = np.logical_and(
        gt,
        pred,
    ).sum()

    union = np.logical_or(
        gt,
        pred,
    ).sum()

    gt_sum = gt.sum()
    pred_sum = pred.sum()

    iou = (
        intersection
        / union
        if union > 0
        else 1.0
    )

    dice_denominator = (
        gt_sum
        + pred_sum
    )

    dice = (
        2.0
        * intersection
        / dice_denominator
        if dice_denominator > 0
        else 1.0
    )

    return {
        "iou":
            float(
                iou
            ),
        "dice":
            float(
                dice
            ),
    }


# ==========================================================
# Spine matching visualization
# ==========================================================

def create_matching_figure(
    noisy,
    spine_gt,
    spine_probability,
    gt_centers,
    pred_centers,
    matches,
    metrics,
    threshold,
    model,
    split,
    instance,
    output_path,
):

    noisy_mip = normalize01(
        noisy.max(
            axis=0
        )
    )

    spine_gt_mip = (
        spine_gt.max(
            axis=0
        )
    )

    spine_probability_mip = (
        spine_probability.max(
            axis=0
        )
    )

    matched_gt = {
        gt_index
        for (
            gt_index,
            pred_index,
            distance,
        ) in matches
    }

    matched_pred = {
        pred_index
        for (
            gt_index,
            pred_index,
            distance,
        ) in matches
    }

    missed_gt = [
        index
        for index in range(
            len(
                gt_centers
            )
        )
        if index not in matched_gt
    ]

    false_positive_pred = [
        index
        for index in range(
            len(
                pred_centers
            )
        )
        if index not in matched_pred
    ]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(
            18,
            5,
        ),
    )

    axes[0].imshow(
        noisy_mip,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[0].set_title(
        "Synthetic image\n"
        "max projection"
    )

    axes[1].imshow(
        spine_gt_mip,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[1].set_title(
        "GT spine mask\n"
        "max projection"
    )

    axes[2].imshow(
        spine_probability_mip,
        cmap="hot",
        vmin=0,
        vmax=1,
    )

    axes[2].set_title(
        f"DeepD3 {model}\n"
        "spine probability"
    )

    axes[3].imshow(
        noisy_mip,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    if len(
        gt_centers
    ) > 0:

        axes[3].scatter(
            gt_centers[:, 2],
            gt_centers[:, 1],
            marker="o",
            facecolors="none",
            edgecolors="cyan",
            s=38,
            linewidths=1.0,
            label="GT centers",
        )

    matched_pred_indices = sorted(
        matched_pred
    )

    if len(
        matched_pred_indices
    ) > 0:

        axes[3].scatter(
            pred_centers[
                matched_pred_indices,
                2,
            ],
            pred_centers[
                matched_pred_indices,
                1,
            ],
            marker="x",
            s=36,
            linewidths=1.2,
            color="magenta",
            label="Matched prediction",
        )

    if len(
        false_positive_pred
    ) > 0:

        axes[3].scatter(
            pred_centers[
                false_positive_pred,
                2,
            ],
            pred_centers[
                false_positive_pred,
                1,
            ],
            marker="+",
            s=38,
            linewidths=1.2,
            color="red",
            label="False positive",
        )

    if len(
        missed_gt
    ) > 0:

        axes[3].scatter(
            gt_centers[
                missed_gt,
                2,
            ],
            gt_centers[
                missed_gt,
                1,
            ],
            marker="x",
            s=46,
            linewidths=1.5,
            color="yellow",
            label="Missed GT",
        )

    for (
        gt_index,
        pred_index,
        distance,
    ) in matches:

        gt = gt_centers[
            gt_index
        ]

        pred = pred_centers[
            pred_index
        ]

        axes[3].plot(
            [
                gt[2],
                pred[2],
            ],
            [
                gt[1],
                pred[1],
            ],
            color="lime",
            linewidth=0.8,
            alpha=0.8,
        )

    axes[3].set_title(
        "Overlay\n"
        "GT + predictions"
    )

    axes[3].legend(
        loc="upper right",
        fontsize=6.5,
        framealpha=0.8,
    )

    for ax in axes:

        ax.axis(
            "off"
        )

    fig.suptitle(
        (
            f"{split}/{instance} — DeepD3 {model} — Spine detection\n"
            f"threshold={threshold:.2f}, "
            f"match={MATCH_DISTANCE_NM:.0f} nm, "
            f"TP={metrics['tp']}, "
            f"FP={metrics['fp']}, "
            f"FN={metrics['fn']}, "
            f"precision={metrics['precision']:.3f}, "
            f"recall={metrics['recall']:.3f}, "
            f"F1={metrics['f1']:.3f}"
        ),
        fontsize=11,
    )

    fig.tight_layout(
        rect=(
            0,
            0,
            1,
            0.90,
        )
    )

    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ==========================================================
# Dendrite visualization
# ==========================================================

def create_dendrite_figure(
    noisy,
    dendrite_gt,
    dendrite_probability,
    dendrite_prediction,
    metrics,
    threshold,
    model,
    split,
    instance,
    output_path,
):

    noisy_mip = normalize01(
        noisy.max(
            axis=0
        )
    )

    dendrite_gt_mip = (
        dendrite_gt.max(
            axis=0
        )
    )

    dendrite_probability_mip = (
        dendrite_probability.max(
            axis=0
        )
    )

    dendrite_prediction_mip = (
        dendrite_prediction.max(
            axis=0
        )
    )

    gt_2d = (
        dendrite_gt_mip
        > 0
    )

    pred_2d = (
        dendrite_prediction_mip
        > 0
    )

    tp_2d = np.logical_and(
        gt_2d,
        pred_2d,
    )

    fp_2d = np.logical_and(
        np.logical_not(
            gt_2d
        ),
        pred_2d,
    )

    fn_2d = np.logical_and(
        gt_2d,
        np.logical_not(
            pred_2d
        ),
    )

    error_overlay = np.zeros(
        (
            gt_2d.shape[0],
            gt_2d.shape[1],
            3,
        ),
        dtype=np.float32,
    )

    # TP = green
    error_overlay[
        tp_2d
    ] = [
        0.0,
        1.0,
        0.0,
    ]

    # FP = red
    error_overlay[
        fp_2d
    ] = [
        1.0,
        0.0,
        0.0,
    ]

    # FN = blue
    error_overlay[
        fn_2d
    ] = [
        0.0,
        0.4,
        1.0,
    ]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(
            18,
            5,
        ),
    )

    axes[0].imshow(
        noisy_mip,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[0].set_title(
        "Synthetic image\n"
        "max projection"
    )

    axes[1].imshow(
        dendrite_gt_mip,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[1].set_title(
        "GT dendrite mask\n"
        "max projection"
    )

    axes[2].imshow(
        dendrite_probability_mip,
        cmap="hot",
        vmin=0,
        vmax=1,
    )

    axes[2].set_title(
        f"DeepD3 {model}\n"
        "dendrite probability"
    )

    axes[3].imshow(
        error_overlay
    )

    axes[3].set_title(
        "Segmentation error\n"
        "Green=TP, Red=FP, Blue=FN"
    )

    for ax in axes:

        ax.axis(
            "off"
        )

    fig.suptitle(
        (
            f"{split}/{instance} — DeepD3 {model} — "
            f"Dendrite segmentation\n"
            f"threshold={threshold:.2f}, "
            f"3D IoU={metrics['iou']:.3f}, "
            f"3D Dice={metrics['dice']:.3f}"
        ),
        fontsize=11,
    )

    fig.tight_layout(
        rect=(
            0,
            0,
            1,
            0.90,
        )
    )

    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ==========================================================
# Process one dataset instance
# ==========================================================

def process_instance(
    dataset_root,
    split,
    instance_name,
    selected_models,
    thresholds,
    threshold_path,
):

    instance_dir = (
        dataset_root
        / split
        / instance_name
    )

    if not instance_dir.is_dir():

        raise FileNotFoundError(
            f"Instance directory not found:\n"
            f"  {instance_dir}"
        )

    noisy_path = (
        instance_dir
        / "noisy.tif"
    )

    spine_gt_path = (
        instance_dir
        / "spine_mask.tif"
    )

    dendrite_gt_path = (
        instance_dir
        / "dendrite_mask.tif"
    )

    require_file(
        noisy_path,
        "Noisy synthetic image",
    )

    require_file(
        spine_gt_path,
        "GT spine mask",
    )

    require_file(
        dendrite_gt_path,
        "GT dendrite mask",
    )

    noisy = tifffile.imread(
        noisy_path
    )

    spine_gt = (
        tifffile.imread(
            spine_gt_path
        )
        > 0
    )

    dendrite_gt = (
        tifffile.imread(
            dendrite_gt_path
        )
        > 0
    )

    if (
        noisy.shape != spine_gt.shape
        or noisy.shape != dendrite_gt.shape
    ):

        raise ValueError(
            "Image / GT shape mismatch.\n"
            f"Noisy image : {noisy.shape}\n"
            f"Spine GT    : {spine_gt.shape}\n"
            f"Dendrite GT : {dendrite_gt.shape}"
        )

    gt_centers = (
        extract_gt_centers(
            spine_gt
        )
    )

    output_dir = (
        instance_dir
        / "deepd3_visualization"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()

    print(
        "=" * 70
    )

    print(
        "DeepD3 qualitative visualization"
    )

    print(
        "=" * 70
    )

    print(
        f"Split      : {split}"
    )

    print(
        f"Instance   : {instance_name}"
    )

    print(
        f"Image shape: {noisy.shape}"
    )

    print(
        f"GT centers : {len(gt_centers)}"
    )

    print(
        f"Spacing    : "
        f"{tuple(SPACING_ZYX_NM)} nm (Z,Y,X)"
    )

    print(
        f"Match      : "
        f"{MATCH_DISTANCE_NM:.0f} nm"
    )

    print(
        f"Thresholds : {threshold_path}"
    )

    print(
        "=" * 70
    )

    for model in selected_models:

        prediction_path = (
            instance_dir
            / "deepd3_predictions"
            / f"{model}.prediction"
        )

        require_file(
            prediction_path,
            f"DeepD3 {model} prediction",
        )

        (
            spine_probability,
            dendrite_probability,
        ) = load_prediction(
            prediction_path
        )

        if (
            spine_probability.shape
            != noisy.shape
            or dendrite_probability.shape
            != noisy.shape
        ):

            raise ValueError(
                f"{model} prediction shape mismatch.\n"
                f"Image     : {noisy.shape}\n"
                f"Spines    : {spine_probability.shape}\n"
                f"Dendrites : {dendrite_probability.shape}"
            )

        # ==================================================
        # Spine detection
        # ==================================================

        spine_threshold = (
            thresholds[
                model
            ][
                "spine_detection"
            ]
        )

        pred_centers = (
            detect_predicted_centers(
                spine_probability,
                spine_threshold,
            )
        )

        matches = (
            match_centers(
                gt_centers,
                pred_centers,
            )
        )

        detection_metrics = (
            calculate_detection_metrics(
                gt_centers,
                pred_centers,
                matches,
            )
        )

        spine_output_path = (
            output_dir
            / f"matching_{model}.png"
        )

        create_matching_figure(
            noisy=noisy,
            spine_gt=spine_gt,
            spine_probability=spine_probability,
            gt_centers=gt_centers,
            pred_centers=pred_centers,
            matches=matches,
            metrics=detection_metrics,
            threshold=spine_threshold,
            model=model,
            split=split,
            instance=instance_name,
            output_path=spine_output_path,
        )

        # ==================================================
        # Dendrite segmentation
        # ==================================================

        dendrite_threshold = (
            thresholds[
                model
            ][
                "dendrite_segmentation"
            ]
        )

        dendrite_prediction = (
            dendrite_probability
            >= dendrite_threshold
        )

        dendrite_metrics = (
            calculate_segmentation_metrics(
                dendrite_gt,
                dendrite_prediction,
            )
        )

        dendrite_output_path = (
            output_dir
            / f"dendrite_{model}.png"
        )

        create_dendrite_figure(
            noisy=noisy,
            dendrite_gt=dendrite_gt,
            dendrite_probability=dendrite_probability,
            dendrite_prediction=dendrite_prediction,
            metrics=dendrite_metrics,
            threshold=dendrite_threshold,
            model=model,
            split=split,
            instance=instance_name,
            output_path=dendrite_output_path,
        )

        print()

        print(
            f"Model      : {model}"
        )

        print(
            "SPINE DETECTION"
        )

        print(
            f"Threshold  : "
            f"{spine_threshold:.2f}"
        )

        print(
            f"GT centers : "
            f"{len(gt_centers)}"
        )

        print(
            f"Predicted  : "
            f"{len(pred_centers)}"
        )

        print(
            "TP / FP / FN: "
            f"{detection_metrics['tp']} / "
            f"{detection_metrics['fp']} / "
            f"{detection_metrics['fn']}"
        )

        print(
            f"Precision  : "
            f"{detection_metrics['precision']:.4f}"
        )

        print(
            f"Recall     : "
            f"{detection_metrics['recall']:.4f}"
        )

        print(
            f"F1         : "
            f"{detection_metrics['f1']:.4f}"
        )

        print(
            f"Saved      : "
            f"{spine_output_path}"
        )

        print()

        print(
            "DENDRITE SEGMENTATION"
        )

        print(
            f"Threshold  : "
            f"{dendrite_threshold:.2f}"
        )

        print(
            f"IoU        : "
            f"{dendrite_metrics['iou']:.4f}"
        )

        print(
            f"Dice       : "
            f"{dendrite_metrics['dice']:.4f}"
        )

        print(
            f"Saved      : "
            f"{dendrite_output_path}"
        )


# ==========================================================
# Main
# ==========================================================

def main():

    args = parse_args()

    dataset_root = (
        args.dataset.resolve()
    )

    (
        thresholds,
        threshold_path,
    ) = load_thresholds(
        dataset_root
    )

    # ------------------------------------------------------
    # Select model(s)
    # ------------------------------------------------------

    if args.model == "all":

        selected_models = (
            MODELS
        )

    else:

        selected_models = (
            args.model,
        )

    # ------------------------------------------------------
    # Select instance(s)
    # ------------------------------------------------------

    split_dir = (
        dataset_root
        / args.split
    )

    if not split_dir.is_dir():

        raise FileNotFoundError(
            f"Split directory not found:\n"
            f"  {split_dir}"
        )

    if (
        args.instance.lower()
        == "all"
    ):

        instance_dirs = sorted(
            path
            for path in split_dir.glob(
                "instance_*"
            )
            if path.is_dir()
        )

        if not instance_dirs:

            raise FileNotFoundError(
                f"No instance directories found in:\n"
                f"  {split_dir}"
            )

        instance_names = [
            path.name
            for path in instance_dirs
        ]

        print()

        print(
            "=" * 70
        )

        print(
            "Visualize DeepD3 Dataset"
        )

        print(
            "=" * 70
        )

        print(
            f"Split     : "
            f"{args.split}"
        )

        print(
            f"Instances : "
            f"{len(instance_names)}"
        )

        print(
            f"Models    : "
            f"{', '.join(selected_models)}"
        )

        print(
            "=" * 70
        )

    else:

        instance_names = [
            args.instance
        ]

    # ------------------------------------------------------
    # Process selected instances
    # ------------------------------------------------------

    total_instances = len(
        instance_names
    )

    for (
        index,
        instance_name,
    ) in enumerate(
        instance_names,
        start=1,
    ):

        if total_instances > 1:

            print()

            print(
                "#" * 70
            )

            print(
                f"Processing instance "
                f"{index}/{total_instances}: "
                f"{instance_name}"
            )

            print(
                "#" * 70
            )

        process_instance(
            dataset_root=dataset_root,
            split=args.split,
            instance_name=instance_name,
            selected_models=selected_models,
            thresholds=thresholds,
            threshold_path=threshold_path,
        )

    print()

    print(
        "=" * 70
    )

    print(
        "Visualization complete"
    )

    print(
        f"Processed "
        f"{total_instances} instance(s)."
    )

    print(
        f"Split: "
        f"{args.split}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()