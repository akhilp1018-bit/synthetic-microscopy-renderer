"""
Visualize DeepD3 Dataset
------------------------

Create qualitative DeepD3 spine-detection visualizations for
synthetic dataset instances.

For each model, the figure contains:

    1. Synthetic noisy image - XY maximum projection
    2. Ground-truth spine mask - XY maximum projection
    3. DeepD3 spine probability - XY maximum projection
    4. GT + predicted spine centers with matching

The visualization uses the SAME object-level evaluation settings
as the quantitative DeepD3 evaluation:

    XY spacing = 94 nm
    Z spacing  = 500 nm

    Gaussian smoothing sigma = 1
    Local-max neighborhood   = (5, 9, 9) ZYX
    Matching distance        = 1000 nm

DeepD3 detection thresholds are loaded from the VALIDATION set:

    outputs/synthetic_dataset_v1/
        deepd3_evaluation/
        validation/
        selected_thresholds.json

Therefore TEST visualization uses thresholds selected on
VALIDATION and does not tune thresholds on the test set.


Usage
-----

Visualize one test instance:

    python deepd3/visualize_deepd3_dataset.py \
        --split test \
        --instance instance_000001

Visualize ALL test instances:

    python deepd3/visualize_deepd3_dataset.py \
        --split test \
        --instance all

Only 32F:

    python deepd3/visualize_deepd3_dataset.py \
        --split test \
        --instance all \
        --model 32F

Only 32F_94nm:

    python deepd3/visualize_deepd3_dataset.py \
        --split test \
        --instance all \
        --model 32F_94nm
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
)


# ==========================================================
# Command-line arguments
# ==========================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Visualize DeepD3 spine-center matching "
            "for synthetic dataset instances."
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
            "both",
        ),
        default="both",
        help=(
            "DeepD3 model to visualize. "
            "Default: both."
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


def normalize01(arr):

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

def load_spine_probability(
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

    probability = np.asarray(
        data["spines"],
        dtype=np.float32,
    )

    probability = np.nan_to_num(
        probability,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    return probability


# ==========================================================
# Load frozen validation thresholds
# ==========================================================

def extract_detection_threshold(
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


def load_detection_thresholds(
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

        thresholds[model] = (
            extract_detection_threshold(
                data[model]
            )
        )

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

    distances = (
        calculate_distance_matrix_nm(
            gt_centers,
            pred_centers,
        )
    )

    candidates = []

    for gt_index in range(
        distances.shape[0]
    ):

        for pred_index in range(
            distances.shape[1]
        ):

            distance = (
                distances[
                    gt_index,
                    pred_index,
                ]
            )

            if (
                distance
                <= MATCH_DISTANCE_NM
            ):

                candidates.append(
                    (
                        float(distance),
                        gt_index,
                        pred_index,
                    )
                )

    candidates.sort(
        key=lambda item: item[0]
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
# Detection metrics
# ==========================================================

def calculate_metrics(
    gt_centers,
    pred_centers,
    matches,
):

    tp = len(
        matches
    )

    fp = (
        len(pred_centers)
        - tp
    )

    fn = (
        len(gt_centers)
        - tp
    )

    precision = (
        tp
        / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp
        / (tp + fn)
        if (tp + fn) > 0
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
# Visualization
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

    # ------------------------------------------------------
    # XY maximum projections
    # ------------------------------------------------------

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


    # ------------------------------------------------------
    # Determine matched / unmatched centers
    # ------------------------------------------------------

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
            len(gt_centers)
        )
        if index not in matched_gt
    ]

    false_positive_pred = [
        index
        for index in range(
            len(pred_centers)
        )
        if index not in matched_pred
    ]


    # ------------------------------------------------------
    # Figure
    # ------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(
            18,
            5,
        ),
    )


    # Panel 1: synthetic image
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


    # Panel 2: GT spine mask
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


    # Panel 3: DeepD3 probability
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


    # Panel 4: matching overlay
    axes[3].imshow(
        noisy_mip,
        cmap="gray",
        vmin=0,
        vmax=1,
    )


    # GT centers
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


    # Matched predictions
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


    # False-positive predictions
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


    # Missed GT
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


    # Matching lines
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


    # Remove axes
    for ax in axes:

        ax.axis(
            "off"
        )


    # Figure title + metrics
    fig.suptitle(
        (
            f"{split}/{instance} — DeepD3 {model}\n"
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


    # Save
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


    # ------------------------------------------------------
    # Input files
    # ------------------------------------------------------

    noisy_path = (
        instance_dir
        / "noisy.tif"
    )

    spine_gt_path = (
        instance_dir
        / "spine_mask.tif"
    )


    require_file(
        noisy_path,
        "Noisy synthetic image",
    )

    require_file(
        spine_gt_path,
        "GT spine mask",
    )


    # ------------------------------------------------------
    # Load image + GT
    # ------------------------------------------------------

    noisy = tifffile.imread(
        noisy_path
    )

    spine_gt = (
        tifffile.imread(
            spine_gt_path
        )
        > 0
    )


    if (
        noisy.shape
        != spine_gt.shape
    ):

        raise ValueError(
            "Image / GT shape mismatch.\n"
            f"Noisy image : {noisy.shape}\n"
            f"Spine GT    : {spine_gt.shape}"
        )


    # ------------------------------------------------------
    # Extract GT centers
    # ------------------------------------------------------

    gt_centers = (
        extract_gt_centers(
            spine_gt
        )
    )


    # ------------------------------------------------------
    # Output directory
    # ------------------------------------------------------

    output_dir = (
        instance_dir
        / "deepd3_visualization"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ------------------------------------------------------
    # Console header
    # ------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "DeepD3 spine matching visualization"
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
        f"Spacing    : {tuple(SPACING_ZYX_NM)} nm (Z,Y,X)"
    )

    print(
        f"Match      : {MATCH_DISTANCE_NM:.0f} nm"
    )

    print(
        f"Thresholds : {threshold_path}"
    )

    print(
        "=" * 70
    )


    # ------------------------------------------------------
    # Process models
    # ------------------------------------------------------

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


        spine_probability = (
            load_spine_probability(
                prediction_path
            )
        )


        if (
            spine_probability.shape
            != noisy.shape
        ):

            raise ValueError(
                f"{model} prediction shape mismatch.\n"
                f"Image      : {noisy.shape}\n"
                f"Prediction : {spine_probability.shape}"
            )


        # Frozen validation threshold
        threshold = (
            thresholds[
                model
            ]
        )


        # Detect predicted centers
        pred_centers = (
            detect_predicted_centers(
                spine_probability,
                threshold,
            )
        )


        # GT-to-prediction matching
        matches = (
            match_centers(
                gt_centers,
                pred_centers,
            )
        )


        # Metrics
        metrics = (
            calculate_metrics(
                gt_centers,
                pred_centers,
                matches,
            )
        )


        # Save figure
        output_path = (
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
            metrics=metrics,
            threshold=threshold,
            model=model,
            split=split,
            instance=instance_name,
            output_path=output_path,
        )


        # Terminal output
        print()

        print(
            f"Model      : {model}"
        )

        print(
            f"Threshold  : {threshold:.2f}"
        )

        print(
            f"GT centers : {len(gt_centers)}"
        )

        print(
            f"Predicted  : {len(pred_centers)}"
        )

        print(
            "TP / FP / FN: "
            f"{metrics['tp']} / "
            f"{metrics['fp']} / "
            f"{metrics['fn']}"
        )

        print(
            f"Precision  : "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Recall     : "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"F1         : "
            f"{metrics['f1']:.4f}"
        )

        print(
            f"Saved      : {output_path}"
        )


# ==========================================================
# Main
# ==========================================================

def main():

    args = parse_args()

    dataset_root = (
        args.dataset.resolve()
    )


    # ------------------------------------------------------
    # Load frozen validation thresholds once
    # ------------------------------------------------------

    (
        thresholds,
        threshold_path,
    ) = load_detection_thresholds(
        dataset_root
    )


    # ------------------------------------------------------
    # Select model(s)
    # ------------------------------------------------------

    if args.model == "both":

        selected_models = (
            "32F",
            "32F_94nm",
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


    if args.instance.lower() == "all":

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
            f"Split     : {args.split}"
        )

        print(
            f"Instances : {len(instance_names)}"
        )

        print(
            f"Models    : {', '.join(selected_models)}"
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


    for index, instance_name in enumerate(
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


    # ------------------------------------------------------
    # Final summary
    # ------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "Visualization complete"
    )

    print(
        f"Processed {total_instances} "
        f"instance(s)."
    )

    print(
        f"Split: {args.split}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()