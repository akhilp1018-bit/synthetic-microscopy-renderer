"""
evaluate_deepd3_dataset.py
--------------------------

Evaluate pretrained DeepD3 predictions on the synthetic microscopy dataset.

Evaluation includes:

1. Dendrite segmentation
   - IoU
   - Dice

2. Spine segmentation
   - IoU
   - Dice

3. Spine detection
   - GT spine centers from 3D connected components
   - predicted spine centers from 3D probability-map peaks
   - one-to-one physical-distance matching
   - precision
   - recall
   - F1 score
   - center-to-center matching distances

4. Recall versus allowed matching distance

Expected instance structure
---------------------------

instance_XXXXXX/
├── dendrite_mask.tif
├── spine_mask.tif
└── deepd3_predictions/
    ├── 32F.prediction
    └── 32F_94nm.prediction

Usage
-----

Evaluate the complete test split:

    python deepd3/evaluate_deepd3_dataset.py

Evaluate only one instance:

    python deepd3/evaluate_deepd3_dataset.py --max-instances 1

Use another split:

    python deepd3/evaluate_deepd3_dataset.py --split validation

Outputs
-------

outputs/synthetic_dataset_v1/deepd3_evaluation/

    per_instance_metrics.csv
    threshold_summary.csv
    recall_vs_distance.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import flammkuchen as fl
import numpy as np
import tifffile

from scipy.ndimage import (
    center_of_mass,
    gaussian_filter,
    label,
    maximum_filter,
)


# ==========================================================
# Defaults
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET = (
    PROJECT_ROOT
    / "outputs"
    / "synthetic_dataset_v1"
)

MODEL_FILES = {
    "32F": "32F.prediction",
    "32F_94nm": "32F_94nm.prediction",
}

# Dataset voxel spacing: Z, Y, X
DEFAULT_SPACING_NM_ZYX = (
    500.0,
    94.0,
    94.0,
)

# Same basic peak-detection settings used in the
# previous DeepD3 spine evaluation.
PEAK_NEIGHBORHOOD_ZYX = (
    5,
    9,
    9,
)

PEAK_SMOOTH_SIGMA = 1.0

DEFAULT_MATCH_DISTANCE_NM = 1000.0

THRESHOLDS = np.arange(
    0.0,
    1.0001,
    0.02,
)

MATCH_DISTANCES_NM = np.arange(
    0.0,
    20000.0 + 1.0,
    500.0,
)


# ==========================================================
# Arguments
# ==========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate DeepD3 predictions against "
            "synthetic ground truth."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Synthetic dataset root.",
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
        "--max-instances",
        type=int,
        default=None,
        help="Optional number of instances to evaluate.",
    )

    parser.add_argument(
        "--match-distance-nm",
        type=float,
        default=DEFAULT_MATCH_DISTANCE_NM,
        help=(
            "Maximum GT/prediction center distance "
            "for a correct spine detection."
        ),
    )

    return parser.parse_args()


# ==========================================================
# File helpers
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


def load_binary_mask(path: Path):
    arr = tifffile.imread(
        path
    )

    return (
        np.asarray(arr) > 0
    )


def load_prediction(path: Path):
    """
    Load dendrite and spine probability volumes
    from a DeepD3 prediction file.
    """

    data = fl.load(
        str(path)
    )

    if "dendrites" not in data:
        raise KeyError(
            f"'dendrites' missing from {path}"
        )

    if "spines" not in data:
        raise KeyError(
            f"'spines' missing from {path}"
        )

    dendrites = np.asarray(
        data["dendrites"],
        dtype=np.float32,
    )

    spines = np.asarray(
        data["spines"],
        dtype=np.float32,
    )

    dendrites = np.nan_to_num(
        dendrites,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    spines = np.nan_to_num(
        spines,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    return dendrites, spines


# ==========================================================
# Segmentation metrics
# ==========================================================

def compute_iou(
    ground_truth,
    prediction,
):
    ground_truth = ground_truth.astype(
        bool
    )

    prediction = prediction.astype(
        bool
    )

    intersection = np.logical_and(
        ground_truth,
        prediction,
    ).sum()

    union = np.logical_or(
        ground_truth,
        prediction,
    ).sum()

    if union == 0:
        return 1.0

    return float(
        intersection / union
    )


def compute_dice(
    ground_truth,
    prediction,
):
    ground_truth = ground_truth.astype(
        bool
    )

    prediction = prediction.astype(
        bool
    )

    intersection = np.logical_and(
        ground_truth,
        prediction,
    ).sum()

    denominator = (
        ground_truth.sum()
        + prediction.sum()
    )

    if denominator == 0:
        return 1.0

    return float(
        2.0
        * intersection
        / denominator
    )


# ==========================================================
# GT spine centers
# ==========================================================

def get_gt_spine_centers(
    spine_mask,
):
    """
    Extract GT spine centers from connected components
    of the binary synthetic spine mask.

    Returns centers in Z, Y, X voxel coordinates.
    """

    labelled_mask, count = label(
        spine_mask
    )

    if count == 0:
        return np.empty(
            (0, 3),
            dtype=np.float32,
        )

    centers = center_of_mass(
        spine_mask,
        labelled_mask,
        range(
            1,
            count + 1,
        ),
    )

    return np.asarray(
        centers,
        dtype=np.float32,
    )


# ==========================================================
# Predicted spine centers
# ==========================================================

def detect_spine_peaks(
    probability,
    threshold,
):
    """
    Detect 3D local maxima in the DeepD3 spine
    probability volume.

    Returns peak coordinates in Z, Y, X.
    """

    smoothed = gaussian_filter(
        probability,
        sigma=PEAK_SMOOTH_SIGMA,
    )

    local_maximum = (
        smoothed
        == maximum_filter(
            smoothed,
            size=PEAK_NEIGHBORHOOD_ZYX,
            mode="nearest",
        )
    )

    above_threshold = (
        smoothed >= threshold
    )

    peak_mask = np.logical_and(
        local_maximum,
        above_threshold,
    )

    coordinates = np.argwhere(
        peak_mask
    )

    if coordinates.size == 0:
        return np.empty(
            (0, 3),
            dtype=np.float32,
        )

    return coordinates.astype(
        np.float32
    )


# ==========================================================
# Center-distance matching
# ==========================================================

def pairwise_distances_nm(
    gt_centers,
    predicted_centers,
    spacing_nm_zyx,
):
    """
    Compute physical pairwise distances between GT
    and predicted centers.

    Coordinates are Z, Y, X.
    """

    if (
        len(gt_centers) == 0
        or len(predicted_centers) == 0
    ):
        return np.empty(
            (
                len(gt_centers),
                len(predicted_centers),
            ),
            dtype=np.float32,
        )

    spacing = np.asarray(
        spacing_nm_zyx,
        dtype=np.float32,
    )

    difference = (
        gt_centers[:, None, :]
        - predicted_centers[None, :, :]
    )

    difference_nm = (
        difference
        * spacing[None, None, :]
    )

    distances = np.sqrt(
        np.sum(
            difference_nm ** 2,
            axis=2,
        )
    )

    return distances


def greedy_match(
    gt_centers,
    predicted_centers,
    max_distance_nm,
    spacing_nm_zyx,
):
    """
    Greedy one-to-one matching of GT and predicted
    spine centers.

    The globally shortest available pair is matched first.
    """

    gt_count = len(
        gt_centers
    )

    pred_count = len(
        predicted_centers
    )

    if gt_count == 0 or pred_count == 0:
        return (
            [],
            0,
            pred_count,
            gt_count,
        )

    distances = pairwise_distances_nm(
        gt_centers,
        predicted_centers,
        spacing_nm_zyx,
    )

    candidates = []

    for gt_index in range(
        gt_count
    ):
        for pred_index in range(
            pred_count
        ):
            distance = float(
                distances[
                    gt_index,
                    pred_index,
                ]
            )

            if distance <= max_distance_nm:
                candidates.append(
                    (
                        distance,
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

    true_positive = len(
        matches
    )

    false_positive = (
        pred_count
        - true_positive
    )

    false_negative = (
        gt_count
        - true_positive
    )

    return (
        matches,
        true_positive,
        false_positive,
        false_negative,
    )


def detection_metrics(
    true_positive,
    false_positive,
    false_negative,
):
    precision_denominator = (
        true_positive
        + false_positive
    )

    recall_denominator = (
        true_positive
        + false_negative
    )

    if precision_denominator > 0:
        precision = (
            true_positive
            / precision_denominator
        )
    else:
        precision = 0.0

    if recall_denominator > 0:
        recall = (
            true_positive
            / recall_denominator
        )
    else:
        recall = 0.0

    if precision + recall > 0:
        f1 = (
            2.0
            * precision
            * recall
            / (
                precision
                + recall
            )
        )
    else:
        f1 = 0.0

    return (
        float(precision),
        float(recall),
        float(f1),
    )


# ==========================================================
# CSV
# ==========================================================

def save_csv(
    path,
    rows,
    fieldnames,
):
    with open(
        path,
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
# Main
# ==========================================================

def main():
    args = parse_args()

    dataset_root = (
        args.dataset.resolve()
    )

    split_dir = (
        dataset_root
        / args.split
    )

    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset split not found:\n"
            f"  {split_dir}"
        )

    instance_dirs = sorted(
        path
        for path in split_dir.glob(
            "instance_*"
        )
        if path.is_dir()
    )

    if args.max_instances is not None:
        if args.max_instances < 1:
            raise ValueError(
                "--max-instances must be >= 1"
            )

        instance_dirs = instance_dirs[
            :args.max_instances
        ]

    if not instance_dirs:
        raise RuntimeError(
            "No dataset instances found."
        )

    output_dir = (
        dataset_root
        / "deepd3_evaluation"
        / args.split
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    spacing_nm_zyx = (
        DEFAULT_SPACING_NM_ZYX
    )

    per_instance_rows = []
    threshold_rows = []

    print()
    print("=" * 70)
    print("DeepD3 synthetic dataset evaluation")
    print("=" * 70)

    print(
        f"Dataset   : {dataset_root}"
    )

    print(
        f"Split     : {args.split}"
    )

    print(
        f"Instances : {len(instance_dirs)}"
    )

    print(
        "Spacing   : "
        f"{spacing_nm_zyx} nm (Z,Y,X)"
    )

    print(
        "Match     : "
        f"{args.match_distance_nm:.0f} nm"
    )

    print("=" * 70)

    # ------------------------------------------------------
    # Process each model separately
    # ------------------------------------------------------

    for (
        model_tag,
        prediction_filename,
    ) in MODEL_FILES.items():

        print()
        print(
            f"Model: {model_tag}"
        )

        model_instance_data = []

        # --------------------------------------------------
        # Load all instances
        # --------------------------------------------------

        for index, instance_dir in enumerate(
            instance_dirs,
            start=1,
        ):

            dendrite_gt_path = (
                instance_dir
                / "dendrite_mask.tif"
            )

            spine_gt_path = (
                instance_dir
                / "spine_mask.tif"
            )

            prediction_path = (
                instance_dir
                / "deepd3_predictions"
                / prediction_filename
            )

            require_file(
                dendrite_gt_path,
                "Dendrite GT",
            )

            require_file(
                spine_gt_path,
                "Spine GT",
            )

            require_file(
                prediction_path,
                f"{model_tag} prediction",
            )

            dendrite_gt = load_binary_mask(
                dendrite_gt_path
            )

            spine_gt = load_binary_mask(
                spine_gt_path
            )

            (
                dendrite_probability,
                spine_probability,
            ) = load_prediction(
                prediction_path
            )

            expected_shape = (
                spine_gt.shape
            )

            if (
                dendrite_probability.shape
                != expected_shape
            ):
                raise ValueError(
                    f"Shape mismatch: "
                    f"{instance_dir.name} "
                    f"{model_tag} dendrite\n"
                    f"GT: {expected_shape}\n"
                    f"Prediction: "
                    f"{dendrite_probability.shape}"
                )

            if (
                spine_probability.shape
                != expected_shape
            ):
                raise ValueError(
                    f"Shape mismatch: "
                    f"{instance_dir.name} "
                    f"{model_tag} spine\n"
                    f"GT: {expected_shape}\n"
                    f"Prediction: "
                    f"{spine_probability.shape}"
                )

            gt_centers = (
                get_gt_spine_centers(
                    spine_gt
                )
            )

            model_instance_data.append(
                {
                    "instance":
                        instance_dir.name,

                    "dendrite_gt":
                        dendrite_gt,

                    "spine_gt":
                        spine_gt,

                    "dendrite_probability":
                        dendrite_probability,

                    "spine_probability":
                        spine_probability,

                    "gt_centers":
                        gt_centers,
                }
            )

            if (
                index % 10 == 0
                or index
                == len(instance_dirs)
            ):
                print(
                    f"  Loaded "
                    f"{index}/"
                    f"{len(instance_dirs)}"
                )

        # --------------------------------------------------
        # Threshold sweep
        # --------------------------------------------------

        model_threshold_results = []

        for threshold in THRESHOLDS:

            total_tp = 0
            total_fp = 0
            total_fn = 0

            dendrite_ious = []
            spine_ious = []

            dendrite_dices = []
            spine_dices = []

            for data in model_instance_data:

                dendrite_prediction = (
                    data[
                        "dendrite_probability"
                    ]
                    >= threshold
                )

                spine_prediction = (
                    data[
                        "spine_probability"
                    ]
                    >= threshold
                )

                dendrite_ious.append(
                    compute_iou(
                        data["dendrite_gt"],
                        dendrite_prediction,
                    )
                )

                dendrite_dices.append(
                    compute_dice(
                        data["dendrite_gt"],
                        dendrite_prediction,
                    )
                )

                spine_ious.append(
                    compute_iou(
                        data["spine_gt"],
                        spine_prediction,
                    )
                )

                spine_dices.append(
                    compute_dice(
                        data["spine_gt"],
                        spine_prediction,
                    )
                )

                predicted_centers = (
                    detect_spine_peaks(
                        data[
                            "spine_probability"
                        ],
                        threshold,
                    )
                )

                (
                    matches,
                    tp,
                    fp,
                    fn,
                ) = greedy_match(
                    data["gt_centers"],
                    predicted_centers,
                    args.match_distance_nm,
                    spacing_nm_zyx,
                )

                total_tp += tp
                total_fp += fp
                total_fn += fn

            (
                precision,
                recall,
                f1,
            ) = detection_metrics(
                total_tp,
                total_fp,
                total_fn,
            )

            row = {
                "model":
                    model_tag,

                "threshold":
                    float(threshold),

                "dendrite_iou":
                    float(
                        np.mean(
                            dendrite_ious
                        )
                    ),

                "dendrite_dice":
                    float(
                        np.mean(
                            dendrite_dices
                        )
                    ),

                "spine_iou":
                    float(
                        np.mean(
                            spine_ious
                        )
                    ),

                "spine_dice":
                    float(
                        np.mean(
                            spine_dices
                        )
                    ),

                "spine_precision":
                    precision,

                "spine_recall":
                    recall,

                "spine_f1":
                    f1,

                "tp":
                    total_tp,

                "fp":
                    total_fp,

                "fn":
                    total_fn,
            }

            threshold_rows.append(
                row
            )

            model_threshold_results.append(
                row
            )

        # --------------------------------------------------
        # Best object-detection threshold
        # --------------------------------------------------

        best_result = max(
            model_threshold_results,
            key=lambda row:
                row["spine_f1"],
        )

        best_threshold = float(
            best_result[
                "threshold"
            ]
        )

        print(
            f"  Best spine threshold: "
            f"{best_threshold:.2f}"
        )

        print(
            f"  Precision: "
            f"{best_result['spine_precision']:.4f}"
        )

        print(
            f"  Recall   : "
            f"{best_result['spine_recall']:.4f}"
        )

        print(
            f"  F1       : "
            f"{best_result['spine_f1']:.4f}"
        )

        # --------------------------------------------------
        # Per-instance metrics at best threshold
        # --------------------------------------------------

        for data in model_instance_data:

            dendrite_prediction = (
                data[
                    "dendrite_probability"
                ]
                >= best_threshold
            )

            spine_prediction = (
                data[
                    "spine_probability"
                ]
                >= best_threshold
            )

            predicted_centers = (
                detect_spine_peaks(
                    data[
                        "spine_probability"
                    ],
                    best_threshold,
                )
            )

            (
                matches,
                tp,
                fp,
                fn,
            ) = greedy_match(
                data["gt_centers"],
                predicted_centers,
                args.match_distance_nm,
                spacing_nm_zyx,
            )

            (
                precision,
                recall,
                f1,
            ) = detection_metrics(
                tp,
                fp,
                fn,
            )

            matched_distances = [
                match[2]
                for match in matches
            ]

            if matched_distances:
                mean_distance = float(
                    np.mean(
                        matched_distances
                    )
                )

                median_distance = float(
                    np.median(
                        matched_distances
                    )
                )
            else:
                mean_distance = np.nan
                median_distance = np.nan

            per_instance_rows.append(
                {
                    "model":
                        model_tag,

                    "instance":
                        data["instance"],

                    "threshold":
                        best_threshold,

                    "gt_spines":
                        len(
                            data[
                                "gt_centers"
                            ]
                        ),

                    "predicted_spines":
                        len(
                            predicted_centers
                        ),

                    "tp":
                        tp,

                    "fp":
                        fp,

                    "fn":
                        fn,

                    "precision":
                        precision,

                    "recall":
                        recall,

                    "f1":
                        f1,

                    "mean_match_distance_nm":
                        mean_distance,

                    "median_match_distance_nm":
                        median_distance,

                    "dendrite_iou":
                        compute_iou(
                            data[
                                "dendrite_gt"
                            ],
                            dendrite_prediction,
                        ),

                    "dendrite_dice":
                        compute_dice(
                            data[
                                "dendrite_gt"
                            ],
                            dendrite_prediction,
                        ),

                    "spine_iou":
                        compute_iou(
                            data[
                                "spine_gt"
                            ],
                            spine_prediction,
                        ),

                    "spine_dice":
                        compute_dice(
                            data[
                                "spine_gt"
                            ],
                            spine_prediction,
                        ),
                }
            )

    # ------------------------------------------------------
    # Recall versus matching distance
    # ------------------------------------------------------

    recall_distance_rows = []

    for (
        model_tag,
        prediction_filename,
    ) in MODEL_FILES.items():

        model_best_rows = [
            row
            for row in threshold_rows
            if row["model"] == model_tag
        ]

        best_threshold = max(
            model_best_rows,
            key=lambda row:
                row["spine_f1"],
        )["threshold"]

        for distance_nm in MATCH_DISTANCES_NM:

            total_tp = 0
            total_fn = 0

            for instance_dir in instance_dirs:

                spine_gt = load_binary_mask(
                    instance_dir
                    / "spine_mask.tif"
                )

                gt_centers = (
                    get_gt_spine_centers(
                        spine_gt
                    )
                )

                prediction_path = (
                    instance_dir
                    / "deepd3_predictions"
                    / prediction_filename
                )

                (
                    _,
                    spine_probability,
                ) = load_prediction(
                    prediction_path
                )

                predicted_centers = (
                    detect_spine_peaks(
                        spine_probability,
                        best_threshold,
                    )
                )

                (
                    _,
                    tp,
                    _,
                    fn,
                ) = greedy_match(
                    gt_centers,
                    predicted_centers,
                    distance_nm,
                    spacing_nm_zyx,
                )

                total_tp += tp
                total_fn += fn

            denominator = (
                total_tp
                + total_fn
            )

            recall = (
                total_tp
                / denominator
                if denominator > 0
                else 0.0
            )

            recall_distance_rows.append(
                {
                    "model":
                        model_tag,

                    "threshold":
                        best_threshold,

                    "matching_distance_nm":
                        float(
                            distance_nm
                        ),

                    "recall":
                        float(
                            recall
                        ),
                }
            )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    save_csv(
        output_dir
        / "per_instance_metrics.csv",
        per_instance_rows,
        [
            "model",
            "instance",
            "threshold",
            "gt_spines",
            "predicted_spines",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "mean_match_distance_nm",
            "median_match_distance_nm",
            "dendrite_iou",
            "dendrite_dice",
            "spine_iou",
            "spine_dice",
        ],
    )

    save_csv(
        output_dir
        / "threshold_summary.csv",
        threshold_rows,
        [
            "model",
            "threshold",
            "dendrite_iou",
            "dendrite_dice",
            "spine_iou",
            "spine_dice",
            "spine_precision",
            "spine_recall",
            "spine_f1",
            "tp",
            "fp",
            "fn",
        ],
    )

    save_csv(
        output_dir
        / "recall_vs_distance.csv",
        recall_distance_rows,
        [
            "model",
            "threshold",
            "matching_distance_nm",
            "recall",
        ],
    )

    print()
    print("=" * 70)
    print("Evaluation complete")
    print("=" * 70)

    print(
        f"Results: {output_dir}"
    )

    print()
    print(
        "per_instance_metrics.csv"
    )

    print(
        "threshold_summary.csv"
    )

    print(
        "recall_vs_distance.csv"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()