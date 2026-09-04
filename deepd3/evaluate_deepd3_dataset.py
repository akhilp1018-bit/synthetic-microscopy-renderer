"""
evaluate_deepd3_dataset.py
--------------------------

Evaluate pretrained DeepD3 predictions on the synthetic microscopy dataset.

VALIDATION:
    - sweep probability thresholds
    - select dendrite segmentation threshold
    - select spine segmentation threshold
    - select spine detection threshold
    - save selected thresholds

TEST:
    - load thresholds selected on validation
    - do NOT optimize thresholds on test
    - calculate final segmentation and detection metrics

Metrics:
    Dendrite segmentation:
        IoU, Dice

    Spine segmentation:
        IoU, Dice

    Spine detection:
        Precision, Recall, F1
        center-to-center physical distance
        recall versus matching distance

Usage:
    python deepd3/evaluate_deepd3_dataset.py --split validation
    python deepd3/evaluate_deepd3_dataset.py --split test
"""

from __future__ import annotations

import argparse
import csv
import json
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

# Z, Y, X spacing in nm
DEFAULT_SPACING_NM_ZYX = (
    500.0,
    94.0,
    94.0,
)

PEAK_NEIGHBORHOOD_ZYX = (
    5,
    9,
    9,
)

PEAK_SMOOTH_SIGMA = 1.0

DEFAULT_MATCH_DISTANCE_NM = 1000.0

# Do not include 0.0.
THRESHOLDS = np.arange(
    0.02,
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
    )

    parser.add_argument(
        "--split",
        choices=(
            "validation",
            "test",
        ),
        default="test",
    )

    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--match-distance-nm",
        type=float,
        default=DEFAULT_MATCH_DISTANCE_NM,
    )

    return parser.parse_args()


# ==========================================================
# File helpers
# ==========================================================

def require_file(path: Path, description: str):

    if not path.is_file():

        raise FileNotFoundError(
            f"{description} not found:\n"
            f"  {path}"
        )


def load_binary_mask(path: Path):

    arr = tifffile.imread(path)

    return np.asarray(arr) > 0


def load_prediction(path: Path):

    data = fl.load(str(path))

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

    ground_truth = ground_truth.astype(bool)
    prediction = prediction.astype(bool)

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

    ground_truth = ground_truth.astype(bool)
    prediction = prediction.astype(bool)

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

def get_gt_spine_centers(spine_mask):

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
        range(1, count + 1),
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
# Physical center distances
# ==========================================================

def pairwise_distances_nm(
    gt_centers,
    predicted_centers,
    spacing_nm_zyx,
):

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

    gt_count = len(gt_centers)
    pred_count = len(predicted_centers)

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

    for gt_index in range(gt_count):

        for pred_index in range(pred_count):

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

        used_gt.add(gt_index)
        used_pred.add(pred_index)

        matches.append(
            (
                gt_index,
                pred_index,
                distance,
            )
        )

    tp = len(matches)

    fp = (
        pred_count
        - tp
    )

    fn = (
        gt_count
        - tp
    )

    return (
        matches,
        tp,
        fp,
        fn,
    )


def detection_metrics(
    tp,
    fp,
    fn,
):

    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    if tp + fn > 0:
        recall = tp / (tp + fn)
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
# CSV helper
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
        writer.writerows(rows)


# ==========================================================
# Load instance data
# ==========================================================

def load_model_instances(
    instance_dirs,
    model_tag,
    prediction_filename,
):

    model_instance_data = []

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

        expected_shape = spine_gt.shape

        if dendrite_probability.shape != expected_shape:

            raise ValueError(
                f"Shape mismatch: "
                f"{instance_dir.name} "
                f"{model_tag} dendrite\n"
                f"GT: {expected_shape}\n"
                f"Prediction: "
                f"{dendrite_probability.shape}"
            )

        if spine_probability.shape != expected_shape:

            raise ValueError(
                f"Shape mismatch: "
                f"{instance_dir.name} "
                f"{model_tag} spine\n"
                f"GT: {expected_shape}\n"
                f"Prediction: "
                f"{spine_probability.shape}"
            )

        gt_centers = get_gt_spine_centers(
            spine_gt
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
            or index == len(instance_dirs)
        ):

            print(
                f"  Loaded "
                f"{index}/"
                f"{len(instance_dirs)}"
            )

    return model_instance_data


# ==========================================================
# Validation threshold sweep
# ==========================================================

def threshold_sweep(
    model_tag,
    model_instance_data,
    match_distance_nm,
    spacing_nm_zyx,
):

    rows = []

    for threshold in THRESHOLDS:

        total_tp = 0
        total_fp = 0
        total_fn = 0

        dendrite_ious = []
        dendrite_dices = []

        spine_ious = []
        spine_dices = []

        for data in model_instance_data:

            dendrite_prediction = (
                data["dendrite_probability"]
                >= threshold
            )

            spine_prediction = (
                data["spine_probability"]
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
                    data["spine_probability"],
                    threshold,
                )
            )

            (
                _,
                tp,
                fp,
                fn,
            ) = greedy_match(
                data["gt_centers"],
                predicted_centers,
                match_distance_nm,
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

        rows.append(
            {
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
        )

    return rows


# ==========================================================
# Threshold selection
# ==========================================================

def select_validation_thresholds(
    threshold_rows,
):

    # Segmentation thresholds selected using IoU.
    dendrite_best = max(
        threshold_rows,
        key=lambda row:
            row["dendrite_iou"],
    )

    spine_segmentation_best = max(
        threshold_rows,
        key=lambda row:
            row["spine_iou"],
    )

    # Object-detection threshold selected using F1.
    spine_detection_best = max(
        threshold_rows,
        key=lambda row:
            row["spine_f1"],
    )

    return {
        "dendrite_segmentation":
            float(
                dendrite_best[
                    "threshold"
                ]
            ),

        "spine_segmentation":
            float(
                spine_segmentation_best[
                    "threshold"
                ]
            ),

        "spine_detection":
            float(
                spine_detection_best[
                    "threshold"
                ]
            ),
    }


# ==========================================================
# Evaluate fixed thresholds
# ==========================================================

def evaluate_fixed_thresholds(
    model_tag,
    model_instance_data,
    thresholds,
    match_distance_nm,
    spacing_nm_zyx,
):

    per_instance_rows = []
    matched_distance_rows = []

    dendrite_threshold = (
        thresholds[
            "dendrite_segmentation"
        ]
    )

    spine_segmentation_threshold = (
        thresholds[
            "spine_segmentation"
        ]
    )

    spine_detection_threshold = (
        thresholds[
            "spine_detection"
        ]
    )

    total_tp = 0
    total_fp = 0
    total_fn = 0

    all_dendrite_iou = []
    all_dendrite_dice = []

    all_spine_iou = []
    all_spine_dice = []

    for data in model_instance_data:

        dendrite_prediction = (
            data["dendrite_probability"]
            >= dendrite_threshold
        )

        spine_prediction = (
            data["spine_probability"]
            >= spine_segmentation_threshold
        )

        predicted_centers = (
            detect_spine_peaks(
                data["spine_probability"],
                spine_detection_threshold,
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
            match_distance_nm,
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

        dendrite_iou = compute_iou(
            data["dendrite_gt"],
            dendrite_prediction,
        )

        dendrite_dice = compute_dice(
            data["dendrite_gt"],
            dendrite_prediction,
        )

        spine_iou = compute_iou(
            data["spine_gt"],
            spine_prediction,
        )

        spine_dice = compute_dice(
            data["spine_gt"],
            spine_prediction,
        )

        all_dendrite_iou.append(
            dendrite_iou
        )

        all_dendrite_dice.append(
            dendrite_dice
        )

        all_spine_iou.append(
            spine_iou
        )

        all_spine_dice.append(
            spine_dice
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        matched_distances = [
            float(match[2])
            for match in matches
        ]

        for (
            gt_index,
            pred_index,
            distance_nm,
        ) in matches:

            matched_distance_rows.append(
                {
                    "model":
                        model_tag,

                    "instance":
                        data["instance"],

                    "gt_id":
                        int(gt_index),

                    "pred_id":
                        int(pred_index),

                    "distance_nm":
                        float(distance_nm),
                }
            )

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

                "dendrite_threshold":
                    dendrite_threshold,

                "spine_segmentation_threshold":
                    spine_segmentation_threshold,

                "spine_detection_threshold":
                    spine_detection_threshold,

                "gt_spines":
                    len(
                        data["gt_centers"]
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
                    dendrite_iou,

                "dendrite_dice":
                    dendrite_dice,

                "spine_iou":
                    spine_iou,

                "spine_dice":
                    spine_dice,
            }
        )

    (
        overall_precision,
        overall_recall,
        overall_f1,
    ) = detection_metrics(
        total_tp,
        total_fp,
        total_fn,
    )

    summary = {
        "model":
            model_tag,

        "dendrite_threshold":
            dendrite_threshold,

        "spine_segmentation_threshold":
            spine_segmentation_threshold,

        "spine_detection_threshold":
            spine_detection_threshold,

        "dendrite_iou":
            float(
                np.mean(
                    all_dendrite_iou
                )
            ),

        "dendrite_dice":
            float(
                np.mean(
                    all_dendrite_dice
                )
            ),

        "spine_iou":
            float(
                np.mean(
                    all_spine_iou
                )
            ),

        "spine_dice":
            float(
                np.mean(
                    all_spine_dice
                )
            ),

        "spine_precision":
            overall_precision,

        "spine_recall":
            overall_recall,

        "spine_f1":
            overall_f1,

        "tp":
            total_tp,

        "fp":
            total_fp,

        "fn":
            total_fn,
    }

    return (
        per_instance_rows,
        matched_distance_rows,
        summary,
    )


# ==========================================================
# Recall versus matching distance
# ==========================================================

def calculate_recall_vs_distance(
    model_tag,
    model_instance_data,
    detection_threshold,
    spacing_nm_zyx,
):

    rows = []

    # Peak locations do not change with matching distance,
    # so calculate them once.
    instance_centers = []

    for data in model_instance_data:

        predicted_centers = detect_spine_peaks(
            data["spine_probability"],
            detection_threshold,
        )

        instance_centers.append(
            (
                data["gt_centers"],
                predicted_centers,
            )
        )

    for distance_nm in MATCH_DISTANCES_NM:

        total_tp = 0
        total_fn = 0

        for (
            gt_centers,
            predicted_centers,
        ) in instance_centers:

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

        if denominator > 0:
            recall = (
                total_tp
                / denominator
            )
        else:
            recall = 0.0

        rows.append(
            {
                "model":
                    model_tag,

                "threshold":
                    detection_threshold,

                "matching_distance_nm":
                    float(distance_nm),

                "recall":
                    float(recall),
            }
        )

    return rows


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

        instance_dirs = (
            instance_dirs[
                :args.max_instances
            ]
        )

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

    validation_output_dir = (
        dataset_root
        / "deepd3_evaluation"
        / "validation"
    )

    threshold_file = (
        validation_output_dir
        / "selected_thresholds.json"
    )

    spacing_nm_zyx = (
        DEFAULT_SPACING_NM_ZYX
    )

    print()
    print("=" * 70)
    print(
        "DeepD3 synthetic dataset evaluation"
    )
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
    # Validation/test threshold logic
    # ------------------------------------------------------

    selected_thresholds = {}

    if args.split == "test":

        require_file(
            threshold_file,
            "Validation threshold file",
        )

        with open(
            threshold_file,
            "r",
            encoding="utf-8",
        ) as file:

            selected_thresholds = (
                json.load(file)
            )

        print()
        print(
            "Using frozen thresholds "
            "selected on validation."
        )

    all_per_instance_rows = []
    all_threshold_rows = []
    all_distance_rows = []
    all_matched_distance_rows = []
    final_summary_rows = []

    # ------------------------------------------------------
    # Models
    # ------------------------------------------------------

    for (
        model_tag,
        prediction_filename,
    ) in MODEL_FILES.items():

        print()
        print(
            f"Model: {model_tag}"
        )

        model_instance_data = (
            load_model_instances(
                instance_dirs,
                model_tag,
                prediction_filename,
            )
        )

        # --------------------------------------------------
        # Validation: sweep and select thresholds
        # --------------------------------------------------

        if args.split == "validation":

            model_threshold_rows = (
                threshold_sweep(
                    model_tag,
                    model_instance_data,
                    args.match_distance_nm,
                    spacing_nm_zyx,
                )
            )

            all_threshold_rows.extend(
                model_threshold_rows
            )

            model_thresholds = (
                select_validation_thresholds(
                    model_threshold_rows
                )
            )

            selected_thresholds[
                model_tag
            ] = model_thresholds

        # --------------------------------------------------
        # Test: use frozen validation thresholds
        # --------------------------------------------------

        else:

            if model_tag not in selected_thresholds:

                raise KeyError(
                    f"{model_tag} missing from "
                    f"{threshold_file}"
                )

            model_thresholds = (
                selected_thresholds[
                    model_tag
                ]
            )

        print(
            "  Dendrite segmentation threshold: "
            f"{model_thresholds['dendrite_segmentation']:.2f}"
        )

        print(
            "  Spine segmentation threshold   : "
            f"{model_thresholds['spine_segmentation']:.2f}"
        )

        print(
            "  Spine detection threshold      : "
            f"{model_thresholds['spine_detection']:.2f}"
        )

        # --------------------------------------------------
        # Evaluate selected/frozen thresholds
        # --------------------------------------------------

        (
            per_instance_rows,
            matched_distance_rows,
            summary,
        ) = evaluate_fixed_thresholds(
            model_tag,
            model_instance_data,
            model_thresholds,
            args.match_distance_nm,
            spacing_nm_zyx,
        )

        all_per_instance_rows.extend(
            per_instance_rows
        )

        all_matched_distance_rows.extend(
            matched_distance_rows
        )

        final_summary_rows.append(
            summary
        )

        print(
            f"  Dendrite IoU : "
            f"{summary['dendrite_iou']:.4f}"
        )

        print(
            f"  Dendrite Dice: "
            f"{summary['dendrite_dice']:.4f}"
        )

        print(
            f"  Spine IoU    : "
            f"{summary['spine_iou']:.4f}"
        )

        print(
            f"  Spine Dice   : "
            f"{summary['spine_dice']:.4f}"
        )

        print(
            f"  Precision    : "
            f"{summary['spine_precision']:.4f}"
        )

        print(
            f"  Recall       : "
            f"{summary['spine_recall']:.4f}"
        )

        print(
            f"  F1           : "
            f"{summary['spine_f1']:.4f}"
        )

        # --------------------------------------------------
        # Recall versus distance
        # --------------------------------------------------

        distance_rows = (
            calculate_recall_vs_distance(
                model_tag,
                model_instance_data,
                model_thresholds[
                    "spine_detection"
                ],
                spacing_nm_zyx,
            )
        )

        all_distance_rows.extend(
            distance_rows
        )

    # ------------------------------------------------------
    # Save selected validation thresholds
    # ------------------------------------------------------

    if args.split == "validation":

        with open(
            threshold_file,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                selected_thresholds,
                file,
                indent=2,
            )

        print()
        print(
            "Saved validation thresholds:"
        )

        print(
            f"  {threshold_file}"
        )

    # ------------------------------------------------------
    # Save per-instance metrics
    # ------------------------------------------------------

    save_csv(
        output_dir
        / "per_instance_metrics.csv",
        all_per_instance_rows,
        [
            "model",
            "instance",
            "dendrite_threshold",
            "spine_segmentation_threshold",
            "spine_detection_threshold",
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

    # ------------------------------------------------------
    # Validation threshold sweep / PR data
    # ------------------------------------------------------

    if args.split == "validation":

        save_csv(
            output_dir
            / "threshold_summary.csv",
            all_threshold_rows,
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

    # ------------------------------------------------------
    # Final summary
    # ------------------------------------------------------

    save_csv(
        output_dir
        / "final_metrics.csv",
        final_summary_rows,
        [
            "model",
            "dendrite_threshold",
            "spine_segmentation_threshold",
            "spine_detection_threshold",
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

    # ------------------------------------------------------
    # Recall versus distance
    # ------------------------------------------------------

    save_csv(
        output_dir
        / "recall_vs_distance.csv",
        all_distance_rows,
        [
            "model",
            "threshold",
            "matching_distance_nm",
            "recall",
        ],
    )

    # ------------------------------------------------------
    # Individual matched center distances
    # ------------------------------------------------------

    save_csv(
        output_dir
        / "matched_center_distances.csv",
        all_matched_distance_rows,
        [
            "model",
            "instance",
            "gt_id",
            "pred_id",
            "distance_nm",
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

    if args.split == "validation":
        print(
            "threshold_summary.csv"
        )

    print(
        "final_metrics.csv"
    )

    print(
        "recall_vs_distance.csv"
    )

    print(
        "matched_center_distances.csv"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()