from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import tifffile
import flammkuchen as fl

from tqdm import tqdm

from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial.distance import cdist

from deepd3.core.dendrite import (
    DendriteSWC,
    xyzr,
    line_w_sphere,
)


# =========================================================
# Paths
# =========================================================

BENCHMARK_DIR = Path("benchmarks/deepd3")

DEFAULT_IMAGE = (
    BENCHMARK_DIR
    / "DeepD3_Benchmark.tif"
)

DEFAULT_REAL_PRED = (
    BENCHMARK_DIR
    / "predictions"
    / "real_32F_94nm.prediction"
)

DEFAULT_SYNTH_PRED = (
    BENCHMARK_DIR
    / "predictions"
    / "synthetic_32F_94nm.prediction"
)

DEFAULT_OUTPUT_DIR = (
    BENCHMARK_DIR
    / "evaluation"
)


# =========================================================
# Frozen segmentation thresholds
# =========================================================
#
# Selected previously on SYNTHETIC VALIDATION data.
#
# IMPORTANT:
# We do NOT tune these thresholds on the real benchmark.
# =========================================================

SEGMENTATION_THRESHOLDS = {

    "real_32F_94nm": {
        "dendrite": 0.01,
        "spine": 0.25,
    },

    "synthetic_32F_94nm": {
        "dendrite": 0.01,
        "spine": 0.54,
    },

}


# =========================================================
# Frozen spine-detection thresholds
# =========================================================
#
# Selected previously on SYNTHETIC VALIDATION data.
#
# Real-trained 32F_94nm:
#     detection threshold = 0.29
#
# Synthetic-trained:
#     detection threshold = 0.60
#
# These are frozen before real benchmark evaluation.
# =========================================================

DETECTION_THRESHOLDS = {

    "real_32F_94nm": 0.29,

    "synthetic_32F_94nm": 0.60,

}


# =========================================================
# Spine detection settings
# =========================================================
#
# Same settings as the synthetic-data evaluation.
# =========================================================

SPACING_NM_ZYX = np.array(
    [
        500.0,
        94.0,
        94.0,
    ],
    dtype=np.float64,
)

PEAK_NEIGHBORHOOD_ZYX = (
    5,
    9,
    9,
)

GAUSSIAN_SIGMA = 1.0

MATCH_DISTANCE_NM = 1000.0


# =========================================================
# Segmentation metrics
# =========================================================

def compute_iou(pred, gt):

    pred = np.asarray(
        pred,
        dtype=bool,
    )

    gt = np.asarray(
        gt,
        dtype=bool,
    )

    intersection = np.logical_and(
        pred,
        gt,
    ).sum()

    union = np.logical_or(
        pred,
        gt,
    ).sum()

    if union == 0:
        return np.nan

    return (
        intersection
        / union
    )


def compute_dice(pred, gt):

    pred = np.asarray(
        pred,
        dtype=bool,
    )

    gt = np.asarray(
        gt,
        dtype=bool,
    )

    intersection = np.logical_and(
        pred,
        gt,
    ).sum()

    total = (
        pred.sum()
        + gt.sum()
    )

    if total == 0:
        return np.nan

    return (
        2.0
        * intersection
        / total
    )


# =========================================================
# Prediction loading
# =========================================================

def load_prediction(path):

    print("\n========================================")
    print("Loading prediction")
    print("========================================")

    print(path)

    data = fl.load(
        path
    )

    if (
        isinstance(data, dict)
        and "prediction" in data
    ):

        prediction = np.asarray(
            data["prediction"]
        )

        dendrite = (
            prediction[..., 0]
        )

        spine = (
            prediction[..., 1]
        )

    elif (
        isinstance(data, dict)
        and "dendrites" in data
        and "spines" in data
    ):

        dendrite = np.asarray(
            data["dendrites"]
        )

        spine = np.asarray(
            data["spines"]
        )

    else:

        prediction = np.asarray(
            data
        )

        if (
            prediction.ndim != 4
            or prediction.shape[-1] < 2
        ):

            raise RuntimeError(
                "Could not understand prediction format: "
                f"{path}"
            )

        dendrite = (
            prediction[..., 0]
        )

        spine = (
            prediction[..., 1]
        )

    print(
        "Dendrite shape:",
        dendrite.shape,
    )

    print(
        "Spine shape   :",
        spine.shape,
    )

    print(
        "Dendrite range:",
        float(dendrite.min()),
        "->",
        float(dendrite.max()),
    )

    print(
        "Spine range   :",
        float(spine.min()),
        "->",
        float(spine.max()),
    )

    return (
        dendrite,
        spine,
    )


# =========================================================
# Spine segmentation ground truth
# =========================================================

def load_spine_mask(path):

    print("\n========================================")
    print("Loading spine GT")
    print("========================================")

    print(path)

    data = fl.load(
        path
    )

    mask = np.asarray(
        data["mask"]
    )

    print(
        "Stored shape:",
        mask.shape,
    )

    # DeepD3 benchmark masks are stored as:
    #
    #     (Z, X, Y)
    #
    # Benchmark TIFF is:
    #
    #     (Z, Y, X)

    mask = mask.transpose(
        0,
        2,
        1,
    )

    mask = (
        mask > 0
    )

    print(
        "Converted shape:",
        mask.shape,
    )

    print(
        "GT voxels:",
        int(mask.sum()),
    )

    return mask


def load_all_spine_gt(
    benchmark_dir,
):

    U = load_spine_mask(
        benchmark_dir
        / "Segmentation_U.mask"
    )

    V = load_spine_mask(
        benchmark_dir
        / "Segmentation_V.mask"
    )

    W = load_spine_mask(
        benchmark_dir
        / "Segmentation_W.mask"
    )

    intersection = (
        U
        & V
        & W
    )

    union = (
        U
        | V
        | W
    )

    print("\n========================================")
    print("Spine GT summary")
    print("========================================")

    print(
        "U voxels           :",
        int(U.sum()),
    )

    print(
        "V voxels           :",
        int(V.sum()),
    )

    print(
        "W voxels           :",
        int(W.sum()),
    )

    print(
        "Intersection voxels:",
        int(intersection.sum()),
    )

    print(
        "Union voxels       :",
        int(union.sum()),
    )

    return {

        "U": U,

        "V": V,

        "W": W,

        "intersection": intersection,

        "union": union,

    }


# =========================================================
# Dendrite SWC rasterization
# =========================================================

def rasterize_dendrite_swc(
    swc_path,
    reference_tif,
    spacing=(1, 1, 1),
):

    print("\n========================================")
    print("Rasterizing dendrite")
    print("========================================")

    print(
        "SWC:",
        swc_path,
    )

    converter = DendriteSWC(
        spacing=list(spacing)
    )

    converter.open(
        str(swc_path),
        str(reference_tif),
    )

    converter.stack = np.zeros(
        converter.ref.shape,
        dtype=np.uint8,
    )

    swc = converter.swc

    node_ids = [
        int(x)
        for x in swc.index
    ]

    node_id_set = set(
        node_ids
    )

    min_id = min(
        node_ids
    )

    max_id = max(
        node_ids
    )

    print(
        "SWC nodes   :",
        len(node_ids),
    )

    print(
        "SWC ID range:",
        min_id,
        "->",
        max_id,
    )

    missing_ids = sorted(
        set(
            range(
                min_id,
                max_id + 1,
            )
        )
        -
        node_id_set
    )

    if missing_ids:

        print(
            "Missing SWC node IDs:",
            missing_ids[:20],
            (
                "..."
                if len(missing_ids) > 20
                else ""
            ),
        )

    else:

        print(
            "Missing SWC node IDs: none"
        )

    root_nodes = 0
    drawn_edges = 0
    missing_parent_edges = 0
    drawing_errors = 0

    for node_id in tqdm(
        node_ids
    ):

        row = swc.loc[
            node_id
        ]

        parent_id = int(
            row.parent
        )

        if parent_id <= 0:

            root_nodes += 1

            continue

        if parent_id not in node_id_set:

            missing_parent_edges += 1

            continue

        p0, r0 = xyzr(
            swc,
            parent_id,
        )

        p1, r1 = xyzr(
            swc,
            node_id,
        )

        try:

            line_w_sphere(
                converter.stack,
                p0,
                p1,
                r0,
                r1,
                255,
                converter.spacing,
            )

            drawn_edges += 1

        except Exception:

            drawing_errors += 1

    print("\nSWC rasterization summary")
    print("-------------------------")

    print(
        "Root nodes      :",
        root_nodes,
    )

    print(
        "Drawn edges     :",
        drawn_edges,
    )

    print(
        "Missing parents :",
        missing_parent_edges,
    )

    print(
        "Drawing errors  :",
        drawing_errors,
    )

    mask = (
        converter.stack > 0
    )

    print(
        "Shape           :",
        mask.shape,
    )

    print(
        "GT voxels       :",
        int(mask.sum()),
    )

    return mask


def load_all_dendrite_gt(
    benchmark_dir,
    reference_tif,
):

    spacing = (
        1,
        1,
        1,
    )

    masks = {}

    for rater in [
        "U",
        "V",
        "W",
    ]:

        swc_path = (
            benchmark_dir
            / f"Dendrite_{rater}.swc"
        )

        masks[rater] = rasterize_dendrite_swc(
            swc_path=swc_path,
            reference_tif=reference_tif,
            spacing=spacing,
        )

    return masks


# =========================================================
# GT shape checks
# =========================================================

def check_shapes(
    image,
    spine_gt,
    dendrite_gt,
):

    image_shape = (
        image.shape
    )

    print("\n========================================")
    print("Checking GT shapes")
    print("========================================")

    for name, mask in spine_gt.items():

        print(
            f"Spine {name}:",
            mask.shape,
        )

        if (
            mask.shape
            != image_shape
        ):

            raise RuntimeError(
                f"Spine GT {name} shape "
                f"{mask.shape} does not match "
                f"benchmark {image_shape}"
            )

    for name, mask in dendrite_gt.items():

        print(
            f"Dendrite {name}:",
            mask.shape,
        )

        if (
            mask.shape
            != image_shape
        ):

            raise RuntimeError(
                f"Dendrite GT {name} shape "
                f"{mask.shape} does not match "
                f"benchmark {image_shape}"
            )


# =========================================================
# Segmentation evaluation
# =========================================================

def evaluate_segmentation_model(
    model_name,
    prediction_path,
    image_shape,
    spine_gt,
    dendrite_gt,
):

    print("\n\n########################################")
    print("SEGMENTATION MODEL:", model_name)
    print("########################################")

    dend_prob, spine_prob = load_prediction(
        prediction_path
    )

    if (
        dend_prob.shape
        != image_shape
    ):

        raise RuntimeError(
            f"Dendrite prediction shape "
            f"{dend_prob.shape} does not match "
            f"benchmark {image_shape}"
        )

    if (
        spine_prob.shape
        != image_shape
    ):

        raise RuntimeError(
            f"Spine prediction shape "
            f"{spine_prob.shape} does not match "
            f"benchmark {image_shape}"
        )

    thresholds = (
        SEGMENTATION_THRESHOLDS[
            model_name
        ]
    )

    dend_thr = (
        thresholds["dendrite"]
    )

    spine_thr = (
        thresholds["spine"]
    )

    print("\nFrozen thresholds")
    print("-----------------")

    print(
        "Dendrite:",
        dend_thr,
    )

    print(
        "Spine   :",
        spine_thr,
    )

    dend_pred = (
        dend_prob >= dend_thr
    )

    spine_pred = (
        spine_prob >= spine_thr
    )

    print("\nPredicted voxels")
    print("----------------")

    print(
        "Dendrite:",
        int(dend_pred.sum()),
    )

    print(
        "Spine   :",
        int(spine_pred.sum()),
    )

    results = []

    # -----------------------------------------------------
    # Dendrite
    # -----------------------------------------------------

    print("\n========================================")
    print("DENDRITE RESULTS")
    print("========================================")

    for rater in [
        "U",
        "V",
        "W",
    ]:

        gt = (
            dendrite_gt[rater]
        )

        iou = compute_iou(
            dend_pred,
            gt,
        )

        dice = compute_dice(
            dend_pred,
            gt,
        )

        print(
            f"{rater}: "
            f"IoU={iou:.4f}, "
            f"Dice={dice:.4f}"
        )

        results.append({

            "model": model_name,

            "structure": "dendrite",

            "ground_truth": rater,

            "threshold": dend_thr,

            "iou": iou,

            "dice": dice,

        })

    # -----------------------------------------------------
    # Spine
    # -----------------------------------------------------

    print("\n========================================")
    print("SPINE RESULTS")
    print("========================================")

    for gt_name in [

        "U",

        "V",

        "W",

        "intersection",

        "union",

    ]:

        gt = (
            spine_gt[gt_name]
        )

        iou = compute_iou(
            spine_pred,
            gt,
        )

        dice = compute_dice(
            spine_pred,
            gt,
        )

        print(
            f"{gt_name}: "
            f"IoU={iou:.4f}, "
            f"Dice={dice:.4f}"
        )

        results.append({

            "model": model_name,

            "structure": "spine",

            "ground_truth": gt_name,

            "threshold": spine_thr,

            "iou": iou,

            "dice": dice,

        })

    return results


# =========================================================
# Benchmark spine-center loading
# =========================================================

def load_benchmark_spine_centers(
    csv_path,
):

    print("\n========================================")
    print("Loading benchmark spine centers")
    print("========================================")

    print(
        csv_path
    )

    df = pd.read_csv(
        csv_path
    )

    required = {
        "Rater",
        "X",
        "Y",
        "Pos",
        "label",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise RuntimeError(
            "Missing required columns: "
            f"{missing}"
        )

    print(
        "Annotation rows :",
        len(df),
    )

    print(
        "Unique raters   :",
        df["Rater"].nunique(),
    )

    print(
        "Unique clusters :",
        df["label"].nunique(),
    )

    clusters = []

    for cluster_id, group in df.groupby(
        "label"
    ):

        # CSV:
        #
        # X   = image X
        # Y   = image Y
        # Pos = image Z
        #
        # Store as Z,Y,X.

        z = float(
            group["Pos"].mean()
        )

        y = float(
            group["Y"].mean()
        )

        x = float(
            group["X"].mean()
        )

        n_raters = int(
            group["Rater"].nunique()
        )

        clusters.append({

            "cluster_id":
                cluster_id,

            "z":
                z,

            "y":
                y,

            "x":
                x,

            "n_raters":
                n_raters,

        })

    cluster_df = pd.DataFrame(
        clusters
    )

    print("\nCluster agreement")
    print("-----------------")

    agreement_counts = (
        cluster_df[
            "n_raters"
        ]
        .value_counts()
        .sort_index()
    )

    for n_raters, count in agreement_counts.items():

        print(
            f"{int(n_raters)} rater(s): "
            f"{int(count)} clusters"
        )

    print("\nCumulative GT counts")
    print("--------------------")

    for min_raters in range(
        1,
        8,
    ):

        count = int(
            (
                cluster_df[
                    "n_raters"
                ]
                >= min_raters
            ).sum()
        )

        print(
            f">= {min_raters} raters: "
            f"{count} clusters"
        )

    return cluster_df


# =========================================================
# Predicted spine-center detection
# =========================================================

def detect_spine_peaks(
    spine_probability,
    threshold,
):

    # Same approach used for synthetic evaluation.

    smoothed = gaussian_filter(
        spine_probability,
        sigma=GAUSSIAN_SIGMA,
    )

    local_maximum = (
        smoothed
        ==
        maximum_filter(
            smoothed,
            size=PEAK_NEIGHBORHOOD_ZYX,
            mode="nearest",
        )
    )

    keep = (
        local_maximum
        &
        (
            smoothed
            >= threshold
        )
    )

    coords_zyx = np.argwhere(
        keep
    )

    return coords_zyx


# =========================================================
# Coordinate conversion
# =========================================================

def pixel_to_nm(
    coords_zyx,
):

    coords_zyx = np.asarray(
        coords_zyx,
        dtype=np.float64,
    )

    return (
        coords_zyx
        * SPACING_NM_ZYX
    )


# =========================================================
# Greedy one-to-one matching
# =========================================================

def greedy_match_centers(
    predicted_zyx,
    gt_zyx,
    max_distance_nm=MATCH_DISTANCE_NM,
):

    predicted_zyx = np.asarray(
        predicted_zyx,
        dtype=np.float64,
    )

    gt_zyx = np.asarray(
        gt_zyx,
        dtype=np.float64,
    )

    n_pred = len(
        predicted_zyx
    )

    n_gt = len(
        gt_zyx
    )

    if n_pred == 0:

        return {

            "tp": 0,

            "fp": 0,

            "fn": n_gt,

            "matched_distances_nm": [],

        }

    if n_gt == 0:

        return {

            "tp": 0,

            "fp": n_pred,

            "fn": 0,

            "matched_distances_nm": [],

        }

    pred_nm = pixel_to_nm(
        predicted_zyx
    )

    gt_nm = pixel_to_nm(
        gt_zyx
    )

    distance_matrix = cdist(
        pred_nm,
        gt_nm,
    )

    pred_indices, gt_indices = np.where(
        distance_matrix
        <= max_distance_nm
    )

    candidate_pairs = []

    for p_idx, g_idx in zip(
        pred_indices,
        gt_indices,
    ):

        candidate_pairs.append(
            (
                float(
                    distance_matrix[
                        p_idx,
                        g_idx,
                    ]
                ),
                int(p_idx),
                int(g_idx),
            )
        )

    candidate_pairs.sort(
        key=lambda x: x[0]
    )

    used_pred = set()
    used_gt = set()

    matched_distances = []

    for (
        distance,
        p_idx,
        g_idx,
    ) in candidate_pairs:

        if p_idx in used_pred:
            continue

        if g_idx in used_gt:
            continue

        used_pred.add(
            p_idx
        )

        used_gt.add(
            g_idx
        )

        matched_distances.append(
            distance
        )

    tp = len(
        matched_distances
    )

    fp = (
        n_pred
        - tp
    )

    fn = (
        n_gt
        - tp
    )

    return {

        "tp":
            tp,

        "fp":
            fp,

        "fn":
            fn,

        "matched_distances_nm":
            matched_distances,

    }


# =========================================================
# Precision / recall / F1
# =========================================================

def detection_metrics(
    tp,
    fp,
    fn,
):

    if (
        tp + fp
    ) > 0:

        precision = (
            tp
            /
            (tp + fp)
        )

    else:

        precision = 0.0

    if (
        tp + fn
    ) > 0:

        recall = (
            tp
            /
            (tp + fn)
        )

    else:

        recall = 0.0

    if (
        precision + recall
    ) > 0:

        f1 = (
            2.0
            * precision
            * recall
            /
            (
                precision
                + recall
            )
        )

    else:

        f1 = 0.0

    return (
        precision,
        recall,
        f1,
    )


# =========================================================
# Spine detection evaluation
# =========================================================

def evaluate_spine_detection(
    model_name,
    prediction_path,
    cluster_df,
):

    print("\n\n########################################")
    print("SPINE DETECTION:", model_name)
    print("########################################")

    _, spine_probability = load_prediction(
        prediction_path
    )

    threshold = (
        DETECTION_THRESHOLDS[
            model_name
        ]
    )

    print(
        "\nFrozen detection threshold:",
        threshold,
    )

    print(
        "Gaussian sigma:",
        GAUSSIAN_SIGMA,
    )

    print(
        "Peak neighborhood ZYX:",
        PEAK_NEIGHBORHOOD_ZYX,
    )

    print(
        "Match distance:",
        MATCH_DISTANCE_NM,
        "nm",
    )

    predicted_centers = detect_spine_peaks(
        spine_probability,
        threshold,
    )

    print(
        "\nPredicted peaks:",
        len(predicted_centers),
    )

    results = []

    # -----------------------------------------------------
    # Test progressively stronger human agreement.
    #
    # >=1:
    # all benchmark clusters
    #
    # >=4:
    # majority/high-agreement spines
    #
    # >=7:
    # unanimous annotations
    # -----------------------------------------------------

    for min_raters in [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ]:

        gt_subset = cluster_df[
            cluster_df["n_raters"]
            >= min_raters
        ].copy()

        gt_centers = gt_subset[
            [
                "z",
                "y",
                "x",
            ]
        ].to_numpy(
            dtype=np.float64
        )

        matching = greedy_match_centers(
            predicted_zyx=predicted_centers,
            gt_zyx=gt_centers,
            max_distance_nm=MATCH_DISTANCE_NM,
        )

        tp = (
            matching["tp"]
        )

        fp = (
            matching["fp"]
        )

        fn = (
            matching["fn"]
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

        matched_distances = (
            matching[
                "matched_distances_nm"
            ]
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

        print("\n----------------------------------------")

        print(
            f"Minimum raters: {min_raters}"
        )

        print("----------------------------------------")

        print(
            "GT clusters :",
            len(gt_centers),
        )

        print(
            "Predictions :",
            len(predicted_centers),
        )

        print(
            "TP / FP / FN:",
            tp,
            "/",
            fp,
            "/",
            fn,
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall   : {recall:.4f}"
        )

        print(
            f"F1       : {f1:.4f}"
        )

        print(
            "Mean match distance   : "
            f"{mean_distance:.1f} nm"
        )

        print(
            "Median match distance : "
            f"{median_distance:.1f} nm"
        )

        results.append({

            "model":
                model_name,

            "min_raters":
                min_raters,

            "gt_clusters":
                len(gt_centers),

            "predicted_peaks":
                len(predicted_centers),

            "threshold":
                threshold,

            "gaussian_sigma":
                GAUSSIAN_SIGMA,

            "match_distance_nm":
                MATCH_DISTANCE_NM,

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

        })

    return results


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Evaluate real-trained and synthetic-trained "
            "DeepD3 models on the DeepD3 benchmark."
        )

    )

    parser.add_argument(

        "--benchmark-dir",

        type=Path,

        default=BENCHMARK_DIR,

    )

    parser.add_argument(

        "--real-prediction",

        type=Path,

        default=DEFAULT_REAL_PRED,

    )

    parser.add_argument(

        "--synthetic-prediction",

        type=Path,

        default=DEFAULT_SYNTH_PRED,

    )

    parser.add_argument(

        "--output-dir",

        type=Path,

        default=DEFAULT_OUTPUT_DIR,

    )

    args = (
        parser.parse_args()
    )

    benchmark_dir = (
        args.benchmark_dir
    )

    reference_tif = (
        benchmark_dir
        / "DeepD3_Benchmark.tif"
    )

    annotation_csv = (
        benchmark_dir
        / "Annotations_and_Clusters.csv"
    )

    # =====================================================
    # Required files
    # =====================================================

    required_files = [

        reference_tif,

        annotation_csv,

        benchmark_dir
        / "Segmentation_U.mask",

        benchmark_dir
        / "Segmentation_V.mask",

        benchmark_dir
        / "Segmentation_W.mask",

        benchmark_dir
        / "Dendrite_U.swc",

        benchmark_dir
        / "Dendrite_V.swc",

        benchmark_dir
        / "Dendrite_W.swc",

        args.real_prediction,

        args.synthetic_prediction,

    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

    # =====================================================
    # Load benchmark image
    # =====================================================

    image = tifffile.imread(
        reference_tif
    )

    print("========================================")
    print("DeepD3 benchmark evaluation")
    print("========================================")

    print("\nBenchmark image")

    print(
        "Shape:",
        image.shape,
    )

    print(
        "dtype:",
        image.dtype,
    )

    print(
        "range:",
        int(image.min()),
        "->",
        int(image.max()),
    )

    # =====================================================
    # Load segmentation GT
    # =====================================================

    spine_gt = load_all_spine_gt(
        benchmark_dir
    )

    dendrite_gt = load_all_dendrite_gt(
        benchmark_dir,
        reference_tif,
    )

    check_shapes(
        image=image,
        spine_gt=spine_gt,
        dendrite_gt=dendrite_gt,
    )

    # =====================================================
    # Load center annotations
    # =====================================================

    cluster_df = load_benchmark_spine_centers(
        annotation_csv
    )

    # =====================================================
    # SEGMENTATION EVALUATION
    # =====================================================

    segmentation_results = []

    segmentation_results.extend(

        evaluate_segmentation_model(

            model_name=
                "real_32F_94nm",

            prediction_path=
                args.real_prediction,

            image_shape=
                image.shape,

            spine_gt=
                spine_gt,

            dendrite_gt=
                dendrite_gt,

        )

    )

    segmentation_results.extend(

        evaluate_segmentation_model(

            model_name=
                "synthetic_32F_94nm",

            prediction_path=
                args.synthetic_prediction,

            image_shape=
                image.shape,

            spine_gt=
                spine_gt,

            dendrite_gt=
                dendrite_gt,

        )

    )

    segmentation_df = pd.DataFrame(
        segmentation_results
    )

    # =====================================================
    # SPINE DETECTION EVALUATION
    # =====================================================

    detection_results = []

    detection_results.extend(

        evaluate_spine_detection(

            model_name=
                "real_32F_94nm",

            prediction_path=
                args.real_prediction,

            cluster_df=
                cluster_df,

        )

    )

    detection_results.extend(

        evaluate_spine_detection(

            model_name=
                "synthetic_32F_94nm",

            prediction_path=
                args.synthetic_prediction,

            cluster_df=
                cluster_df,

        )

    )

    detection_df = pd.DataFrame(
        detection_results
    )

    # =====================================================
    # Save
    # =====================================================

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    segmentation_csv = (
        args.output_dir
        / "segmentation_results.csv"
    )

    detection_csv = (
        args.output_dir
        / "spine_detection_results.csv"
    )

    segmentation_df.to_csv(
        segmentation_csv,
        index=False,
    )

    detection_df.to_csv(
        detection_csv,
        index=False,
    )

    # =====================================================
    # Print segmentation summary
    # =====================================================

    print("\n\n========================================")
    print("FINAL SEGMENTATION SUMMARY")
    print("========================================")

    print(

        segmentation_df[
            [
                "model",
                "structure",
                "ground_truth",
                "threshold",
                "iou",
                "dice",
            ]
        ].to_string(
            index=False
        )

    )

    # =====================================================
    # Print detection summary
    # =====================================================

    print("\n\n========================================")
    print("FINAL SPINE DETECTION SUMMARY")
    print("========================================")

    print(

        detection_df[
            [
                "model",
                "min_raters",
                "gt_clusters",
                "predicted_peaks",
                "threshold",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "mean_match_distance_nm",
            ]
        ].to_string(
            index=False
        )

    )

    print("\nSaved:")

    print(
        segmentation_csv
    )

    print(
        detection_csv
    )

    print("\nDone.")


if __name__ == "__main__":

    main()