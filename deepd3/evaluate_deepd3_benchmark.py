from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import tifffile
import flammkuchen as fl

from tqdm import tqdm

from deepd3.core.dendrite import (
    DendriteSWC,
    xyzr,
    line_w_sphere,
)


# =========================================================
# Default paths
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
# Frozen thresholds
# =========================================================
#
# These thresholds were selected on SYNTHETIC VALIDATION
# data before benchmark evaluation.
#
# They are NOT tuned using benchmark annotations.
#
# Real-trained model:
# Week 8 validation thresholds
#
# Synthetic-trained model:
# Week 9 validation thresholds
# =========================================================

THRESHOLDS = {

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
# Metrics
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

    # -----------------------------------------------------
    # Expected format:
    #
    # {
    #     "dendrites": ...
    #     "spines": ...
    # }
    #
    # But support full prediction arrays as well.
    # -----------------------------------------------------

    if (
        isinstance(data, dict)
        and "prediction" in data
    ):

        prediction = np.asarray(
            data["prediction"]
        )

        dendrite = prediction[..., 0]
        spine = prediction[..., 1]

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

        dendrite = prediction[..., 0]
        spine = prediction[..., 1]

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
# Spine benchmark masks
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

    # -----------------------------------------------------
    # Official DeepD3 spine masks are stored as:
    #
    #     (Z, X, Y)
    #
    # Benchmark TIFF is:
    #
    #     (Z, Y, X)
    #
    # Therefore transpose:
    #
    #     (0, 2, 1)
    # -----------------------------------------------------

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
# Robust dendrite SWC rasterization
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

    # -----------------------------------------------------
    # Load using official DeepD3 DendriteSWC parser.
    # -----------------------------------------------------

    converter = DendriteSWC(
        spacing=list(spacing)
    )

    converter.open(
        str(swc_path),
        str(reference_tif),
    )

    # -----------------------------------------------------
    # Create empty stack matching benchmark TIFF.
    # -----------------------------------------------------

    converter.stack = np.zeros(
        converter.ref.shape,
        dtype=np.uint8,
    )

    swc = converter.swc

    # -----------------------------------------------------
    # Inspect SWC IDs
    # -----------------------------------------------------

    node_ids = [
        int(x)
        for x
        in swc.index
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

    # -----------------------------------------------------
    # Robust rasterization
    #
    # Official DeepD3 loops:
    #
    # for i in range(1, self.swc.shape[0]):
    #
    # This assumes consecutive node IDs.
    #
    # Benchmark Dendrite_V contains a missing ID,
    # causing KeyError.
    #
    # We instead iterate through ACTUAL node IDs.
    #
    # Parent references remain unchanged.
    # -----------------------------------------------------

    drawn_edges = 0

    missing_parent_edges = 0

    drawing_errors = 0

    root_nodes = 0

    for node_id in tqdm(
        node_ids
    ):

        row = swc.loc[
            node_id
        ]

        parent_id = int(
            row.parent
        )

        # Root node
        if parent_id <= 0:

            root_nodes += 1

            continue

        # Parent reference missing from SWC
        if (
            parent_id
            not in node_id_set
        ):

            missing_parent_edges += 1

            continue

        # -------------------------------------------------
        # Get coordinates exactly like DeepD3.
        #
        # xyzr returns:
        #
        # (y, x, z), radius
        # -------------------------------------------------

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

        except Exception as exc:

            drawing_errors += 1

            # Original DeepD3 also silently ignores
            # drawing failures.
            #
            # We count them so we can inspect whether
            # anything unusual occurred.

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

    # -----------------------------------------------------
    # The benchmark SWC coordinate ranges correspond
    # closely to image pixel coordinates.
    #
    # Therefore use unit spacing for this first direct
    # rasterization.
    #
    # We will inspect the resulting masks before treating
    # dendrite IoUs as final thesis numbers.
    # -----------------------------------------------------

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
# Shape checking
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
                f"benchmark image "
                f"{image_shape}"

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
                f"benchmark image "
                f"{image_shape}"

            )


# =========================================================
# Model evaluation
# =========================================================

def evaluate_model(
    model_name,
    prediction_path,
    image_shape,
    spine_gt,
    dendrite_gt,
):

    print("\n\n########################################")
    print("MODEL:", model_name)
    print("########################################")

    dend_prob, spine_prob = load_prediction(
        prediction_path
    )

    # -----------------------------------------------------
    # Shape check
    # -----------------------------------------------------

    if (
        dend_prob.shape
        != image_shape
    ):

        raise RuntimeError(

            f"Dendrite prediction shape "
            f"{dend_prob.shape} "
            f"does not match benchmark "
            f"{image_shape}"

        )

    if (
        spine_prob.shape
        != image_shape
    ):

        raise RuntimeError(

            f"Spine prediction shape "
            f"{spine_prob.shape} "
            f"does not match benchmark "
            f"{image_shape}"

        )

    # -----------------------------------------------------
    # Frozen thresholds
    # -----------------------------------------------------

    thresholds = (
        THRESHOLDS[
            model_name
        ]
    )

    dend_thr = (
        thresholds[
            "dendrite"
        ]
    )

    spine_thr = (
        thresholds[
            "spine"
        ]
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

    # -----------------------------------------------------
    # Threshold probability maps
    # -----------------------------------------------------

    dend_pred = (
        dend_prob
        >= dend_thr
    )

    spine_pred = (
        spine_prob
        >= spine_thr
    )

    print("\nPredicted voxels")
    print("----------------")

    print(
        "Dendrite:",
        int(
            dend_pred.sum()
        ),
    )

    print(
        "Spine   :",
        int(
            spine_pred.sum()
        ),
    )

    results = []

    # =====================================================
    # Dendrite segmentation
    # =====================================================

    print("\n========================================")
    print("DENDRITE RESULTS")
    print("========================================")

    for rater in [
        "U",
        "V",
        "W",
    ]:

        gt = (
            dendrite_gt[
                rater
            ]
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

            "model":
                model_name,

            "structure":
                "dendrite",

            "ground_truth":
                rater,

            "threshold":
                dend_thr,

            "iou":
                iou,

            "dice":
                dice,

        })

    # =====================================================
    # Spine segmentation
    # =====================================================

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
            spine_gt[
                gt_name
            ]
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

            "model":
                model_name,

            "structure":
                "spine",

            "ground_truth":
                gt_name,

            "threshold":
                spine_thr,

            "iou":
                iou,

            "dice":
                dice,

        })

    return results


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser(

        description=(

            "Evaluate real-trained and "
            "synthetic-trained DeepD3 models "
            "on the DeepD3 benchmark."

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

    # -----------------------------------------------------
    # Required-file checks
    # -----------------------------------------------------

    required_files = [

        reference_tif,

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

    # -----------------------------------------------------
    # Benchmark image
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Spine ground truth
    # -----------------------------------------------------

    spine_gt = load_all_spine_gt(
        benchmark_dir
    )

    # -----------------------------------------------------
    # Dendrite ground truth
    # -----------------------------------------------------

    dendrite_gt = load_all_dendrite_gt(

        benchmark_dir,

        reference_tif,

    )

    # -----------------------------------------------------
    # Verify all dimensions
    # -----------------------------------------------------

    check_shapes(

        image=image,

        spine_gt=spine_gt,

        dendrite_gt=dendrite_gt,

    )

    # -----------------------------------------------------
    # Evaluate both models
    # -----------------------------------------------------

    results = []

    results.extend(

        evaluate_model(

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

    results.extend(

        evaluate_model(

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

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    args.output_dir.mkdir(

        parents=True,

        exist_ok=True,

    )

    df = pd.DataFrame(
        results
    )

    output_csv = (

        args.output_dir
        / "segmentation_results.csv"

    )

    df.to_csv(

        output_csv,

        index=False,

    )

    # -----------------------------------------------------
    # Print summary
    # -----------------------------------------------------

    print("\n\n========================================")
    print("FINAL SEGMENTATION SUMMARY")
    print("========================================")

    summary_columns = [

        "model",

        "structure",

        "ground_truth",

        "threshold",

        "iou",

        "dice",

    ]

    print(

        df[
            summary_columns
        ].to_string(
            index=False
        )

    )

    print("\nSaved:")
    print(output_csv)

    print("\nDone.")


if __name__ == "__main__":

    main()