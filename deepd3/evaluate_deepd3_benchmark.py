from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import tifffile
import flammkuchen as fl

# Use the same official DeepD3 dendrite conversion class
# that is available in the installed package.
from deepd3.core.dendrite import DendriteSWC


# =========================================================
# Default paths
# =========================================================

BENCHMARK_DIR = Path("benchmarks/deepd3")

DEFAULT_IMAGE = BENCHMARK_DIR / "DeepD3_Benchmark.tif"

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
# Important:
# These thresholds were selected BEFORE benchmark evaluation.
#
# Real-trained model thresholds:
# selected on synthetic validation during Week 8
#
# Synthetic-trained model thresholds:
# selected on synthetic validation during Week 9
#
# We do NOT tune thresholds on benchmark annotations.
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
# Utility
# =========================================================

def compute_iou(pred, gt):
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return np.nan

    return intersection / union


def compute_dice(pred, gt):
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)

    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()

    if total == 0:
        return np.nan

    return 2.0 * intersection / total


# =========================================================
# Load DeepD3 prediction
# =========================================================

def load_prediction(path):

    print(f"\nLoading prediction:")
    print(path)

    data = fl.load(path)

    if "prediction" in data:
        prediction = np.asarray(data["prediction"])

        dendrite = prediction[..., 0]
        spine = prediction[..., 1]

    else:
        dendrite = np.asarray(data["dendrites"])
        spine = np.asarray(data["spines"])

    print("Dendrite shape:", dendrite.shape)
    print("Spine shape   :", spine.shape)

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

    return dendrite, spine


# =========================================================
# Spine masks
# =========================================================

def load_spine_mask(path):

    data = fl.load(path)

    mask = np.asarray(data["mask"])

    print(f"\nLoading spine GT: {path.name}")
    print("Stored shape:", mask.shape)

    # Official DeepD3 conversion:
    # stored mask = (Z, X, Y)
    # benchmark image = (Z, Y, X)
    mask = mask.transpose(0, 2, 1)

    mask = mask > 0

    print("Converted shape:", mask.shape)
    print("GT voxels:", int(mask.sum()))

    return mask


def load_all_spine_gt(benchmark_dir):

    U = load_spine_mask(
        benchmark_dir / "Segmentation_U.mask"
    )

    V = load_spine_mask(
        benchmark_dir / "Segmentation_V.mask"
    )

    W = load_spine_mask(
        benchmark_dir / "Segmentation_W.mask"
    )

    intersection = U & V & W
    union = U | V | W

    print("\nSpine GT summary")
    print("----------------")
    print("U voxels           :", int(U.sum()))
    print("V voxels           :", int(V.sum()))
    print("W voxels           :", int(W.sum()))
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
# Dendrite SWC conversion
# =========================================================

def rasterize_dendrite_swc(
    swc_path,
    reference_tif,
    spacing=(1, 1, 1),
):

    print("\n========================================")
    print("Rasterizing dendrite")
    print("========================================")
    print("SWC:", swc_path)

    converter = DendriteSWC(
        spacing=list(spacing)
    )

    converter.open(
        str(swc_path),
        str(reference_tif),
    )

    # We do NOT need to save an intermediate TIFF.
    # Reproduce DeepD3's conversion in memory.

    converter.stack = np.zeros(
        converter.ref.shape,
        dtype=np.uint8,
    )

    converter._binarize_swc_w_spheres()

    mask = converter.stack > 0

    print("Shape    :", mask.shape)
    print("GT voxels:", int(mask.sum()))

    return mask


def load_all_dendrite_gt(
    benchmark_dir,
    reference_tif,
):

    # IMPORTANT:
    #
    # The benchmark SWC coordinates behave like image-pixel
    # coordinates, and the official implementation uses the
    # spacing supplied to DendriteSWC.
    #
    # For these benchmark SWCs we initially reproduce their
    # coordinate-space rasterization with unit spacing.
    #
    spacing = (1, 1, 1)

    masks = {}

    for rater in ["U", "V", "W"]:

        swc = (
            benchmark_dir
            / f"Dendrite_{rater}.swc"
        )

        masks[rater] = rasterize_dendrite_swc(
            swc_path=swc,
            reference_tif=reference_tif,
            spacing=spacing,
        )

    return masks


# =========================================================
# Evaluate one model
# =========================================================

def evaluate_model(
    model_name,
    prediction_path,
    spine_gt,
    dendrite_gt,
):

    print("\n\n########################################")
    print("MODEL:", model_name)
    print("########################################")

    dend_prob, spine_prob = load_prediction(
        prediction_path
    )

    thresholds = THRESHOLDS[model_name]

    dend_thr = thresholds["dendrite"]
    spine_thr = thresholds["spine"]

    print("\nFrozen thresholds")
    print("-----------------")
    print("Dendrite:", dend_thr)
    print("Spine   :", spine_thr)

    dend_pred = dend_prob >= dend_thr
    spine_pred = spine_prob >= spine_thr

    print("\nPredicted voxels")
    print("----------------")
    print("Dendrite:", int(dend_pred.sum()))
    print("Spine   :", int(spine_pred.sum()))

    rows = []

    # -----------------------------------------------------
    # Dendrite
    # -----------------------------------------------------

    print("\nDENDRITE RESULTS")
    print("----------------")

    for rater in ["U", "V", "W"]:

        gt = dendrite_gt[rater]

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

        rows.append({
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

    print("\nSPINE RESULTS")
    print("-------------")

    for gt_name in [
        "U",
        "V",
        "W",
        "intersection",
        "union",
    ]:

        gt = spine_gt[gt_name]

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

        rows.append({
            "model": model_name,
            "structure": "spine",
            "ground_truth": gt_name,
            "threshold": spine_thr,
            "iou": iou,
            "dice": dice,
        })

    return rows


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate DeepD3 models on the official "
            "multi-rater DeepD3 benchmark."
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

    args = parser.parse_args()

    benchmark_dir = args.benchmark_dir

    reference_tif = (
        benchmark_dir
        / "DeepD3_Benchmark.tif"
    )

    # -----------------------------------------------------
    # Check benchmark
    # -----------------------------------------------------

    image = tifffile.imread(
        reference_tif
    )

    print("========================================")
    print("DeepD3 benchmark evaluation")
    print("========================================")

    print("\nBenchmark image")
    print("Shape:", image.shape)
    print("dtype:", image.dtype)

    # -----------------------------------------------------
    # Ground truth
    # -----------------------------------------------------

    spine_gt = load_all_spine_gt(
        benchmark_dir
    )

    dendrite_gt = load_all_dendrite_gt(
        benchmark_dir,
        reference_tif,
    )

    # Verify shapes
    for name, mask in spine_gt.items():

        if mask.shape != image.shape:
            raise RuntimeError(
                f"Spine GT {name} shape "
                f"{mask.shape} does not match "
                f"benchmark {image.shape}"
            )

    for name, mask in dendrite_gt.items():

        if mask.shape != image.shape:
            raise RuntimeError(
                f"Dendrite GT {name} shape "
                f"{mask.shape} does not match "
                f"benchmark {image.shape}"
            )

    # -----------------------------------------------------
    # Evaluate
    # -----------------------------------------------------

    results = []

    results.extend(
        evaluate_model(
            model_name="real_32F_94nm",
            prediction_path=args.real_prediction,
            spine_gt=spine_gt,
            dendrite_gt=dendrite_gt,
        )
    )

    results.extend(
        evaluate_model(
            model_name="synthetic_32F_94nm",
            prediction_path=args.synthetic_prediction,
            spine_gt=spine_gt,
            dendrite_gt=dendrite_gt,
        )
    )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame(results)

    csv_path = (
        args.output_dir
        / "segmentation_results.csv"
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\n\n========================================")
    print("SUMMARY")
    print("========================================")

    print(
        df[
            [
                "model",
                "structure",
                "ground_truth",
                "threshold",
                "iou",
                "dice",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(csv_path)


if __name__ == "__main__":
    main()