"""
visualize_deepd3_benchmark.py
-----------------------------

Visualize real-trained and synthetic-trained DeepD3 predictions
on a representative region of the real DeepD3 benchmark.

The figure compares the benchmark image, expert annotation
intersection (U ∩ V ∩ W), real-trained DeepD3 prediction, and
synthetic-trained DeepD3 prediction for spines and dendrites.


Usage
-----

Run from the repository root:

    python deepd3/visualize_deepd3_benchmark.py


Input
-----

benchmarks/deepd3/
├── DeepD3_Benchmark.tif
├── Segmentation_U.mask
├── Segmentation_V.mask
├── Segmentation_W.mask
├── Dendrite_U.swc
├── Dendrite_V.swc
├── Dendrite_W.swc
└── predictions/
    ├── real_32F_94nm.prediction
    └── synthetic_32F_94nm.prediction


Output
------

benchmarks/deepd3/qualitative/
└── deepd3_benchmark_comparison.png


Important
---------

The same frozen segmentation thresholds used for quantitative
benchmark evaluation are used for visualization.
"""

from pathlib import Path

import numpy as np
import tifffile
import flammkuchen as fl
import matplotlib.pyplot as plt

from deepd3.core.dendrite import (
    DendriteSWC,
    xyzr,
    line_w_sphere,
)


# =========================================================
# Paths
# =========================================================

BENCHMARK_DIR = Path("benchmarks/deepd3")

IMAGE_PATH = (
    BENCHMARK_DIR
    / "DeepD3_Benchmark.tif"
)

REAL_PRED_PATH = (
    BENCHMARK_DIR
    / "predictions"
    / "real_32F_94nm.prediction"
)

SYNTH_PRED_PATH = (
    BENCHMARK_DIR
    / "predictions"
    / "synthetic_32F_94nm.prediction"
)

OUTPUT_DIR = (
    BENCHMARK_DIR
    / "qualitative"
)


# =========================================================
# Frozen segmentation thresholds
# =========================================================

REAL_DENDRITE_THRESHOLD = 0.01
REAL_SPINE_THRESHOLD = 0.25

SYNTH_DENDRITE_THRESHOLD = 0.01
SYNTH_SPINE_THRESHOLD = 0.54


# =========================================================
# Selected benchmark region
# Previous crop 02
# =========================================================

Z_START = 30
Z_END = 45

Y0 = 100
Y1 = 300

X0 = 550
X1 = 900


# =========================================================
# Prediction loading
# =========================================================

def load_prediction(path):

    print(f"\nLoading prediction: {path}")

    data = fl.load(path)

    if isinstance(data, dict) and "prediction" in data:

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

        prediction = np.asarray(data)

        if (
            prediction.ndim != 4
            or prediction.shape[-1] < 2
        ):

            raise RuntimeError(
                f"Unknown prediction format: {path}"
            )

        dendrite = prediction[..., 0]
        spine = prediction[..., 1]

    print("Dendrite shape:", dendrite.shape)
    print("Spine shape   :", spine.shape)

    return dendrite, spine


# =========================================================
# Spine GT
# =========================================================

def load_spine_mask(path):

    data = fl.load(path)

    mask = np.asarray(
        data["mask"]
    )

    # Stored as Z,X,Y
    # Convert to Z,Y,X

    mask = mask.transpose(
        0,
        2,
        1,
    )

    return mask > 0


def load_spine_intersection():

    U = load_spine_mask(
        BENCHMARK_DIR / "Segmentation_U.mask"
    )

    V = load_spine_mask(
        BENCHMARK_DIR / "Segmentation_V.mask"
    )

    W = load_spine_mask(
        BENCHMARK_DIR / "Segmentation_W.mask"
    )

    intersection = U & V & W

    print(
        "\nSpine GT intersection voxels:",
        int(intersection.sum()),
    )

    return intersection


# =========================================================
# Dendrite GT rasterization
# =========================================================

def rasterize_dendrite_swc(
    swc_path,
    reference_tif,
    spacing=(1, 1, 1),
):

    print(f"\nRasterizing: {swc_path}")

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

    node_id_set = set(node_ids)

    for node_id in node_ids:

        row = swc.loc[node_id]

        parent_id = int(
            row.parent
        )

        if parent_id <= 0:
            continue

        if parent_id not in node_id_set:
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

        except Exception:
            pass

    return (
        converter.stack > 0
    )


def load_dendrite_intersection():

    U = rasterize_dendrite_swc(
        BENCHMARK_DIR / "Dendrite_U.swc",
        IMAGE_PATH,
    )

    V = rasterize_dendrite_swc(
        BENCHMARK_DIR / "Dendrite_V.swc",
        IMAGE_PATH,
    )

    W = rasterize_dendrite_swc(
        BENCHMARK_DIR / "Dendrite_W.swc",
        IMAGE_PATH,
    )

    intersection = (
        U
        & V
        & W
    )

    print(
        "\nDendrite GT intersection voxels:",
        int(intersection.sum()),
    )

    return intersection


# =========================================================
# Helpers
# =========================================================

def max_project(volume):

    return np.max(
        volume[
            Z_START:Z_END
        ],
        axis=0,
    )


def crop(image):

    return image[
        Y0:Y1,
        X0:X1,
    ]


def normalize_image(image):

    image = image.astype(
        np.float32
    )

    lo = np.percentile(
        image,
        1
    )

    hi = np.percentile(
        image,
        99.5
    )

    if hi <= lo:

        return np.zeros_like(
            image,
            dtype=np.float32,
        )

    image = (
        image - lo
    ) / (
        hi - lo
    )

    return np.clip(
        image,
        0,
        1,
    )


# =========================================================
# Main
# =========================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Load benchmark
    # -----------------------------------------------------

    print("Loading benchmark image...")

    benchmark = tifffile.imread(
        IMAGE_PATH
    )

    print(
        "Benchmark shape:",
        benchmark.shape,
    )

    # -----------------------------------------------------
    # Load predictions
    # -----------------------------------------------------

    real_dend_prob, real_spine_prob = load_prediction(
        REAL_PRED_PATH
    )

    synth_dend_prob, synth_spine_prob = load_prediction(
        SYNTH_PRED_PATH
    )

    # -----------------------------------------------------
    # Load expert GT intersections
    # -----------------------------------------------------

    spine_gt = load_spine_intersection()

    dendrite_gt = load_dendrite_intersection()

    # -----------------------------------------------------
    # Threshold predictions
    # -----------------------------------------------------

    real_spine_mask = (
        real_spine_prob
        >= REAL_SPINE_THRESHOLD
    )

    synth_spine_mask = (
        synth_spine_prob
        >= SYNTH_SPINE_THRESHOLD
    )

    real_dend_mask = (
        real_dend_prob
        >= REAL_DENDRITE_THRESHOLD
    )

    synth_dend_mask = (
        synth_dend_prob
        >= SYNTH_DENDRITE_THRESHOLD
    )

    # -----------------------------------------------------
    # Projection + crop
    # -----------------------------------------------------

    benchmark_crop = crop(
        normalize_image(
            max_project(
                benchmark
            )
        )
    )

    spine_gt_crop = crop(
        max_project(
            spine_gt
        )
    )

    real_spine_crop = crop(
        max_project(
            real_spine_mask
        )
    )

    synth_spine_crop = crop(
        max_project(
            synth_spine_mask
        )
    )

    dendrite_gt_crop = crop(
        max_project(
            dendrite_gt
        )
    )

    real_dend_crop = crop(
        max_project(
            real_dend_mask
        )
    )

    synth_dend_crop = crop(
        max_project(
            synth_dend_mask
        )
    )

    # =====================================================
    # Plot
    # =====================================================

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(16, 8),
    )

    # -----------------------------------------------------
    # Spine row
    # -----------------------------------------------------

    axes[0, 0].imshow(
        benchmark_crop,
        cmap="gray",
    )

    axes[0, 0].set_title(
        "Real benchmark"
    )

    axes[0, 1].imshow(
        spine_gt_crop,
        cmap="gray",
    )

    axes[0, 1].set_title(
        "Expert intersection\n(U ∩ V ∩ W)"
    )

    axes[0, 2].imshow(
        real_spine_crop,
        cmap="gray",
    )

    axes[0, 2].set_title(
        "Real-trained\nDeepD3"
    )

    axes[0, 3].imshow(
        synth_spine_crop,
        cmap="gray",
    )

    axes[0, 3].set_title(
        "Synthetic-trained\nDeepD3"
    )

    # -----------------------------------------------------
    # Dendrite row
    # -----------------------------------------------------

    axes[1, 0].imshow(
        benchmark_crop,
        cmap="gray",
    )

    axes[1, 0].set_title(
        "Real benchmark"
    )

    axes[1, 1].imshow(
        dendrite_gt_crop,
        cmap="gray",
    )

    axes[1, 1].set_title(
        "Expert intersection\n(U ∩ V ∩ W)"
    )

    axes[1, 2].imshow(
        real_dend_crop,
        cmap="gray",
    )

    axes[1, 2].set_title(
        "Real-trained\nDeepD3"
    )

    axes[1, 3].imshow(
        synth_dend_crop,
        cmap="gray",
    )

    axes[1, 3].set_title(
        "Synthetic-trained\nDeepD3"
    )

    # -----------------------------------------------------
    # Row labels
    # -----------------------------------------------------

    axes[0, 0].set_ylabel(
        "Spine",
        fontsize=14,
    )

    axes[1, 0].set_ylabel(
        "Dendrite",
        fontsize=14,
    )

    for ax in axes.flat:

        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "Representative region from the DeepD3 real benchmark",
        fontsize=16,
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "deepd3_benchmark_comparison.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\nSaved:")
    print(output_path)


if __name__ == "__main__":
    main()