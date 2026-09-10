from pathlib import Path

import numpy as np
import tifffile
import flammkuchen as fl
import matplotlib.pyplot as plt


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
# Frozen thresholds
# =========================================================

REAL_SPINE_THRESHOLD = 0.25
SYNTH_SPINE_THRESHOLD = 0.54


# =========================================================
# Final selected benchmark region
# =========================================================
#
# This is the previous Crop 02.
# We do not need to call it "Crop 02" in the thesis.
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

    print(f"Loading prediction: {path}")

    data = fl.load(path)

    if isinstance(data, dict) and "prediction" in data:

        prediction = np.asarray(
            data["prediction"]
        )

        spine = (
            prediction[..., 1]
        )

    elif (
        isinstance(data, dict)
        and "spines" in data
    ):

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
                f"Unknown prediction format: {path}"
            )

        spine = (
            prediction[..., 1]
        )

    print(
        "Spine shape:",
        spine.shape,
    )

    print(
        "Spine range:",
        float(spine.min()),
        "->",
        float(spine.max()),
    )

    return spine


# =========================================================
# Spine annotation loading
# =========================================================

def load_spine_mask(path):

    data = fl.load(path)

    mask = np.asarray(
        data["mask"]
    )

    # Stored benchmark format:
    #
    # Z, X, Y
    #
    # Convert to:
    #
    # Z, Y, X

    mask = mask.transpose(
        0,
        2,
        1,
    )

    return (
        mask > 0
    )


def load_spine_intersection():

    U = load_spine_mask(
        BENCHMARK_DIR
        / "Segmentation_U.mask"
    )

    V = load_spine_mask(
        BENCHMARK_DIR
        / "Segmentation_V.mask"
    )

    W = load_spine_mask(
        BENCHMARK_DIR
        / "Segmentation_W.mask"
    )

    intersection = (
        U
        & V
        & W
    )

    print(
        "Intersection voxels:",
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
    # Load real benchmark image
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
    # Load expert intersection
    # -----------------------------------------------------

    spine_intersection = (
        load_spine_intersection()
    )

    # -----------------------------------------------------
    # Load model probabilities
    # -----------------------------------------------------

    real_spine_prob = load_prediction(
        REAL_PRED_PATH
    )

    synth_spine_prob = load_prediction(
        SYNTH_PRED_PATH
    )

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

    # -----------------------------------------------------
    # Maximum-intensity projections
    # -----------------------------------------------------

    benchmark_projection = max_project(
        benchmark
    )

    gt_projection = max_project(
        spine_intersection
    )

    real_projection = max_project(
        real_spine_mask
    )

    synth_projection = max_project(
        synth_spine_mask
    )

    # -----------------------------------------------------
    # Crop selected region
    # -----------------------------------------------------

    benchmark_crop = crop(
        normalize_image(
            benchmark_projection
        )
    )

    gt_crop = crop(
        gt_projection
    )

    real_crop = crop(
        real_projection
    )

    synth_crop = crop(
        synth_projection
    )

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4),
    )

    axes[0].imshow(
        benchmark_crop,
        cmap="gray",
    )

    axes[0].set_title(
        "Real benchmark"
    )

    axes[1].imshow(
        gt_crop,
        cmap="gray",
    )

    axes[1].set_title(
        "Expert intersection\n(U ∩ V ∩ W)"
    )

    axes[2].imshow(
        real_crop,
        cmap="gray",
    )

    axes[2].set_title(
        "Real-trained\nDeepD3"
    )

    axes[3].imshow(
        synth_crop,
        cmap="gray",
    )

    axes[3].set_title(
        "Synthetic-trained\nDeepD3"
    )

    for ax in axes:
        ax.axis(
            "off"
        )

    fig.suptitle(
        "Representative region from the DeepD3 benchmark"
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "benchmark_spine_final.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print("\nSaved:")
    print(output_path)


if __name__ == "__main__":
    main()