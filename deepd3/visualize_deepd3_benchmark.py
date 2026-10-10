"""
visualize_deepd3_benchmark.py
-----------------------------

Create a qualitative comparison of real-trained and synthetic-trained
DeepD3 models on the real DeepD3 benchmark.

Rows:
    (a) Spine segmentation
    (b) Dendrite segmentation
    (c) Spine detection

Columns:
    1. Real benchmark image
    2. Expert annotation
    3. Real-trained DeepD3
    4. Synthetic-trained DeepD3

For segmentation:
    Expert annotation = intersection of U, V and W.

For spine detection:
    Expert annotation = spine clusters identified by all seven raters.

The same frozen validation thresholds and the same peak-detection
procedure used for quantitative benchmark evaluation are used here.

Run from repository root:

    python deepd3/visualize_deepd3_benchmark.py

Output:

    benchmarks/deepd3/qualitative/deepd3_benchmark_comparison.png
"""

from pathlib import Path
import json

import numpy as np
import tifffile
import flammkuchen as fl
import matplotlib.pyplot as plt

from deepd3.core.dendrite import (
    DendriteSWC,
    xyzr,
    line_w_sphere,
)


from evaluate_deepd3_benchmark import (
    load_benchmark_spine_centers,
    detect_spine_peaks,
)


# =========================================================
# Paths
# =========================================================

BENCHMARK_DIR = Path("benchmarks/deepd3")

IMAGE_PATH = (
    BENCHMARK_DIR
    / "DeepD3_Benchmark.tif"
)

ANNOTATION_CSV = (
    BENCHMARK_DIR
    / "Annotations_and_Clusters.csv"
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

THRESHOLDS_PATH = Path(
    "outputs/synthetic_dataset_v1/"
    "deepd3_evaluation/validation/"
    "selected_thresholds.json"
)

OUTPUT_DIR = (
    BENCHMARK_DIR
    / "qualitative"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "deepd3_benchmark_comparison.png"
)


# =========================================================
# Representative benchmark region
# =========================================================

Z_START = 30
Z_END = 45

Y0 = 100
Y1 = 300

X0 = 550
X1 = 900


# =========================================================
# Load frozen validation thresholds
# =========================================================

def load_frozen_thresholds(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Threshold file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        selected = json.load(file)

    real = selected[
        "32F_94nm"
    ]

    synthetic = selected[
        "synthetic_32F_94nm"
    ]

    thresholds = {

        "real_dendrite":
            float(
                real[
                    "dendrite_segmentation"
                ]
            ),

        "real_spine":
            float(
                real[
                    "spine_segmentation"
                ]
            ),

        "real_detection":
            float(
                real[
                    "spine_detection"
                ]
            ),

        "synthetic_dendrite":
            float(
                synthetic[
                    "dendrite_segmentation"
                ]
            ),

        "synthetic_spine":
            float(
                synthetic[
                    "spine_segmentation"
                ]
            ),

        "synthetic_detection":
            float(
                synthetic[
                    "spine_detection"
                ]
            ),
    }

    return thresholds


# =========================================================
# Load DeepD3 predictions
# =========================================================

def load_prediction(path):

    print("\nLoading prediction:")
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
                f"Unknown prediction format: {path}"
            )

        dendrite = prediction[..., 0]
        spine = prediction[..., 1]

    dendrite = np.nan_to_num(
        dendrite,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    spine = np.nan_to_num(
        spine,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    print(
        "Dendrite shape:",
        dendrite.shape,
    )

    print(
        "Spine shape   :",
        spine.shape,
    )

    return (
        dendrite,
        spine,
    )


# =========================================================
# Spine segmentation GT
# =========================================================

def load_spine_mask(path):

    data = fl.load(
        path
    )

    mask = np.asarray(
        data["mask"]
    )

    # Benchmark mask storage:
    # Z, X, Y
    #
    # Convert to:
    # Z, Y, X

    mask = mask.transpose(
        0,
        2,
        1,
    )

    return mask > 0


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
        "\nSpine intersection voxels:",
        int(
            intersection.sum()
        ),
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

    print(
        "\nRasterizing:",
        swc_path,
    )

    converter = DendriteSWC(
        spacing=list(
            spacing
        )
    )

    converter.open(
        str(
            swc_path
        ),
        str(
            reference_tif
        ),
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

    for node_id in node_ids:

        row = swc.loc[
            node_id
        ]

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
        BENCHMARK_DIR
        / "Dendrite_U.swc",
        IMAGE_PATH,
    )

    V = rasterize_dendrite_swc(
        BENCHMARK_DIR
        / "Dendrite_V.swc",
        IMAGE_PATH,
    )

    W = rasterize_dendrite_swc(
        BENCHMARK_DIR
        / "Dendrite_W.swc",
        IMAGE_PATH,
    )

    intersection = (
        U
        & V
        & W
    )

    print(
        "\nDendrite intersection voxels:",
        int(
            intersection.sum()
        ),
    )

    return intersection


# =========================================================
# Image helpers
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

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    lo = np.percentile(
        image,
        1,
    )

    hi = np.percentile(
        image,
        99.5,
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
# Detection helpers
# =========================================================

def centers_in_selected_region(
    centers_zyx,
):

    centers_zyx = np.asarray(
        centers_zyx,
        dtype=np.float64,
    )

    selected = []

    for center in centers_zyx:

        z, y, x = center

        if not (
            Z_START <= z < Z_END
            and
            Y0 <= y < Y1
            and
            X0 <= x < X1
        ):
            continue

        selected.append(
            [
                z,
                y - Y0,
                x - X0,
            ]
        )

    if not selected:

        return np.empty(
            (0, 3),
            dtype=np.float64,
        )

    return np.asarray(
        selected,
        dtype=np.float64,
    )


def show_gt_centers(
    ax,
    image,
    centers,
):

    ax.imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    if len(centers) == 0:
        return

    ax.scatter(
        centers[:, 2],
        centers[:, 1],
        marker="o",
        facecolors="none",
        edgecolors="cyan",
        s=45,
        linewidths=1.2,
    )


def show_prediction_centers(
    ax,
    image,
    centers,
):

    ax.imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    if len(centers) == 0:
        return

    ax.scatter(
        centers[:, 2],
        centers[:, 1],
        marker="x",
        color="magenta",
        s=42,
        linewidths=1.3,
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
    # Frozen thresholds
    # -----------------------------------------------------

    thresholds = (
        load_frozen_thresholds(
            THRESHOLDS_PATH
        )
    )

    print(
        "\n========================================"
    )

    print(
        "Frozen validation thresholds"
    )

    print(
        "========================================"
    )

    print(
        "Real-trained:"
    )

    print(
        "  Dendrite segmentation:",
        thresholds[
            "real_dendrite"
        ],
    )

    print(
        "  Spine segmentation   :",
        thresholds[
            "real_spine"
        ],
    )

    print(
        "  Spine detection      :",
        thresholds[
            "real_detection"
        ],
    )

    print(
        "\nSynthetic-trained:"
    )

    print(
        "  Dendrite segmentation:",
        thresholds[
            "synthetic_dendrite"
        ],
    )

    print(
        "  Spine segmentation   :",
        thresholds[
            "synthetic_spine"
        ],
    )

    print(
        "  Spine detection      :",
        thresholds[
            "synthetic_detection"
        ],
    )

    # -----------------------------------------------------
    # Benchmark image
    # -----------------------------------------------------

    print(
        "\nLoading benchmark image..."
    )

    benchmark = tifffile.imread(
        IMAGE_PATH
    )

    print(
        "Benchmark shape:",
        benchmark.shape,
    )

    # -----------------------------------------------------
    # Predictions
    # -----------------------------------------------------

    (
        real_dend_prob,
        real_spine_prob,
    ) = load_prediction(
        REAL_PRED_PATH
    )

    (
        synth_dend_prob,
        synth_spine_prob,
    ) = load_prediction(
        SYNTH_PRED_PATH
    )

    # -----------------------------------------------------
    # Shape check
    # -----------------------------------------------------

    for name, volume in [

        (
            "real dendrite",
            real_dend_prob,
        ),

        (
            "real spine",
            real_spine_prob,
        ),

        (
            "synthetic dendrite",
            synth_dend_prob,
        ),

        (
            "synthetic spine",
            synth_spine_prob,
        ),

    ]:

        if volume.shape != benchmark.shape:

            raise RuntimeError(
                f"{name} shape {volume.shape} "
                f"does not match benchmark "
                f"{benchmark.shape}"
            )

    # -----------------------------------------------------
    # Segmentation GT
    # -----------------------------------------------------

    spine_gt = (
        load_spine_intersection()
    )

    dendrite_gt = (
        load_dendrite_intersection()
    )

    # -----------------------------------------------------
    # Segmentation predictions
    # -----------------------------------------------------

    real_spine_mask = (
        real_spine_prob
        >= thresholds[
            "real_spine"
        ]
    )

    synth_spine_mask = (
        synth_spine_prob
        >= thresholds[
            "synthetic_spine"
        ]
    )

    real_dend_mask = (
        real_dend_prob
        >= thresholds[
            "real_dendrite"
        ]
    )

    synth_dend_mask = (
        synth_dend_prob
        >= thresholds[
            "synthetic_dendrite"
        ]
    )

    # =====================================================
    # Spine detection
    # =====================================================

    # Load exactly the same multi-rater benchmark clusters
    # used in quantitative evaluation.

    cluster_df = (
        load_benchmark_spine_centers(
            ANNOTATION_CSV
        )
    )

    # Thesis comparison uses the strictest benchmark GT:
    # clusters identified by all seven raters.

    gt_subset = cluster_df[
        cluster_df[
            "n_raters"
        ] >= 7
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

    # Use EXACTLY the same peak-detection function
    # as quantitative benchmark evaluation.

    real_detection_centers = (
        detect_spine_peaks(
            real_spine_prob,
            thresholds[
                "real_detection"
            ],
        )
    )

    synth_detection_centers = (
        detect_spine_peaks(
            synth_spine_prob,
            thresholds[
                "synthetic_detection"
            ],
        )
    )

    print(
        "\n========================================"
    )

    print(
        "Full benchmark detection counts"
    )

    print(
        "========================================"
    )

    print(
        "7-rater GT centers          :",
        len(
            gt_centers
        ),
    )

    print(
        "Real-trained predicted peaks:",
        len(
            real_detection_centers
        ),
    )

    print(
        "Synthetic-trained peaks     :",
        len(
            synth_detection_centers
        ),
    )

    # -----------------------------------------------------
    # Create MIPs for selected Z region
    # -----------------------------------------------------

    benchmark_mip = (
        normalize_image(
            max_project(
                benchmark
            )
        )
    )

    benchmark_crop = crop(
        benchmark_mip
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

    # -----------------------------------------------------
    # Detection centers inside displayed region
    # -----------------------------------------------------

    gt_crop_centers = (
        centers_in_selected_region(
            gt_centers
        )
    )

    real_crop_centers = (
        centers_in_selected_region(
            real_detection_centers
        )
    )

    synth_crop_centers = (
        centers_in_selected_region(
            synth_detection_centers
        )
    )

    print(
        "\n========================================"
    )

    print(
        "Detection centers in displayed region"
    )

    print(
        "========================================"
    )

    print(
        "Expert GT        :",
        len(
            gt_crop_centers
        ),
    )

    print(
        "Real-trained     :",
        len(
            real_crop_centers
        ),
    )

    print(
        "Synthetic-trained:",
        len(
            synth_crop_centers
        ),
    )

    # =====================================================
    # Plot
    # =====================================================

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(
            16,
            11,
        ),
    )

    # =====================================================
    # Row (a): Spine segmentation
    # =====================================================

    axes[0, 0].imshow(
        benchmark_crop,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[0, 1].imshow(
        spine_gt_crop,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[0, 2].imshow(
        real_spine_crop,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[0, 3].imshow(
        synth_spine_crop,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    # =====================================================
    # Row (b): Dendrite segmentation
    # =====================================================

    axes[1, 0].imshow(
        benchmark_crop,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[1, 1].imshow(
        dendrite_gt_crop,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[1, 2].imshow(
        real_dend_crop,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[1, 3].imshow(
        synth_dend_crop,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    # =====================================================
    # Row (c): Spine detection
    # =====================================================

    # Original benchmark image

    axes[2, 0].imshow(
        benchmark_crop,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    # Expert 7-rater centers

    show_gt_centers(
        axes[2, 1],
        benchmark_crop,
        gt_crop_centers,
    )

    # Real-trained predicted centers

    show_prediction_centers(
        axes[2, 2],
        benchmark_crop,
        real_crop_centers,
    )

    # Synthetic-trained predicted centers

    show_prediction_centers(
        axes[2, 3],
        benchmark_crop,
        synth_crop_centers,
    )

    # =====================================================
    # Column titles
    # =====================================================

    column_titles = [

        "Real benchmark",

        "Expert annotation",

        "Real-trained\nDeepD3",

        "Synthetic-trained\nDeepD3",

    ]

    for column, title in enumerate(
        column_titles
    ):

        axes[
            0,
            column,
        ].set_title(
            title,
            fontsize=13,
        )

    # =====================================================
    # Row labels
    # =====================================================

    axes[
        0,
        0,
    ].set_ylabel(
        "(a) Spine segmentation",
        fontsize=12,
        labelpad=10,
    )

    axes[
        1,
        0,
    ].set_ylabel(
        "(b) Dendrite segmentation",
        fontsize=12,
        labelpad=10,
    )

    axes[
        2,
        0,
    ].set_ylabel(
        "(c) Spine detection",
        fontsize=12,
        labelpad=10,
    )

    # =====================================================
    # Remove ticks
    # =====================================================

    for ax in axes.flat:

        ax.set_xticks([])
        ax.set_yticks([])

    # No large figure title.
    # Explanation will be in the thesis caption.

    fig.tight_layout(
        pad=1.0,
    )

    # =====================================================
    # Save
    # =====================================================

    fig.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        "\n========================================"
    )

    print(
        "Saved qualitative benchmark figure"
    )

    print(
        "========================================"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()