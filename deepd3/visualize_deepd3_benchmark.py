from pathlib import Path
import argparse

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
    / "qualitative"
)


# =========================================================
# Frozen thresholds
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


def load_spine_gt(benchmark_dir):

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

    return {
        "U": U,
        "V": V,
        "W": W,
        "intersection": intersection,
        "union": union,
    }


# =========================================================
# Dendrite GT rasterization
# =========================================================

def rasterize_dendrite_swc(
    swc_path,
    reference_tif,
    spacing=(1, 1, 1),
):

    print(f"\nRasterizing {swc_path}")

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


def load_dendrite_gt(
    benchmark_dir,
    reference_tif,
):

    masks = {}

    for rater in [
        "U",
        "V",
        "W",
    ]:

        masks[rater] = rasterize_dendrite_swc(
            swc_path=(
                benchmark_dir
                / f"Dendrite_{rater}.swc"
            ),
            reference_tif=reference_tif,
            spacing=(1, 1, 1),
        )

    return masks


# =========================================================
# Projection helpers
# =========================================================

def normalize_image(img):

    img = img.astype(
        np.float32
    )

    lo = np.percentile(
        img,
        1
    )

    hi = np.percentile(
        img,
        99.5
    )

    if hi <= lo:
        return np.zeros_like(
            img,
            dtype=np.float32
        )

    img = (
        img - lo
    ) / (
        hi - lo
    )

    return np.clip(
        img,
        0,
        1,
    )


def max_project(
    volume,
    z_start,
    z_end,
):

    z_start = max(
        0,
        z_start,
    )

    z_end = min(
        volume.shape[0],
        z_end,
    )

    return np.max(
        volume[
            z_start:z_end
        ],
        axis=0,
    )


# =========================================================
# Crop helper
# =========================================================

def crop_xy(
    image,
    y0,
    y1,
    x0,
    x1,
):

    return image[
        y0:y1,
        x0:x1,
    ]


# =========================================================
# Spine qualitative figure
# =========================================================

def make_spine_figure(
    image,
    spine_gt,
    real_spine,
    synth_spine,
    z_start,
    z_end,
    y0,
    y1,
    x0,
    x1,
    output_path,
):

    raw_projection = max_project(
        image,
        z_start,
        z_end,
    )

    gt_projection = max_project(
        spine_gt["union"],
        z_start,
        z_end,
    )

    gt_intersection = max_project(
        spine_gt["intersection"],
        z_start,
        z_end,
    )

    real_projection = max_project(
        real_spine,
        z_start,
        z_end,
    )

    synth_projection = max_project(
        synth_spine,
        z_start,
        z_end,
    )

    raw_crop = crop_xy(
        normalize_image(
            raw_projection
        ),
        y0,
        y1,
        x0,
        x1,
    )

    union_crop = crop_xy(
        gt_projection,
        y0,
        y1,
        x0,
        x1,
    )

    intersection_crop = crop_xy(
        gt_intersection,
        y0,
        y1,
        x0,
        x1,
    )

    real_crop = crop_xy(
        real_projection,
        y0,
        y1,
        x0,
        x1,
    )

    synth_crop = crop_xy(
        synth_projection,
        y0,
        y1,
        x0,
        x1,
    )

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(18, 4),
    )

    axes[0].imshow(
        raw_crop,
        cmap="gray",
    )

    axes[0].set_title(
        "Real benchmark"
    )

    axes[1].imshow(
        union_crop,
        cmap="gray",
    )

    axes[1].set_title(
        "Spine GT union"
    )

    axes[2].imshow(
        intersection_crop,
        cmap="gray",
    )

    axes[2].set_title(
        "Spine GT intersection"
    )

    axes[3].imshow(
        real_crop,
        cmap="magma",
        vmin=0,
        vmax=1,
    )

    axes[3].set_title(
        "Real-trained\nspine probability"
    )

    axes[4].imshow(
        synth_crop,
        cmap="magma",
        vmin=0,
        vmax=1,
    )

    axes[4].set_title(
        "Synthetic-trained\nspine probability"
    )

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"Spine comparison | "
        f"Z={z_start}:{z_end}, "
        f"Y={y0}:{y1}, "
        f"X={x0}:{x1}"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Saved:",
        output_path,
    )


# =========================================================
# Dendrite qualitative figure
# =========================================================

def make_dendrite_figure(
    image,
    dendrite_gt,
    real_dend,
    synth_dend,
    z_start,
    z_end,
    y0,
    y1,
    x0,
    x1,
    output_path,
    rater="U",
):

    raw_projection = max_project(
        image,
        z_start,
        z_end,
    )

    gt_projection = max_project(
        dendrite_gt[rater],
        z_start,
        z_end,
    )

    real_probability = max_project(
        real_dend,
        z_start,
        z_end,
    )

    synth_probability = max_project(
        synth_dend,
        z_start,
        z_end,
    )

    real_mask = (
        real_probability
        >= SEGMENTATION_THRESHOLDS[
            "real_32F_94nm"
        ]["dendrite"]
    )

    synth_mask = (
        synth_probability
        >= SEGMENTATION_THRESHOLDS[
            "synthetic_32F_94nm"
        ]["dendrite"]
    )

    raw_crop = crop_xy(
        normalize_image(
            raw_projection
        ),
        y0,
        y1,
        x0,
        x1,
    )

    gt_crop = crop_xy(
        gt_projection,
        y0,
        y1,
        x0,
        x1,
    )

    real_crop = crop_xy(
        real_mask,
        y0,
        y1,
        x0,
        x1,
    )

    synth_crop = crop_xy(
        synth_mask,
        y0,
        y1,
        x0,
        x1,
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(15, 4),
    )

    axes[0].imshow(
        raw_crop,
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
        f"Dendrite GT ({rater})"
    )

    axes[2].imshow(
        real_crop,
        cmap="gray",
    )

    axes[2].set_title(
        "Real-trained\nsegmentation"
    )

    axes[3].imshow(
        synth_crop,
        cmap="gray",
    )

    axes[3].set_title(
        "Synthetic-trained\nsegmentation"
    )

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"Dendrite comparison | "
        f"Z={z_start}:{z_end}, "
        f"Y={y0}:{y1}, "
        f"X={x0}:{x1}"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Saved:",
        output_path,
    )


# =========================================================
# Default qualitative crops
# =========================================================

def get_default_crops():

    # These are simply representative starting regions.
    #
    # After viewing them, we can adjust coordinates
    # if a crop is too empty or not informative.

    return [

        {
            "name": "crop_01",
            "z_start": 20,
            "z_end": 35,
            "y0": 40,
            "y1": 220,
            "x0": 150,
            "x1": 450,
        },

        {
            "name": "crop_02",
            "z_start": 30,
            "z_end": 45,
            "y0": 100,
            "y1": 300,
            "x0": 550,
            "x1": 900,
        },

        {
            "name": "crop_03",
            "z_start": 35,
            "z_end": 55,
            "y0": 80,
            "y1": 300,
            "x0": 950,
            "x1": 1300,
        },

    ]


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create qualitative comparison figures "
            "for the DeepD3 benchmark."
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

    benchmark_dir = (
        args.benchmark_dir
    )

    reference_tif = (
        benchmark_dir
        / "DeepD3_Benchmark.tif"
    )

    # =====================================================
    # Load benchmark
    # =====================================================

    print(
        "Loading benchmark image..."
    )

    image = tifffile.imread(
        reference_tif
    )

    print(
        "Benchmark shape:",
        image.shape,
    )

    # =====================================================
    # Load predictions
    # =====================================================

    real_dend, real_spine = load_prediction(
        args.real_prediction
    )

    synth_dend, synth_spine = load_prediction(
        args.synthetic_prediction
    )

    # =====================================================
    # Load GT
    # =====================================================

    spine_gt = load_spine_gt(
        benchmark_dir
    )

    dendrite_gt = load_dendrite_gt(
        benchmark_dir,
        reference_tif,
    )

    # =====================================================
    # Output directory
    # =====================================================

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =====================================================
    # Create qualitative examples
    # =====================================================

    crops = get_default_crops()

    for crop in crops:

        name = (
            crop["name"]
        )

        print(
            f"\nCreating {name}"
        )

        spine_path = (
            args.output_dir
            / f"{name}_spine.png"
        )

        dendrite_path = (
            args.output_dir
            / f"{name}_dendrite.png"
        )

        make_spine_figure(

            image=image,

            spine_gt=spine_gt,

            real_spine=real_spine,

            synth_spine=synth_spine,

            z_start=crop["z_start"],

            z_end=crop["z_end"],

            y0=crop["y0"],

            y1=crop["y1"],

            x0=crop["x0"],

            x1=crop["x1"],

            output_path=spine_path,

        )

        make_dendrite_figure(

            image=image,

            dendrite_gt=dendrite_gt,

            real_dend=real_dend,

            synth_dend=synth_dend,

            z_start=crop["z_start"],

            z_end=crop["z_end"],

            y0=crop["y0"],

            y1=crop["y1"],

            x0=crop["x0"],

            x1=crop["x1"],

            output_path=dendrite_path,

            rater="U",

        )

    print("\n========================================")
    print("Done")
    print("========================================")

    print(
        "Figures saved to:",
        args.output_dir,
    )


if __name__ == "__main__":
    main()