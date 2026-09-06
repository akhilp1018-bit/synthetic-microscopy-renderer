"""
run_deepd3_dataset.py
---------------------

Run pretrained DeepD3 models on a synthetic microscopy dataset.

The script processes all instances in a selected dataset split and runs
DeepD3 inference on the requested input image of each instance.

Expected project structure
--------------------------

deepd3/
├── models/
│   ├── DeepD3_32F.h5
│   └── DeepD3_32F_94nm.h5
└── run_deepd3_dataset.py

outputs/
└── synthetic_dataset_v1/
    ├── train/
    ├── validation/
    └── test/

The pretrained DeepD3 model files are not included in this repository
and must be downloaded separately.

Usage
-----

Run from the repository root.

Process the complete test split:

    python deepd3/run_deepd3_dataset.py

Process only one instance:

    python deepd3/run_deepd3_dataset.py --max-instances 1

Process another split:

    python deepd3/run_deepd3_dataset.py --split validation

Use another input image filename:

    python deepd3/run_deepd3_dataset.py \
        --image-name clean.tif

Use a custom dataset:

    python deepd3/run_deepd3_dataset.py \
        --dataset outputs/another_dataset

Use custom model paths:

    python deepd3/run_deepd3_dataset.py \
        --model-32f path/to/DeepD3_32F.h5 \
        --model-32f-94nm path/to/DeepD3_32F_94nm.h5

Overwrite existing predictions:

    python deepd3/run_deepd3_dataset.py --overwrite

Output
------

Predictions are stored inside each processed dataset instance:

instance_XXXXXX/
└── deepd3_predictions/
    ├── 32F.prediction
    └── 32F_94nm.prediction
"""

from __future__ import annotations

import argparse
from pathlib import Path

import flammkuchen as fl
import numpy as np

from deepd3.core.analysis import Stack


# ==========================================================
# Default paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET = (
    PROJECT_ROOT
    / "outputs"
    / "synthetic_dataset_v1"
)

DEFAULT_MODEL_32F = (
    PROJECT_ROOT
    / "deepd3"
    / "models"
    / "DeepD3_32F.h5"
)

DEFAULT_MODEL_32F_94NM = (
    PROJECT_ROOT
    / "deepd3"
    / "models"
    / "DeepD3_32F_94nm.h5"
)


# ==========================================================
# Arguments
# ==========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run pretrained DeepD3 models on synthetic data "
            "and save raw, uncleaned probability maps."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
    )

    parser.add_argument(
        "--model-32f",
        type=Path,
        default=DEFAULT_MODEL_32F,
    )

    parser.add_argument(
        "--model-32f-94nm",
        type=Path,
        default=DEFAULT_MODEL_32F_94NM,
    )

    parser.add_argument(
        "--image-name",
        default="noisy.tif",
    )

    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--inset-size",
        type=int,
        default=96,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# ==========================================================
# Helpers
# ==========================================================

def require_file(path: Path, description: str):
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} not found:\n"
            f"  {path}"
        )


def find_input_image(
    instance_dir: Path,
    image_name: str,
) -> Path:

    image_path = (
        instance_dir
        / image_name
    )

    require_file(
        image_path,
        "Input image",
    )

    return image_path


# ==========================================================
# Raw DeepD3 inference
# ==========================================================

def run_model(
    image_path: Path,
    model_path: Path,
    model_tag: str,
    output_dir: Path,
    tile_size: int,
    inset_size: int,
    overwrite: bool,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_output = (
        output_dir
        / f"{model_tag}.prediction"
    )

    if (
        final_output.exists()
        and not overwrite
    ):
        print(
            f"    {model_tag}: already exists, skipping"
        )
        return final_output

    print(
        f"    Running model: {model_tag}"
    )

    print(
        f"      Model : {model_path}"
    )

    print(
        f"      Image : {image_path}"
    )

    # ------------------------------------------------------
    # Load stack
    # ------------------------------------------------------

    stack = Stack(
        str(image_path)
    )
    
    stack.predictInset(
        str(model_path),
        tile_size,
        inset_size,
    )

    prediction = np.asarray(
        stack.prediction,
        dtype=np.float32,
    )

    if prediction.ndim < 4:
        raise ValueError(
            f"Unexpected DeepD3 prediction shape: "
            f"{prediction.shape}"
        )

    if prediction.shape[-1] < 2:
        raise ValueError(
            f"Expected at least two prediction channels, "
            f"got shape {prediction.shape}"
        )

    dendrites = (
        prediction[..., 0]
        .astype(
            np.float32,
            copy=True,
        )
    )

    spines = (
        prediction[..., 1]
        .astype(
            np.float32,
            copy=True,
        )
    )

    # ------------------------------------------------------
    # Diagnostic information
    # ------------------------------------------------------

    print(
        "      Dendrite probability range: "
        f"{float(np.nanmin(dendrites)):.8f} "
        f"to "
        f"{float(np.nanmax(dendrites)):.8f}"
    )

    print(
        "      Spine probability range   : "
        f"{float(np.nanmin(spines)):.8f} "
        f"to "
        f"{float(np.nanmax(spines)):.8f}"
    )

    print(
        "      Dendrite unique values    : "
        f"{len(np.unique(dendrites))}"
    )

    print(
        "      Spine unique values       : "
        f"{len(np.unique(spines))}"
    )

    # ------------------------------------------------------
    # Save RAW probabilities
    # ------------------------------------------------------

    fl.save(
        str(final_output),
        {
            "dendrites":
                dendrites,

            "spines":
                spines,
        },
        compression="blosc",
    )

    print(
        f"      Saved raw probabilities: "
        f"{final_output}"
    )

    return final_output


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

    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found:\n"
            f"  {dataset_root}"
        )

    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset split not found:\n"
            f"  {split_dir}"
        )

    model_32f = (
        args.model_32f.resolve()
    )

    model_32f_94nm = (
        args.model_32f_94nm.resolve()
    )

    require_file(
        model_32f,
        "DeepD3 32F model",
    )

    require_file(
        model_32f_94nm,
        "DeepD3 32F 94 nm model",
    )

    models = [
        (
            "32F",
            model_32f,
        ),

        (
            "32F_94nm",
            model_32f_94nm,
        ),
    ]

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

    print()
    print("=" * 70)
    print(
        "DeepD3 RAW probability inference"
    )
    print("=" * 70)

    print(
        f"Dataset     : {dataset_root}"
    )

    print(
        f"Split       : {args.split}"
    )

    print(
        f"Instances   : {len(instance_dirs)}"
    )

    print(
        f"Input image : {args.image_name}"
    )

    print(
        f"Tile size   : {args.tile_size}"
    )

    print(
        f"Inset size  : {args.inset_size}"
    )

    print("=" * 70)

    completed = 0

    for index, instance_dir in enumerate(
        instance_dirs,
        start=1,
    ):

        print()
        print(
            f"[{index}/{len(instance_dirs)}] "
            f"{instance_dir.name}"
        )

        image_path = (
            find_input_image(
                instance_dir,
                args.image_name,
            )
        )

        prediction_dir = (
            instance_dir
            / "deepd3_predictions"
        )

        for (
            model_tag,
            model_path,
        ) in models:

            run_model(
                image_path=image_path,
                model_path=model_path,
                model_tag=model_tag,
                output_dir=prediction_dir,
                tile_size=args.tile_size,
                inset_size=args.inset_size,
                overwrite=args.overwrite,
            )

        completed += 1

    print()
    print("=" * 70)
    print(
        "DeepD3 raw inference complete"
    )
    print("=" * 70)

    print(
        f"Processed instances : {completed}"
    )

    print(
        f"Models              : {len(models)}"
    )

    print(
        f"Total predictions   : "
        f"{completed * len(models)}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()