"""
run_deepd3_dataset.py
---------------------

Run DeepD3 inference on the synthetic microscopy dataset.

This script supports:

1. Original pretrained DeepD3 32F model
2. Original pretrained DeepD3 32F 94 nm model
3. Synthetic-trained DeepD3 32F 94 nm model


Expected project structure
--------------------------

deepd3/
├── models/
│   ├── DeepD3_32F.h5
│   ├── DeepD3_32F_94nm.h5
│   └── synthetic_94nm/
│       └── synthetic_32F_94nm_best.h5
│
└── run_deepd3_dataset.py

outputs/
└── synthetic_dataset_v1/
    ├── train/
    ├── validation/
    └── test/


Usage
-----

Run all commands from the repository root.


1. Run original pretrained models on the test split
----------------------------------------------------

    python deepd3/run_deepd3_dataset.py \
        --split test


2. Run original pretrained models on validation
-------------------------------------------------

    python deepd3/run_deepd3_dataset.py \
        --split validation


3. Run the synthetic-trained model on validation
-------------------------------------------------

    python deepd3/run_deepd3_dataset.py \
        --dataset outputs/synthetic_dataset_v1 \
        --split validation \
        --model-synthetic-94nm \
        deepd3/models/synthetic_94nm/synthetic_32F_94nm_best.h5

If predictions from the original pretrained models already exist,
they are skipped automatically. The synthetic-trained prediction is
then generated separately.


4. Run the synthetic-trained model on the test split
-----------------------------------------------------

    python deepd3/run_deepd3_dataset.py \
        --dataset outputs/synthetic_dataset_v1 \
        --split test \
        --model-synthetic-94nm \
        deepd3/models/synthetic_94nm/synthetic_32F_94nm_best.h5


5. Process only one instance
-----------------------------

Useful for testing the inference pipeline before processing a
complete split:

    python deepd3/run_deepd3_dataset.py \
        --split validation \
        --model-synthetic-94nm \
        deepd3/models/synthetic_94nm/synthetic_32F_94nm_best.h5 \
        --max-instances 1


6. Use another input image
---------------------------

The default input is:

    noisy.tif

To use another image, for example clean.tif:

    python deepd3/run_deepd3_dataset.py \
        --split validation \
        --image-name clean.tif


7. Overwrite existing predictions
----------------------------------

    python deepd3/run_deepd3_dataset.py \
        --split validation \
        --overwrite

WARNING:
Using --overwrite causes predictions for all requested models to be
recomputed and existing prediction files to be replaced.

Do not use --overwrite when adding the synthetic-trained model if the
existing original-model predictions should be preserved.


Output
------

Predictions are stored inside each dataset instance:

instance_XXXXXX/
└── deepd3_predictions/
    ├── 32F.prediction
    ├── 32F_94nm.prediction
    └── synthetic_32F_94nm.prediction


Prediction file contents
------------------------

Each .prediction file contains:

    dendrites  -> raw dendrite probability volume
    spines     -> raw spine probability volume

The probability volumes have shape:

    (Z, Y, X)

For the current synthetic dataset:

    (64, 128, 128)


Model naming
------------

32F.prediction
    Original DeepD3 32F model.

32F_94nm.prediction
    Original pretrained DeepD3 32F model for 94 nm resolution.

synthetic_32F_94nm.prediction
    DeepD3 32F model trained on the synthetic microscopy dataset.
    The saved model corresponds to the best training checkpoint
    selected using validation loss.
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

DEFAULT_MODEL_SYNTHETIC_94NM = (
    PROJECT_ROOT
    / "deepd3"
    / "models"
    / "synthetic_94nm"
    / "synthetic_32F_94nm_best.h5"
)


# ==========================================================
# Arguments
# ==========================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run DeepD3 models on synthetic data "
            "using whole-image inference and save raw, "
            "uncleaned probability maps."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Root directory of the synthetic dataset.",
    )

    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
        help="Dataset split to process.",
    )

    parser.add_argument(
        "--model-32f",
        type=Path,
        default=DEFAULT_MODEL_32F,
        help="Path to original DeepD3 32F model.",
    )

    parser.add_argument(
        "--model-32f-94nm",
        type=Path,
        default=DEFAULT_MODEL_32F_94NM,
        help="Path to original DeepD3 32F 94 nm model.",
    )

    parser.add_argument(
        "--model-synthetic-94nm",
        type=Path,
        default=None,
        help=(
            "Optional path to synthetic-trained "
            "DeepD3 32F 94 nm model."
        ),
    )

    parser.add_argument(
        "--image-name",
        default="noisy.tif",
        help="Input image filename inside each instance.",
    )

    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Optional maximum number of instances to process.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing prediction files.",
    )

    return parser.parse_args()


# ==========================================================
# Helpers
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

    # ------------------------------------------------------
    # Skip existing output unless overwrite was requested
    # ------------------------------------------------------

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
    # Load image stack
    # ------------------------------------------------------

    stack = Stack(
        str(image_path)
    )

    # ------------------------------------------------------
    # DeepD3 whole-image inference
    #
    # Important:
    # - No predictInset()
    # - No cleaning
    # - No clipping
    # - No thresholding
    # ------------------------------------------------------

    stack.predictWholeImage(
        str(model_path)
    )

    # ------------------------------------------------------
    # Get RAW prediction
    # ------------------------------------------------------

    prediction = np.asarray(
        stack.prediction,
        dtype=np.float32,
    )

    print(
        f"      Prediction shape          : "
        f"{prediction.shape}"
    )

    # Expected:
    # Z x Y x X x channels

    if prediction.ndim != 4:
        raise ValueError(
            "Unexpected DeepD3 prediction shape: "
            f"{prediction.shape}. "
            "Expected a 4D array "
            "(Z, Y, X, channels)."
        )

    if prediction.shape[-1] < 2:
        raise ValueError(
            "Expected at least two prediction channels, "
            f"got shape {prediction.shape}"
        )

    # ------------------------------------------------------
    # Extract RAW probability channels
    #
    # DeepD3:
    # channel 0 = dendrites
    # channel 1 = spines
    # ------------------------------------------------------

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

    dendrite_nan_count = int(
        np.isnan(dendrites).sum()
    )

    spine_nan_count = int(
        np.isnan(spines).sum()
    )

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
        f"      Dendrite NaN count        : "
        f"{dendrite_nan_count}"
    )

    print(
        f"      Spine NaN count           : "
        f"{spine_nan_count}"
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
    #
    # Important:
    # - no np.clip()
    # - no thresholding
    # - no cleanSpines()
    # - no cleanDendrite()
    # ------------------------------------------------------

    fl.save(
        str(final_output),
        {
            "dendrites": dendrites,
            "spines": spines,
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

    # ------------------------------------------------------
    # Validate dataset
    # ------------------------------------------------------

    if not dataset_root.is_dir():
        raise FileNotFoundError(
            "Dataset directory not found:\n"
            f"  {dataset_root}"
        )

    if not split_dir.is_dir():
        raise FileNotFoundError(
            "Dataset split not found:\n"
            f"  {split_dir}"
        )

    # ------------------------------------------------------
    # Original pretrained models
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # Optional synthetic-trained model
    #
    # This gets its own model tag so that it DOES NOT
    # overwrite the original 32F_94nm predictions.
    # ------------------------------------------------------

    if args.model_synthetic_94nm is not None:

        model_synthetic_94nm = (
            args.model_synthetic_94nm.resolve()
        )

        require_file(
            model_synthetic_94nm,
            (
                "Synthetic-trained DeepD3 "
                "32F 94 nm model"
            ),
        )

        models.append(
            (
                "synthetic_32F_94nm",
                model_synthetic_94nm,
            )
        )

    # ------------------------------------------------------
    # Find dataset instances
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "DeepD3 RAW probability inference"
    )

    print("=" * 70)

    print(
        f"Dataset          : {dataset_root}"
    )

    print(
        f"Split            : {args.split}"
    )

    print(
        f"Instances        : {len(instance_dirs)}"
    )

    print(
        f"Input image      : {args.image_name}"
    )

    print(
        "Inference method : predictWholeImage"
    )

    print(
        "Cleaning         : disabled"
    )

    print(
        "Clipping         : disabled"
    )

    print(
        f"Models requested : "
        f"{', '.join(tag for tag, _ in models)}"
    )

    print("=" * 70)

    # ------------------------------------------------------
    # Process instances
    # ------------------------------------------------------

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
                overwrite=args.overwrite,
            )

        completed += 1

    # ------------------------------------------------------
    # Final summary
    # ------------------------------------------------------

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
        f"Models considered   : {len(models)}"
    )

    print(
        "Existing predictions may have been skipped "
        "when --overwrite was not used."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()