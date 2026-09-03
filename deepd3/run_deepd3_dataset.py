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
import shutil
import subprocess
import sys
from pathlib import Path


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
# Command-line arguments
# ==========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run pretrained DeepD3 models on a synthetic "
            "microscopy dataset split."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=(
            "Dataset root containing train/validation/test folders. "
            f"Default: {DEFAULT_DATASET}"
        ),
    )

    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
        help="Dataset split to process. Default: test.",
    )

    parser.add_argument(
        "--model-32f",
        type=Path,
        default=DEFAULT_MODEL_32F,
        help="Path to DeepD3_32F.h5.",
    )

    parser.add_argument(
        "--model-32f-94nm",
        type=Path,
        default=DEFAULT_MODEL_32F_94NM,
        help="Path to DeepD3_32F_94nm.h5.",
    )

    parser.add_argument(
        "--image-name",
        default="noisy.tif",
        help=(
            "Input image filename inside each instance. "
            "Default: noisy.tif."
        ),
    )

    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help=(
            "Optional maximum number of instances to process. "
            "Useful for testing."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing DeepD3 prediction files.",
    )

    return parser.parse_args()


# ==========================================================
# File helpers
# ==========================================================

def require_file(path: Path, description: str):
    """
    Check that a required file exists.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} not found:\n  {path}"
        )


def find_input_image(
    instance_dir: Path,
    image_name: str,
) -> Path:
    """
    Return the requested microscopy image from a dataset instance.

    The exact filename supplied through --image-name is required.
    """

    image_path = instance_dir / image_name

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Input image not found:\n"
            f"  {image_path}"
        )

    return image_path


def prediction_candidates(
    image_path: Path,
) -> list[Path]:
    """
    Return possible output filenames created by DeepD3.

    Depending on the DeepD3 version, batch inference may create:

        image.tif.prediction

    or:

        image.prediction
    """

    return [
        Path(str(image_path) + ".prediction"),
        image_path.with_suffix(".prediction"),
    ]


# ==========================================================
# DeepD3 inference
# ==========================================================

def run_model(
    image_path: Path,
    model_path: Path,
    model_tag: str,
    output_dir: Path,
    overwrite: bool,
):
    """
    Run one DeepD3 model on one microscopy image.

    The resulting .prediction file is moved into the
    deepd3_predictions folder of the dataset instance.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_output = (
        output_dir
        / f"{model_tag}.prediction"
    )

    if final_output.exists() and not overwrite:
        print(
            f"    {model_tag}: already exists, skipping"
        )
        return final_output

    # Remove temporary prediction files from an interrupted run.
    for candidate in prediction_candidates(image_path):
        if candidate.exists():
            candidate.unlink()

    command = [
        sys.executable,
        "-m",
        "deepd3.inference.batch",
        str(image_path),
        str(model_path),
    ]

    print(f"    Running model: {model_tag}")
    print(f"      Model : {model_path}")
    print(f"      Image : {image_path}")

    result = subprocess.run(
        command,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"DeepD3 inference failed.\n"
            f"Instance: {image_path.parent.name}\n"
            f"Model: {model_tag}\n"
            f"Exit code: {result.returncode}"
        )

    produced_prediction = None

    for candidate in prediction_candidates(image_path):
        if candidate.exists():
            produced_prediction = candidate
            break

    if produced_prediction is None:
        checked = "\n".join(
            f"  {path}"
            for path in prediction_candidates(image_path)
        )

        raise FileNotFoundError(
            "DeepD3 finished without creating a "
            ".prediction file.\n\n"
            f"Checked:\n{checked}"
        )

    if final_output.exists():
        final_output.unlink()

    shutil.move(
        str(produced_prediction),
        str(final_output),
    )

    print(
        f"      Saved : {final_output}"
    )

    return final_output


# ==========================================================
# Main
# ==========================================================

def main():
    args = parse_args()

    # ------------------------------------------------------
    # Validate dataset
    # ------------------------------------------------------

    dataset_root = args.dataset.resolve()
    split_dir = dataset_root / args.split

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

    # ------------------------------------------------------
    # Validate models
    # ------------------------------------------------------

    model_32f = args.model_32f.resolve()
    model_32f_94nm = args.model_32f_94nm.resolve()

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
    # Find dataset instances
    # ------------------------------------------------------

    instance_dirs = sorted(
        path
        for path in split_dir.glob("instance_*")
        if path.is_dir()
    )

    if not instance_dirs:
        raise RuntimeError(
            f"No instance_* directories found in:\n"
            f"  {split_dir}"
        )

    if args.max_instances is not None:
        if args.max_instances < 1:
            raise ValueError(
                "--max-instances must be at least 1."
            )

        instance_dirs = (
            instance_dirs[
                : args.max_instances
            ]
        )

    # ------------------------------------------------------
    # Print configuration
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("DeepD3 dataset inference")
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
        f"Model 32F   : {model_32f}"
    )

    print(
        f"Model 94 nm : {model_32f_94nm}"
    )

    print(
        f"Python      : {sys.executable}"
    )

    print("=" * 70)

    # ------------------------------------------------------
    # Run inference
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

        image_path = find_input_image(
            instance_dir,
            args.image_name,
        )

        print(
            f"    Input: {image_path.name}"
        )

        prediction_dir = (
            instance_dir
            / "deepd3_predictions"
        )

        for model_tag, model_path in models:
            run_model(
                image_path=image_path,
                model_path=model_path,
                model_tag=model_tag,
                output_dir=prediction_dir,
                overwrite=args.overwrite,
            )

        completed += 1

    # ------------------------------------------------------
    # Finished
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("DeepD3 inference completed")
    print("=" * 70)

    print(
        f"Processed instances : {completed}"
    )

    print(
        f"Split               : {args.split}"
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