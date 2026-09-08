from __future__ import annotations

import argparse
import os
from pathlib import Path

import flammkuchen as fl
import numpy as np
import pandas as pd
import tifffile


# ============================================================
# Synthetic dataset -> DeepD3 .d3set
# ============================================================

def load_instance(instance_dir: Path):
    image_path = instance_dir / "noisy.tif"
    dendrite_path = instance_dir / "dendrite_mask.tif"
    spine_path = instance_dir / "spine_mask.tif"

    required = [image_path, dendrite_path, spine_path]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

    image = tifffile.imread(image_path)
    dendrite = tifffile.imread(dendrite_path) > 0
    spine = tifffile.imread(spine_path) > 0

    if image.ndim != 3:
        raise ValueError(
            f"{instance_dir.name}: expected ZYX volume, "
            f"got shape {image.shape}"
        )

    if not (
        image.shape == dendrite.shape == spine.shape
    ):
        raise ValueError(
            f"{instance_dir.name}: shape mismatch\n"
            f"  image:    {image.shape}\n"
            f"  dendrite: {dendrite.shape}\n"
            f"  spine:    {spine.shape}"
        )

    return image, dendrite, spine


def prepare_split(
    dataset_dir: Path,
    split: str,
    output_file: Path,
    resolution_xy: float = 0.094,
    resolution_z: float = 0.5,
):
    split_dir = dataset_dir / split

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Split not found: {split_dir}"
        )

    instances = sorted(
        p
        for p in split_dir.glob("instance_*")
        if p.is_dir()
    )

    if not instances:
        raise RuntimeError(
            f"No instance directories found in {split_dir}"
        )

    print()
    print("=" * 70)
    print(f"Preparing DeepD3 {split} dataset")
    print("=" * 70)
    print(f"Source:       {split_dir}")
    print(f"Instances:    {len(instances)}")
    print(f"XY spacing:   {resolution_xy} µm")
    print(f"Z spacing:    {resolution_z} µm")
    print(f"Output:       {output_file}")
    print("=" * 70)

    stacks = {}
    dendrites = {}
    spines = {}
    metadata = []

    for i, instance_dir in enumerate(instances):
        image, dendrite, spine = load_instance(instance_dir)

        key = f"x{i}"

        stacks[key] = image
        dendrites[key] = dendrite
        spines[key] = spine

        depth, height, width = image.shape

        metadata.append(
            {
                "crop": False,
                "X": 0,
                "Y": 0,
                "Width": width,
                "Height": height,
                "Depth": depth,
                "Z_begin": 0,
                "Z_end": depth - 1,
                "Resolution_XY": float(resolution_xy),
                "Resolution_Z": float(resolution_z),
                "Generated_from": str(instance_dir),
            }
        )

        print(
            f"[{i + 1:03d}/{len(instances):03d}] "
            f"{instance_dir.name} | "
            f"shape={image.shape} | "
            f"dendrite={int(dendrite.sum())} px | "
            f"spine={int(spine.sum())} px"
        )

    dataset = {
        "data": {
            "stacks": stacks,
            "dendrites": dendrites,
            "spines": spines,
        },
        "meta": pd.DataFrame(metadata),
    }

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fl.save(
        str(output_file),
        dataset,
        compression="blosc",
    )

    print()
    print(f"Saved: {output_file}")
    print()


# ============================================================
# Sanity check
# ============================================================

def check_dataset(
    dataset_file: Path,
    batch_size: int = 4,
):
    from deepd3.training.stream import DataGeneratorStream

    print()
    print("=" * 70)
    print("Checking DeepD3 dataset")
    print("=" * 70)
    print(dataset_file)

    generator = DataGeneratorStream(
        str(dataset_file),
        batch_size=batch_size,
        samples_per_epoch=16,
        size=(1, 128, 128),
        target_resolution=0.094,
        min_content=50,
        augment=False,
        shuffle=False,
    )

    X, Y = generator[0]

    Y_dendrite = Y[0]
    Y_spine = Y[1]

    print()
    print("Batch shapes")
    print(f"X:          {X.shape}")
    print(f"Dendrite:   {Y_dendrite.shape}")
    print(f"Spine:      {Y_spine.shape}")

    print()
    print("Value ranges")
    print(
        f"X:        "
        f"{float(X.min()):.4f} -> "
        f"{float(X.max()):.4f}"
    )

    print(
        f"Dendrite: "
        f"{float(Y_dendrite.min()):.4f} -> "
        f"{float(Y_dendrite.max()):.4f}"
    )

    print(
        f"Spine:    "
        f"{float(Y_spine.min()):.4f} -> "
        f"{float(Y_spine.max()):.4f}"
    )

    print()
    print("Sanity check completed.")
    print()


# ============================================================
# DeepD3 training
# ============================================================

def train(
    train_file: Path,
    validation_file: Path,
    output_dir: Path,
    batch_size: int = 32,
    epochs: int = 30,
    samples_per_epoch: int = 50000,
    validation_samples: int = 1280,
):
    os.environ["SM_FRAMEWORK"] = "tf.keras"

    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        CSVLogger,
        LearningRateScheduler,
        ModelCheckpoint,
    )
    from tensorflow.keras.optimizers import Adam

    import segmentation_models as sm

    sm.set_framework("tf.keras")

    from deepd3.model import DeepD3_Model
    from deepd3.training.stream import DataGeneratorStream

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        output_dir
        / "synthetic_32F_94nm_best.h5"
    )

    log_path = (
        output_dir
        / "synthetic_32F_94nm_training.csv"
    )

    print()
    print("=" * 70)
    print("DeepD3 synthetic training")
    print("=" * 70)
    print(f"Training data:       {train_file}")
    print(f"Validation data:     {validation_file}")
    print(f"Batch size:          {batch_size}")
    print(f"Training samples:    {samples_per_epoch}")
    print(f"Validation samples:  {validation_samples}")
    print(f"Epochs:              {epochs}")
    print(f"Resolution:          0.094 µm")
    print(f"Model output:        {model_path}")
    print(f"Training log:        {log_path}")
    print("=" * 70)

    # --------------------------------------------------------
    # Generators
    # --------------------------------------------------------

    dg_training = DataGeneratorStream(
        str(train_file),
        batch_size=batch_size,
        samples_per_epoch=samples_per_epoch,
        size=(1, 128, 128),
        target_resolution=0.094,
        min_content=50,
        augment=True,
        shuffle=True,
    )

    dg_validation = DataGeneratorStream(
        str(validation_file),
        batch_size=batch_size,
        samples_per_epoch=validation_samples,
        size=(1, 128, 128),
        target_resolution=0.094,
        min_content=50,
        augment=False,
        shuffle=False,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = DeepD3_Model(
        filters=32
    )

    model.compile(
        optimizer=Adam(
            learning_rate=0.0005
        ),
        loss=[
            sm.losses.dice_loss,
            "mse",
        ],
        metrics=[
            "acc",
            sm.metrics.iou_score,
        ],
    )

    model.summary()

    # --------------------------------------------------------
    # Learning-rate schedule
    # --------------------------------------------------------

    def schedule(epoch, lr):
        if epoch < 15:
            return lr

        return lr * tf.math.exp(-0.1)

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    checkpoint = ModelCheckpoint(
        str(model_path),
        save_best_only=True,
    )

    csv_logger = CSVLogger(
        str(log_path)
    )

    lr_scheduler = LearningRateScheduler(
        schedule
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        dg_training,
        epochs=epochs,
        validation_data=dg_validation,
        callbacks=[
            checkpoint,
            csv_logger,
            lr_scheduler,
        ],
    )

    print()
    print("=" * 70)
    print("Training completed")
    print("=" * 70)
    print(f"Best model: {model_path}")
    print(f"Log:        {log_path}")
    print()


# ============================================================
# Command line
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, verify and train DeepD3 "
            "using the synthetic microscopy dataset."
        )
    )

    parser.add_argument(
        "command",
        choices=[
            "prepare",
            "check",
            "train",
        ],
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "outputs/synthetic_dataset_v1"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=50000,
    )

    parser.add_argument(
        "--validation-samples",
        type=int,
        default=1280,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "deepd3/models/synthetic_94nm"
        ),
    )

    args = parser.parse_args()

    training_dir = (
        args.dataset
        / "deepd3_training"
    )

    train_file = (
        training_dir
        / "synthetic_train.d3set"
    )

    validation_file = (
        training_dir
        / "synthetic_validation.d3set"
    )

    if args.command == "prepare":

        prepare_split(
            dataset_dir=args.dataset,
            split="train",
            output_file=train_file,
        )

        prepare_split(
            dataset_dir=args.dataset,
            split="validation",
            output_file=validation_file,
        )

    elif args.command == "check":

        check_dataset(
            train_file,
            batch_size=4,
        )

        check_dataset(
            validation_file,
            batch_size=4,
        )

    elif args.command == "train":

        if not train_file.exists():
            raise FileNotFoundError(
                f"{train_file} does not exist. "
                "Run 'prepare' first."
            )

        if not validation_file.exists():
            raise FileNotFoundError(
                f"{validation_file} does not exist. "
                "Run 'prepare' first."
            )

        train(
            train_file=train_file,
            validation_file=validation_file,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            samples_per_epoch=args.samples_per_epoch,
            validation_samples=args.validation_samples,
        )


if __name__ == "__main__":
    main()