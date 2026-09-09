from pathlib import Path
import argparse
import numpy as np
import tifffile
import flammkuchen as fl

from deepd3.core.analysis import Stack


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DEFAULT_BENCHMARK = Path(
    "benchmarks/deepd3/DeepD3_Benchmark.tif"
)

DEFAULT_REAL_MODEL = Path(
    "deepd3/models/DeepD3_32F_94nm.h5"
)

DEFAULT_SYNTHETIC_MODEL = Path(
    "deepd3/models/synthetic_94nm/synthetic_32F_94nm_best.h5"
)

DEFAULT_OUTPUT_DIR = Path(
    "benchmarks/deepd3/predictions"
)


# ---------------------------------------------------------
# Inference
# ---------------------------------------------------------

def run_model(
    image_path: Path,
    model_path: Path,
    output_path: Path,
    resolution_xy=0.094,
    resolution_z=0.5,
    overwrite=False,
):

    if output_path.exists() and not overwrite:
        print(f"\nSkipping existing prediction:")
        print(output_path)
        return

    if not image_path.exists():
        raise FileNotFoundError(
            f"Benchmark image not found: {image_path}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    print("\n========================================")
    print("DeepD3 benchmark inference")
    print("========================================")
    print("Image :", image_path)
    print("Model :", model_path)
    print("Output:", output_path)

    # Check raw TIFF
    image = tifffile.imread(image_path)

    print("\nInput stack")
    print("Shape :", image.shape)
    print("dtype :", image.dtype)
    print("range :", image.min(), "->", image.max())

    expected_shape = image.shape

    # -----------------------------------------------------
    # Load through DeepD3 Stack
    # -----------------------------------------------------

    stack = Stack(
        str(image_path)
    )

    print("\nDeepD3 stack shape:", stack.stack.shape)

    # -----------------------------------------------------
    # Official whole-image inference
    # -----------------------------------------------------

    print("\nRunning predictWholeImage() ...")

    success = stack.predictWholeImage(
        str(model_path)
    )

    if not success:
        raise RuntimeError(
            "DeepD3 predictWholeImage() failed."
        )

    prediction = stack.prediction

    print("\nPrediction")
    print("Shape :", prediction.shape)
    print("dtype :", prediction.dtype)

    if prediction.shape[:3] != expected_shape:
        raise RuntimeError(
            f"Prediction shape {prediction.shape[:3]} "
            f"does not match benchmark shape {expected_shape}"
        )

    # DeepD3 prediction channels:
    # 0 = dendrite
    # 1 = spine
    # 2 = duplicate dendrite
    dendrite = prediction[..., 0]
    spine = prediction[..., 1]

    print("\nDendrite probability")
    print(
        "range:",
        float(dendrite.min()),
        "->",
        float(dendrite.max())
    )

    print("\nSpine probability")
    print(
        "range:",
        float(spine.min()),
        "->",
        float(spine.max())
    )

    # Important:
    # NO cleaning
    # NO clipping
    # NO thresholding

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save exactly as raw DeepD3 prediction
    fl.save(
        output_path,
        {
            "prediction": prediction.astype(np.float32),
            "dendrites": dendrite.astype(np.float32),
            "spines": spine.astype(np.float32),
            "image_shape": np.array(expected_shape),
            "resolution_xy_um": resolution_xy,
            "resolution_z_um": resolution_z,
            "model": str(model_path),
        },
    )

    print("\nSaved:")
    print(output_path)

    print("\nDone.")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run DeepD3 whole-image inference on the "
            "official DeepD3 benchmark."
        )
    )

    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK,
    )

    parser.add_argument(
        "--model",
        choices=["real", "synthetic", "both"],
        default="both",
    )

    parser.add_argument(
        "--real-model",
        type=Path,
        default=DEFAULT_REAL_MODEL,
    )

    parser.add_argument(
        "--synthetic-model",
        type=Path,
        default=DEFAULT_SYNTHETIC_MODEL,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    if args.model in ["real", "both"]:

        run_model(
            image_path=args.benchmark,
            model_path=args.real_model,
            output_path=(
                args.output_dir /
                "real_32F_94nm.prediction"
            ),
            overwrite=args.overwrite,
        )

    if args.model in ["synthetic", "both"]:

        run_model(
            image_path=args.benchmark,
            model_path=args.synthetic_model,
            output_path=(
                args.output_dir /
                "synthetic_32F_94nm.prediction"
            ),
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()