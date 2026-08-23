#!/usr/bin/env python3

"""
Benchmark voxel-grid and Gaussian-splatting renderers.

Runs:
    - Voxel / Torch CPU
    - Voxel / Torch GPU
    - Gaussian splatting / Torch CPU
    - Gaussian splatting / Torch GPU

For:
    Small  = [64, 128, 128] ZYX
    Medium = [128, 256, 256] ZYX
    Large  = [256, 512, 512] ZYX

The script:
    1. Creates temporary benchmark YAML configurations.
    2. Calls the existing scripts/render.py.
    3. Extracts renderer time and GPU memory from stdout.
    4. Saves all measurements to timings.csv.

Noise is disabled because this benchmark measures renderer performance.
Run from the repository root: PYTHONPATH=. python benchmarks/benchmark_renderers.py
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


# ============================================================
# Repository paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RENDER_SCRIPT = ROOT / "scripts" / "render.py"

VOXEL_CONFIG = ROOT / "configs" / "test_voxelgrid.yaml"

GAUSSIAN_CONFIG = (
    ROOT / "configs" / "test_gaussian_splatting.yaml"
)


# ============================================================
# Benchmark sizes
# ============================================================

BENCHMARK_SIZES = {
    "Small": [64, 128, 128],
    "Medium": [128, 256, 256],
    "Large": [256, 512, 512],
}


# ============================================================
# Renderer configurations
# ============================================================

RENDERERS = {
    "Voxel": VOXEL_CONFIG,
    "Gaussian splatting": GAUSSIAN_CONFIG,
}


# ============================================================
# Parse renderer output
# ============================================================

TIME_PATTERN = re.compile(
    r"total renderer time:\s*([0-9.]+)s"
)

ALLOCATED_PATTERN = re.compile(
    r"peak allocated GPU memory:\s*([0-9.]+)\s*MB"
)

RESERVED_PATTERN = re.compile(
    r"peak reserved GPU memory\s*:\s*([0-9.]+)\s*MB"
)


def extract_metrics(output):
    """Extract timing and GPU memory from renderer stdout."""

    time_match = TIME_PATTERN.search(output)

    if not time_match:
        raise RuntimeError(
            "Could not find 'total renderer time' "
            "in renderer output."
        )

    total_time = float(
        time_match.group(1)
    )

    allocated_match = (
        ALLOCATED_PATTERN.search(output)
    )

    reserved_match = (
        RESERVED_PATTERN.search(output)
    )

    allocated = (
        float(allocated_match.group(1))
        if allocated_match
        else None
    )

    reserved = (
        float(reserved_match.group(1))
        if reserved_match
        else None
    )

    return (
        total_time,
        allocated,
        reserved,
    )


# ============================================================
# Create benchmark configuration
# ============================================================

def create_benchmark_config(
    base_config_path,
    shape_zyx,
    temp_directory,
    renderer_name,
    size_name,
):
    """
    Copy an existing renderer config and modify only
    benchmark-specific settings.
    """

    with open(
        base_config_path,
        "r",
        encoding="utf-8",
    ) as f:

        config = yaml.safe_load(f)


    # --------------------------------------------------------
    # Fixed benchmark volume
    # --------------------------------------------------------

    config["grid"]["shape_mode"] = "fixed"

    config["grid"]["output_shape_zyx"] = (
        list(shape_zyx)
    )


    # --------------------------------------------------------
    # Disable noise
    # --------------------------------------------------------

    if "noise" not in config:
        config["noise"] = {}

    config["noise"]["enabled"] = False


    # --------------------------------------------------------
    # Separate benchmark output directory
    # --------------------------------------------------------

    safe_renderer = (
        renderer_name
        .lower()
        .replace(" ", "_")
    )

    config["output"]["output_dir"] = str(
        ROOT
        / "outputs"
        / "benchmark_renders"
        / safe_renderer
        / size_name.lower()
    )


    # --------------------------------------------------------
    # Temporary YAML
    # --------------------------------------------------------

    temp_config = (
        Path(temp_directory)
        / f"{safe_renderer}_{size_name.lower()}.yaml"
    )

    with open(
        temp_config,
        "w",
        encoding="utf-8",
    ) as f:

        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
        )

    return temp_config


# ============================================================
# Run one benchmark
# ============================================================

def run_benchmark(
    renderer,
    config_path,
    device,
    shape_zyx,
):
    """
    Execute one renderer benchmark.
    """

    env = os.environ.copy()

    # Make repository importable.
    existing_pythonpath = env.get(
        "PYTHONPATH",
        "",
    )

    if existing_pythonpath:
        env["PYTHONPATH"] = (
            str(ROOT)
            + os.pathsep
            + existing_pythonpath
        )
    else:
        env["PYTHONPATH"] = str(ROOT)


    # --------------------------------------------------------
    # CPU / GPU selection
    # --------------------------------------------------------

    if device == "CPU":

        # Hides CUDA from PyTorch.
        env["CUDA_VISIBLE_DEVICES"] = ""

    else:

        # Keep Slurm's CUDA_VISIBLE_DEVICES setting.
        pass


    command = [
        sys.executable,
        str(RENDER_SCRIPT),
        "--config",
        str(config_path),
    ]


    print()
    print("=" * 70)

    print(
        f"Renderer : {renderer}"
    )

    print(
        f"Device   : Torch {device}"
    )

    print(
        f"Shape    : {shape_zyx}"
    )

    print("=" * 70)


    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


    # Print renderer output to terminal.
    print(result.stdout)


    if result.returncode != 0:

        raise RuntimeError(
            f"Renderer failed with exit code "
            f"{result.returncode}"
        )


    total_time, allocated, reserved = (
        extract_metrics(
            result.stdout
        )
    )


    z_slices = shape_zyx[0]

    time_per_slice = (
        total_time / z_slices
    )


    return {
        "total_time_s": total_time,
        "time_per_slice_s": time_per_slice,
        "peak_allocated_mb": allocated,
        "peak_reserved_mb": reserved,
    }


# ============================================================
# Main benchmark
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark synthetic microscopy renderers."
        )
    )


    parser.add_argument(
        "--output",
        default="outputs/benchmarks/timings.csv",
        help="Output CSV file.",
    )


    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=list(BENCHMARK_SIZES.keys()),
        default=list(BENCHMARK_SIZES.keys()),
        help="Benchmark sizes to run.",
    )


    parser.add_argument(
        "--devices",
        nargs="+",
        choices=["CPU", "GPU"],
        default=["CPU", "GPU"],
        help="Devices to benchmark.",
    )


    parser.add_argument(
        "--renderers",
        nargs="+",
        choices=list(RENDERERS.keys()),
        default=list(RENDERERS.keys()),
        help="Renderers to benchmark.",
    )


    args = parser.parse_args()


    output_path = ROOT / args.output

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    results = []


    # --------------------------------------------------------
    # Temporary benchmark configs
    # --------------------------------------------------------

    with tempfile.TemporaryDirectory(
        dir=ROOT
    ) as temp_dir:


        for size_name in args.sizes:

            shape_zyx = (
                BENCHMARK_SIZES[size_name]
            )


            for renderer_name in args.renderers:

                base_config = (
                    RENDERERS[renderer_name]
                )


                benchmark_config = (
                    create_benchmark_config(
                        base_config,
                        shape_zyx,
                        temp_dir,
                        renderer_name,
                        size_name,
                    )
                )


                for device in args.devices:

                    metrics = run_benchmark(
                        renderer_name,
                        benchmark_config,
                        device,
                        shape_zyx,
                    )


                    result = {

                        "size":
                            size_name,

                        "shape_zyx":
                            "x".join(
                                map(
                                    str,
                                    shape_zyx,
                                )
                            ),

                        "z_slices":
                            shape_zyx[0],

                        "renderer":
                            renderer_name,

                        "device":
                            device,

                        "total_time_s":
                            metrics[
                                "total_time_s"
                            ],

                        "time_per_slice_s":
                            metrics[
                                "time_per_slice_s"
                            ],

                        "peak_allocated_mb":
                            metrics[
                                "peak_allocated_mb"
                            ],

                        "peak_reserved_mb":
                            metrics[
                                "peak_reserved_mb"
                            ],
                    }


                    results.append(
                        result
                    )


                    print()
                    print(
                        "RESULT:",
                        result,
                    )


    # ========================================================
    # Save CSV
    # ========================================================

    columns = [

        "size",

        "shape_zyx",

        "z_slices",

        "renderer",

        "device",

        "total_time_s",

        "time_per_slice_s",

        "peak_allocated_mb",

        "peak_reserved_mb",
    ]


    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=columns,
        )

        writer.writeheader()

        writer.writerows(
            results
        )


    print()
    print("=" * 70)

    print(
        f"Benchmark complete."
    )

    print(
        f"Results saved to:"
    )

    print(
        output_path
    )

    print("=" * 70)


if __name__ == "__main__":
    main()