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

Each measured repetition runs in an independent fresh Python process.
No separate warm-up run is used because a warm-up in a different subprocess
would not warm the CUDA context of the measured process.

Noise is disabled because this benchmark measures renderer performance.
Run from the repository root:
    PYTHONPATH=. python benchmarks/benchmark_renderers.py
"""

import argparse
import csv
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RENDER_SCRIPT = ROOT / "scripts" / "render.py"

VOXEL_CONFIG = ROOT / "configs" / "test_voxelgrid.yaml"
GAUSSIAN_CONFIG = ROOT / "configs" / "test_gaussian_splatting.yaml"

BENCHMARK_SIZES = {
    "Small": [64, 128, 128],
    "Medium": [128, 256, 256],
}

RENDERERS = {
    "Voxel": VOXEL_CONFIG,
    "Gaussian splatting": GAUSSIAN_CONFIG,
}


def collect_system_metadata():
    """Collect reproducibility metadata for the benchmark."""
    metadata = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }

    try:
        import torch

        metadata["torch_version"] = torch.__version__
        metadata["cuda_version"] = torch.version.cuda
        metadata["cuda_available"] = torch.cuda.is_available()

        if torch.cuda.is_available():
            metadata["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            metadata["gpu_memory_mb"] = round(
                props.total_memory / (1024 ** 2), 2
            )
        else:
            metadata["gpu_name"] = None
            metadata["gpu_memory_mb"] = None

    except Exception as exc:
        metadata["torch_version"] = None
        metadata["cuda_version"] = None
        metadata["cuda_available"] = None
        metadata["gpu_name"] = None
        metadata["gpu_memory_mb"] = None
        metadata["metadata_warning"] = str(exc)

    return metadata


TIME_PATTERN = re.compile(r"total renderer time:\s*([0-9.]+)s")
ALLOCATED_PATTERN = re.compile(
    r"peak allocated GPU memory:\s*([0-9.]+)\s*MB"
)
RESERVED_PATTERN = re.compile(
    r"peak reserved GPU memory\s*:\s*([0-9.]+)\s*MB"
)


def extract_metrics(output):
    """Extract renderer timing and GPU memory from renderer stdout."""
    time_match = TIME_PATTERN.search(output)

    if not time_match:
        raise RuntimeError(
            "Could not find 'total renderer time' in renderer output."
        )

    total_time = float(time_match.group(1))

    allocated_match = ALLOCATED_PATTERN.search(output)
    reserved_match = RESERVED_PATTERN.search(output)

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

    return total_time, allocated, reserved


def create_benchmark_config(
    base_config_path,
    shape_zyx,
    temp_directory,
    renderer_name,
    size_name,
):
    """Create a temporary config with only benchmark-specific overrides."""
    with open(base_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["grid"]["shape_mode"] = "fixed"
    config["grid"]["output_shape_zyx"] = list(shape_zyx)

    if "noise" not in config:
        config["noise"] = {}
    config["noise"]["enabled"] = False

    safe_renderer = renderer_name.lower().replace(" ", "_")

    config["output"]["output_dir"] = str(
        ROOT
        / "outputs"
        / "benchmark_renders"
        / safe_renderer
        / size_name.lower()
    )

    temp_config = (
        Path(temp_directory)
        / f"{safe_renderer}_{size_name.lower()}.yaml"
    )

    with open(temp_config, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return temp_config


def run_benchmark(
    renderer,
    config_path,
    device,
    shape_zyx,
    repetition,
):
    """Execute one independent benchmark repetition."""
    env = os.environ.copy()

    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        env["PYTHONPATH"] = (
            str(ROOT) + os.pathsep + existing_pythonpath
        )
    else:
        env["PYTHONPATH"] = str(ROOT)

    if device == "CPU":
        env["CUDA_VISIBLE_DEVICES"] = ""

    command = [
        sys.executable,
        str(RENDER_SCRIPT),
        "--config",
        str(config_path),
    ]

    print()
    print("=" * 70)
    print(f"Renderer : {renderer}")
    print(f"Device   : Torch {device}")
    print(f"Shape    : {shape_zyx}")
    print(f"Run      : repetition {repetition}")
    print("=" * 70)

    wall_start = time.perf_counter()

    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    wall_time = time.perf_counter() - wall_start

    print(result.stdout)

    if result.returncode != 0:
        raise RuntimeError(
            f"Renderer failed with exit code {result.returncode}"
        )

    total_time, allocated, reserved = extract_metrics(result.stdout)

    return {
        "total_time_s": total_time,
        "wall_time_s": wall_time,
        "time_per_slice_s": total_time / shape_zyx[0],
        "peak_allocated_mb": allocated,
        "peak_reserved_mb": reserved,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark synthetic microscopy renderers."
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
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Number of independent measured repetitions per configuration.",
    )

    args = parser.parse_args()

    if args.repetitions < 1:
        raise ValueError("--repetitions must be at least 1.")

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = collect_system_metadata()
    metadata["repetitions"] = args.repetitions
    metadata["process_model"] = (
        "independent fresh Python process per measured repetition"
    )
    metadata["warmup_runs"] = 0
    metadata["benchmark_sizes"] = {
        name: BENCHMARK_SIZES[name] for name in args.sizes
    }
    metadata["renderers"] = args.renderers
    metadata["devices"] = args.devices
    metadata["noise_enabled"] = False

    metadata_path = output_path.with_name(
        output_path.stem + "_metadata.yaml"
    )

    with open(metadata_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)

    results = []

    with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
        for size_name in args.sizes:
            shape_zyx = BENCHMARK_SIZES[size_name]

            for renderer_name in args.renderers:
                base_config = RENDERERS[renderer_name]

                benchmark_config = create_benchmark_config(
                    base_config,
                    shape_zyx,
                    temp_dir,
                    renderer_name,
                    size_name,
                )

                for device in args.devices:
                    for repetition in range(1, args.repetitions + 1):
                        metrics = run_benchmark(
                            renderer_name,
                            benchmark_config,
                            device,
                            shape_zyx,
                            repetition,
                        )

                        result = {
                            "size": size_name,
                            "shape_zyx": "x".join(map(str, shape_zyx)),
                            "z_slices": shape_zyx[0],
                            "renderer": renderer_name,
                            "device": device,
                            "repetition": repetition,
                            "total_time_s": metrics["total_time_s"],
                            "wall_time_s": metrics["wall_time_s"],
                            "time_per_slice_s": metrics["time_per_slice_s"],
                            "peak_allocated_mb": metrics["peak_allocated_mb"],
                            "peak_reserved_mb": metrics["peak_reserved_mb"],
                        }

                        results.append(result)
                        print()
                        print("RESULT:", result)

    columns = [
        "size",
        "shape_zyx",
        "z_slices",
        "renderer",
        "device",
        "repetition",
        "total_time_s",
        "wall_time_s",
        "time_per_slice_s",
        "peak_allocated_mb",
        "peak_reserved_mb",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)

    print()
    print("=" * 70)
    print("Benchmark complete.")
    print(f"Results saved to:\n{output_path}")
    print(f"Metadata saved to:\n{metadata_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
