#!/usr/bin/env python3
"""
Create a thesis-ready overview of manually annotated neuronal meshes.

For each YAML config, the script reads the same labelled-component input used by
the renderer:

    input.mode
    input.labelled_dir
    input.dendrite_pattern
    input.spine_pattern
    input.scale_to_nm

The dendritic shaft is rendered in gray and each individual spine is assigned a
different deterministic color. The configured scale_to_nm factor is applied to
all mesh coordinates before visualization so that physical sizes are handled
consistently across samples.

No mesh files are modified.

Example
-------
python scripts/visualize_annotated_samples.py \
    --configs \
        configs/sample_001.yaml \
        configs/sample_002.yaml \
        configs/sample_003.yaml \
        configs/sample_004.yaml \
    --output outputs/figures/annotated_meshes_colored.png \
    --view xy
"""

from __future__ import annotations

import argparse
import colorsys
import math
from pathlib import Path

import numpy as np
import pyvista as pv
import yaml
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize multiple labelled neuronal meshes using their renderer configs."
        )
    )

    parser.add_argument(
        "--configs",
        nargs="+",
        type=Path,
        required=True,
        help="One or more renderer YAML config files.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG path for the combined figure.",
    )

    parser.add_argument(
        "--view",
        choices=("xy", "xz", "yz", "iso"),
        default="xy",
        help="Camera view. XY is recommended for the thesis overview.",
    )

    parser.add_argument(
        "--panel-size",
        type=int,
        default=1000,
        help="Width/height of each panel in pixels.",
    )

    parser.add_argument(
        "--independent-scale",
        action="store_true",
        help=(
            "Fit each sample independently. By default all orthographic panels "
            "use the same physical display scale."
        ),
    )

    parser.add_argument(
        "--margin",
        type=float,
        default=1.08,
        help="Camera margin around each sample.",
    )

    parser.add_argument(
        "--individual-output-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for saving individual sample panels "
            "as sample_001.png, sample_002.png, etc."
        ),
    )

    return parser.parse_args()


def repo_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_config(config_path: Path):
    config_path = repo_path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found:\n  {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    input_cfg = config.get("input", {})

    if input_cfg.get("mode") != "labelled_components":
        raise ValueError(
            f"{config_path.name}: input.mode must be 'labelled_components'."
        )

    labelled_dir_value = input_cfg.get("labelled_dir")
    if not labelled_dir_value:
        raise KeyError(
            f"{config_path.name}: missing input.labelled_dir."
        )

    labelled_dir = repo_path(labelled_dir_value)

    if not labelled_dir.is_dir():
        raise FileNotFoundError(
            f"{config_path.name}: labelled directory not found:\n"
            f"  {labelled_dir}"
        )

    dendrite_pattern = input_cfg.get(
        "dendrite_pattern",
        "dendrite*.ply",
    )
    spine_pattern = input_cfg.get(
        "spine_pattern",
        "spine*.ply",
    )

    scale_to_nm = float(
        input_cfg.get("scale_to_nm", 1.0)
    )

    dendrite_files = sorted(
        labelled_dir.glob(dendrite_pattern)
    )
    spine_files = sorted(
        labelled_dir.glob(spine_pattern)
    )

    if not dendrite_files:
        raise FileNotFoundError(
            f"{config_path.name}: no dendrite mesh matched "
            f"{dendrite_pattern!r}."
        )

    if not spine_files:
        raise FileNotFoundError(
            f"{config_path.name}: no spine meshes matched "
            f"{spine_pattern!r}."
        )

    # Prefer the configured output name as sample label; otherwise use folder name.
    sample_name = (
        config.get("output", {}).get("output_name")
        or labelled_dir.name
    )

    return {
        "config_path": config_path,
        "sample_name": sample_name,
        "labelled_dir": labelled_dir,
        "dendrite_file": dendrite_files[0],
        "spine_files": spine_files,
        "scale_to_nm": scale_to_nm,
    }


def scaled_mesh(path: Path, scale_to_nm: float):
    """Read a mesh and apply the renderer's configured coordinate scale."""
    mesh = pv.read(str(path))

    if mesh.n_points == 0:
        return mesh

    mesh.points = (
        np.asarray(mesh.points, dtype=np.float64)
        * scale_to_nm
    )

    return mesh


def generate_spine_colors(count: int):
    """
    Generate deterministic visually separated colors.

    The golden-ratio hue step avoids giving neighboring spine IDs nearly
    identical colors.
    """
    colors = []
    golden_ratio = 0.618033988749895

    for index in range(count):
        hue = (index * golden_ratio) % 1.0
        saturation = 0.70 + 0.15 * (index % 2)
        value = 0.86 + 0.08 * ((index // 2) % 2)

        colors.append(
            colorsys.hsv_to_rgb(
                hue,
                min(saturation, 0.90),
                min(value, 0.96),
            )
        )

    return colors


def get_sample_bounds(sample):
    """
    Determine scaled physical bounds while loading one file at a time.

    This avoids merging all meshes and keeps peak memory use lower.
    """
    mins = np.array(
        [np.inf, np.inf, np.inf],
        dtype=np.float64,
    )
    maxs = np.array(
        [-np.inf, -np.inf, -np.inf],
        dtype=np.float64,
    )

    all_files = [
        sample["dendrite_file"],
        *sample["spine_files"],
    ]

    for mesh_path in all_files:
        mesh = scaled_mesh(
            mesh_path,
            sample["scale_to_nm"],
        )

        if mesh.n_points == 0:
            continue

        points = np.asarray(
            mesh.points,
            dtype=np.float64,
        )

        mins = np.minimum(
            mins,
            points.min(axis=0),
        )
        maxs = np.maximum(
            maxs,
            points.max(axis=0),
        )

        del mesh

    if not np.all(np.isfinite(mins)):
        raise RuntimeError(
            f"No valid geometry found for {sample['sample_name']}."
        )

    return mins, maxs


def view_extent(bounds, view):
    mins, maxs = bounds
    extent = maxs - mins

    if view == "xy":
        return float(max(extent[0], extent[1]))
    if view == "xz":
        return float(max(extent[0], extent[2]))
    if view == "yz":
        return float(max(extent[1], extent[2]))

    return float(max(extent))


def set_camera(plotter, view, center, display_extent, margin):
    cx, cy, cz = center

    if view == "xy":
        plotter.view_xy()
        plotter.enable_parallel_projection()
        plotter.camera.focal_point = (cx, cy, cz)
        plotter.camera.position = (cx, cy, cz + 1.0)
        plotter.camera.up = (0.0, 1.0, 0.0)
        plotter.camera.parallel_scale = (
            0.5 * display_extent * margin
        )

    elif view == "xz":
        plotter.view_xz()
        plotter.enable_parallel_projection()
        plotter.camera.focal_point = (cx, cy, cz)
        plotter.camera.position = (cx, cy - 1.0, cz)
        plotter.camera.up = (0.0, 0.0, 1.0)
        plotter.camera.parallel_scale = (
            0.5 * display_extent * margin
        )

    elif view == "yz":
        plotter.view_yz()
        plotter.enable_parallel_projection()
        plotter.camera.focal_point = (cx, cy, cz)
        plotter.camera.position = (cx + 1.0, cy, cz)
        plotter.camera.up = (0.0, 0.0, 1.0)
        plotter.camera.parallel_scale = (
            0.5 * display_extent * margin
        )

    elif view == "iso":
        plotter.disable_parallel_projection()
        plotter.view_isometric()
        plotter.reset_camera()


def render_panel(
    sample,
    bounds,
    output_path,
    panel_letter,
    view,
    panel_size,
    display_extent,
    margin,
):
    mins, maxs = bounds
    center = 0.5 * (mins + maxs)

    plotter = pv.Plotter(
        off_screen=True,
        window_size=(panel_size, panel_size),
    )
    plotter.set_background("white")

    # Dendritic shaft in neutral gray.
    dendrite_mesh = scaled_mesh(
        sample["dendrite_file"],
        sample["scale_to_nm"],
    )

    plotter.add_mesh(
        dendrite_mesh,
        color=(0.68, 0.68, 0.68),
        show_edges=False,
        smooth_shading=False,
    )

    # Individual annotated spines in distinct colors.
    colors = generate_spine_colors(
        len(sample["spine_files"])
    )

    for index, (spine_file, color) in enumerate(
        zip(sample["spine_files"], colors),
        start=1,
    ):
        spine_mesh = scaled_mesh(
            spine_file,
            sample["scale_to_nm"],
        )

        plotter.add_mesh(
            spine_mesh,
            color=color,
            show_edges=False,
            smooth_shading=False,
        )

        if index % 25 == 0 or index == len(sample["spine_files"]):
            print(
                f"{sample['sample_name']}: "
                f"loaded {index}/{len(sample['spine_files'])} spines"
            )

    set_camera(
        plotter,
        view,
        center,
        display_extent,
        margin,
    )

    plotter.add_text(
        f"({panel_letter})",
        position="upper_left",
        font_size=16,
        color="black",
        shadow=False,
    )

    # Small XYZ orientation triad for spatial reference.
    plotter.add_axes(
        line_width=2,
        labels_off=False,
    )

    plotter.screenshot(
        str(output_path),
        transparent_background=False,
    )
    plotter.close()


def combine_panels(panel_paths, output_path):
    images = [
        Image.open(path).convert("RGB")
        for path in panel_paths
    ]

    count = len(images)
    columns = 2 if count > 1 else 1
    rows = math.ceil(count / columns)

    panel_w = max(image.width for image in images)
    panel_h = max(image.height for image in images)

    canvas = Image.new(
        "RGB",
        (columns * panel_w, rows * panel_h),
        "white",
    )

    for index, image in enumerate(images):
        row = index // columns
        col = index % columns

        x = col * panel_w
        y = row * panel_h

        canvas.paste(
            ImageOps.pad(
                image,
                (panel_w, panel_h),
                color="white",
            ),
            (x, y),
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(
        output_path,
        dpi=(300, 300),
    )


def main():
    args = parse_args()

    samples = [
        load_config(config_path)
        for config_path in args.configs
    ]

    print("\nConfigured samples")
    print("------------------")

    for sample in samples:
        print(
            f"{sample['sample_name']}: "
            f"{len(sample['spine_files'])} spines, "
            f"scale_to_nm={sample['scale_to_nm']:g}, "
            f"folder={sample['labelled_dir']}"
        )

    # First pass: determine physical bounds without keeping all meshes in memory.
    bounds_list = []

    for sample in samples:
        bounds = get_sample_bounds(sample)
        bounds_list.append(bounds)

    if args.independent_scale:
        display_extents = [
            view_extent(bounds, args.view)
            for bounds in bounds_list
        ]
    else:
        common_extent = max(
            view_extent(bounds, args.view)
            for bounds in bounds_list
        )
        display_extents = [
            common_extent
            for _ in samples
        ]

    output_path = repo_path(args.output)

    temporary_dir = (
        output_path.parent
        / f".{output_path.stem}_panels"
    )
    temporary_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    panel_letters = "abcdefghijklmnopqrstuvwxyz"
    panel_paths = []

    individual_output_dir = None
    if args.individual_output_dir is not None:
        individual_output_dir = repo_path(args.individual_output_dir)
        individual_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    for index, (sample, bounds, display_extent) in enumerate(
        zip(samples, bounds_list, display_extents)
    ):
        panel_letter = panel_letters[index]
        panel_path = temporary_dir / f"panel_{index:02d}.png"

        render_panel(
            sample=sample,
            bounds=bounds,
            output_path=panel_path,
            panel_letter=panel_letter,
            view=args.view,
            panel_size=args.panel_size,
            display_extent=display_extent,
            margin=args.margin,
        )

        panel_paths.append(panel_path)

        if individual_output_dir is not None:
            individual_path = (
                individual_output_dir
                / f"{sample['sample_name']}.png"
            )

            # Copy the already rendered thesis-style panel so that the
            # individual file is identical to its panel in the combined figure.
            Image.open(panel_path).convert("RGB").save(
                individual_path,
                dpi=(300, 300),
            )

            print(
                f"Saved individual  : {individual_path}"
            )

    combine_panels(
        panel_paths,
        output_path,
    )

    print("\nSaved combined figure:")
    print(f"  {output_path}")

    if not args.independent_scale and args.view != "iso":
        print(
            "All panels use the same physical display scale "
            "after applying each config's scale_to_nm."
        )


if __name__ == "__main__":
    main()
