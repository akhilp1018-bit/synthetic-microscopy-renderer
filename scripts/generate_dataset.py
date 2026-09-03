"""
Synthetic dataset generator for DeepD3 training data.

IMPORTANT
---------
This script and dataset_v1.yaml are specifically for dataset generation.
The dataset configuration is NOT intended for scripts/render.py.

The normal rendering pipeline continues to use:
    scripts/render.py
    configs/

Dataset generation uses:
    scripts/generate_dataset.py
    configs/dataset_v1.yaml

Both pipelines reuse the same rendering functions from src/.

Pipeline
--------
1. Select an enabled source sample.
2. Randomly rotate dendrite + spines together.
3. Select a random location on the dendrite.
4. Create a fixed-size microscopy volume.
5. Render dendrite and spines using PyTorch.
6. Generate clean and noisy images.
7. Generate dendrite, spine and combined GT masks.
8. Save images, masks and metadata.

Coordinate convention
---------------------
Mesh coordinates : XYZ in nanometres
Image arrays      : ZYX [Z, Y, X]

Run
---
python scripts/generate_dataset.py --config configs/dataset_v1.yaml
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.settings import load_config, resolve_path

from src.mesh_io import (
    find_labelled_component_paths,
    get_combined_bbox_nm,
)

from src.transforms import (
    rotate_labelled_components,
    select_random_geometry_center,
)

from src.roi import (
    compute_full_bbox,
    compute_voxel_grid,
)

from src.psf import (
    make_gaussian_psf_matched_zyx,
    load_psf_zyx,
)

from src.voxel_renderer import (
    ensure_psf_odd_xy,
    render_single_mesh_voxel,
)

from src.splatting_renderer import (
    render_single_mesh_splatting,
)

from src.noise import apply_noise_if_enabled

from src.io_utils import (
    ensure_dir,
    tensor_to_stack,
    binary_mask_to_stack,
    save_stack_imagej_zyx,
)


# ---------------------------------------------------------------------
# PSF
# ---------------------------------------------------------------------

def load_psf(config, grid_cfg, device):
    """Load or generate the microscope PSF."""

    cfg = config["psf"]
    mode = cfg.get("mode", "gaussian_2p")

    if mode in (
        "bornwolf",
        "bornwolf_1p",
        "bornwolf_2p",
    ):
        two_photon = cfg.get(
            "two_photon_like",
            False,
        )

        if mode == "bornwolf_1p":
            two_photon = False

        elif mode == "bornwolf_2p":
            two_photon = True

        psf = load_psf_zyx(
            str(resolve_path(cfg["path"])),
            two_photon_like=two_photon,
            verbose=True,
        )

    elif mode == "gaussian_2p":

        psf = make_gaussian_psf_matched_zyx(
            shape_zyx=tuple(
                cfg.get(
                    "shape_zyx",
                    [13, 65, 65],
                )
            ),
            lambda_nm=float(
                cfg.get("lambda_nm", 488.0)
            ),
            na=float(
                cfg.get("na", 1.0)
            ),
            n=float(
                cfg.get(
                    "refractive_index",
                    1.33,
                )
            ),
            xy_um_per_px=float(
                grid_cfg["xy_um_per_px"]
            ),
            z_step_um=float(
                grid_cfg["z_step_um"]
            ),
            sigma_scale_xy=float(
                cfg.get("sigma_scale_xy", 1.0)
            ),
            sigma_scale_z=float(
                cfg.get("sigma_scale_z", 1.0)
            ),
            two_photon_like=True,
            verbose=True,
        )

    else:
        raise ValueError(
            f"Unknown PSF mode: {mode}"
        )

    return ensure_psf_odd_xy(
        psf,
        renormalize=True,
        device=device,
    )


# ---------------------------------------------------------------------
# Render one mesh
# ---------------------------------------------------------------------

def render_mesh(
    mesh_path,
    grid,
    psf,
    config,
    device,
    tag,
):
    """Render one mesh using the renderer selected in YAML."""

    method = config["renderer"].get(
        "method",
        "voxel_grid",
    )

    if method == "voxel_grid":

        return render_single_mesh_voxel(
            mesh_path=mesh_path,
            grid=grid,
            psf_eff=psf,
            config=config,
            device=device,
        )

    if method == "gaussian_splatting":

        return render_single_mesh_splatting(
            mesh_path=mesh_path,
            grid=grid,
            psf_eff=psf,
            config=config,
            device=device,
            tag=tag,
        )

    raise ValueError(
        f"Unknown renderer: {method}"
    )


# ---------------------------------------------------------------------
# GT mask
# ---------------------------------------------------------------------

def make_mask(volume, threshold):
    """Create a binary GT mask from a clean rendered volume."""

    vmax = float(
        volume.max().item()
    )

    if vmax <= 0:
        return torch.zeros_like(
            volume,
            dtype=torch.float32,
        )

    return (
        volume
        > float(threshold) * vmax
    ).to(torch.float32)


# ---------------------------------------------------------------------
# TIFF saving
# ---------------------------------------------------------------------

def save_tiff(
    volume,
    path,
    bit_depth,
    xy_um,
    z_um,
    is_mask=False,
):
    """Save intensity image or GT mask."""

    if is_mask:
        stack = binary_mask_to_stack(
            volume,
            bit_depth=bit_depth,
        )
    else:
        stack = tensor_to_stack(
            volume,
            bit_depth=bit_depth,
        )

    return save_stack_imagej_zyx(
        out_dir=path.parent,
        filename=path.name,
        stack_zyx=stack,
        xy_um_per_px=xy_um,
        z_step_um=z_um,
    )


# ---------------------------------------------------------------------
# Find enabled source samples
# ---------------------------------------------------------------------

def get_enabled_samples(config):

    samples = []

    for name, cfg in config["samples"].items():

        if not cfg.get(
            "enabled",
            False,
        ):
            continue

        labelled_dir = resolve_path(
            cfg["labelled_dir"]
        )

        dendrite, spines = (
            find_labelled_component_paths(
                labelled_dir,
                dendrite_pattern=cfg.get(
                    "dendrite_pattern",
                    "dendrite*.ply",
                ),
                spine_pattern=cfg.get(
                    "spine_pattern",
                    "spine*.ply",
                ),
            )
        )

        samples.append({
            "name": name,
            "scale_to_nm": float(
                cfg.get(
                    "scale_to_nm",
                    1.0,
                )
            ),
            "dendrite": dendrite,
            "spines": spines,
        })

    if not samples:
        raise RuntimeError(
            "No source samples are enabled."
        )

    return samples


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic microscopy training data."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/dataset_v1.yaml",
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    dataset_cfg = config["dataset"]
    generation_cfg = config["generation"]
    grid_cfg = config["grid"]
    output_cfg = config["output"]
    mask_cfg = config["masks"]

    # -----------------------------------------------------------------
    # Dataset settings
    # -----------------------------------------------------------------

    output_dir = resolve_path(
        dataset_cfg["output_dir"]
    )

    ensure_dir(output_dir)

    total_instances = int(
        dataset_cfg["total_instances"]
    )

    shape_zyx = tuple(
        map(
            int,
            grid_cfg["output_shape_zyx"],
        )
    )

    if len(shape_zyx) != 3:
        raise ValueError(
            "output_shape_zyx must be [Z, Y, X]"
        )

    xy_nm = float(
        grid_cfg["xy_nm_per_px"]
    )

    xy_um = (
        xy_nm / 1000.0
    )

    z_um = float(
        grid_cfg["z_step_um"]
    )

    bit_depth = int(
        output_cfg.get(
            "image_bit_depth",
            8,
        )
    )

    if bit_depth not in (8, 16):
        raise ValueError(
            "image_bit_depth must be 8 or 16"
        )

    # -----------------------------------------------------------------
    # PyTorch device
    # -----------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    samples = get_enabled_samples(
        config
    )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    print("=" * 60)
    print("Synthetic Dataset Generator")
    print("=" * 60)

    print(
        f"Dataset       : "
        f"{dataset_cfg['name']}"
    )

    print(
        f"Instances     : "
        f"{total_instances}"
    )

    print(
        f"Source meshes : "
        f"{[s['name'] for s in samples]}"
    )

    print(
        f"Shape ZYX     : "
        f"{shape_zyx}"
    )

    print(
        f"XY resolution : "
        f"{xy_nm} nm/px"
    )

    print(
        f"Z spacing     : "
        f"{z_um} um"
    )

    print(
        f"Bit depth     : "
        f"{bit_depth}"
    )

    print(
        f"Device        : "
        f"{device}"
    )

    if device.type == "cuda":
        print(
            f"GPU           : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print("=" * 60)

    # =================================================================
    # Generate dataset
    # =================================================================

    for index in range(
        1,
        total_instances + 1,
    ):

        instance_name = (
            f"instance_{index:06d}"
        )

        instance_dir = (
            output_dir
            / instance_name
        )

        ensure_dir(
            instance_dir
        )

        # Balanced use of all enabled source samples.
        sample = samples[
            (index - 1)
            % len(samples)
        ]

        print(
            f"\n[{index}/{total_instances}] "
            f"{instance_name} "
            f"<- {sample['name']}"
        )

        # -------------------------------------------------------------
        # Temporary transformed meshes
        # -------------------------------------------------------------
        #
        # These PLY files are automatically deleted after rendering.
        #

        with tempfile.TemporaryDirectory(
            prefix=f"{instance_name}_"
        ) as tmp:

            # ---------------------------------------------------------
            # Random orientation
            # ---------------------------------------------------------

            dendrite, spines, transform = (
                rotate_labelled_components(
                    dendrite_path=sample["dendrite"],
                    spine_paths=sample["spines"],
                    output_dir=Path(tmp),
                    scale_to_nm=sample["scale_to_nm"],
                    random_orientation=generation_cfg.get(
                        "random_orientation",
                        True,
                    ),
                )
            )

            # ---------------------------------------------------------
            # Random FOV location on actual dendrite geometry
            # ---------------------------------------------------------

            location_mode = generation_cfg.get(
                "location",
                "geometry_random",
            )

            if location_mode != "geometry_random":
                raise ValueError(
                    f"Unknown location mode: "
                    f"{location_mode}"
                )

            center_xyz = (
                select_random_geometry_center(
                    dendrite
                )
            )

            print(
                f"Center XYZ nm : "
                f"{center_xyz}"
            )

            # ---------------------------------------------------------
            # Fixed rendering grid
            # ---------------------------------------------------------

            bbox = get_combined_bbox_nm(
                [dendrite] + spines
            )

            render_bbox = (
                compute_full_bbox(
                    bbox,
                    margin=0.05,
                )
            )

            grid = compute_voxel_grid(
                render_bbox,
                xy_um_per_px=xy_um,
                z_step_um=z_um,
                output_shape_zyx=shape_zyx,
                fixed_center_xyz_nm=center_xyz,
            )

            renderer_grid_cfg = {
                **grid_cfg,
                "xy_um_per_px": xy_um,
            }

            # ---------------------------------------------------------
            # Microscope PSF
            # ---------------------------------------------------------

            psf = load_psf(
                config,
                renderer_grid_cfg,
                device,
            )

            # ---------------------------------------------------------
            # Render dendrite using PyTorch
            # ---------------------------------------------------------

            print(
                "Rendering dendrite..."
            )

            dendrite_volume = render_mesh(
                dendrite,
                grid,
                psf,
                config,
                device,
                "dendrite",
            )

            # ---------------------------------------------------------
            # Render spines using PyTorch
            # ---------------------------------------------------------

            print(
                f"Rendering "
                f"{len(spines)} spines..."
            )

            spine_volume = (
                torch.zeros_like(
                    dendrite_volume
                )
            )

            for i, spine in enumerate(
                spines,
                start=1,
            ):

                spine_render = render_mesh(
                    spine,
                    grid,
                    psf,
                    config,
                    device,
                    f"spine_{i}",
                )

                spine_volume += (
                    spine_render
                )

                del spine_render

            # ---------------------------------------------------------
            # Clean microscopy image
            # ---------------------------------------------------------

            clean = (
                dendrite_volume
                + spine_volume
            )

            # ---------------------------------------------------------
            # Ground-truth masks
            # ---------------------------------------------------------
            #
            # GT is calculated before microscopy noise is added.
            #

            dendrite_mask = make_mask(
                dendrite_volume,
                mask_cfg.get(
                    "dendrite_rel_threshold",
                    0.1,
                ),
            )

            spine_mask = make_mask(
                spine_volume,
                mask_cfg.get(
                    "spine_rel_threshold",
                    0.1,
                ),
            )

            combined_mask = (
                (dendrite_mask > 0)
                | (spine_mask > 0)
            ).to(torch.float32)

            # ---------------------------------------------------------
            # Add microscopy noise using PyTorch
            # ---------------------------------------------------------

            noisy = (
                apply_noise_if_enabled(
                    clean.clone(),
                    config,
                )
            )

            # ---------------------------------------------------------
            # Save outputs
            # ---------------------------------------------------------

            outputs = {
                "clean": (
                    clean,
                    False,
                ),
                "noisy": (
                    noisy,
                    False,
                ),
                "dendrite_mask": (
                    dendrite_mask,
                    True,
                ),
                "spine_mask": (
                    spine_mask,
                    True,
                ),
                "combined_mask": (
                    combined_mask,
                    True,
                ),
            }

            for (
                name,
                (volume, is_mask),
            ) in outputs.items():

                if output_cfg.get(
                    f"save_{name}",
                    True,
                ):

                    save_tiff(
                        volume,
                        instance_dir
                        / f"{name}.tif",
                        bit_depth,
                        xy_um,
                        z_um,
                        is_mask=is_mask,
                    )

            # ---------------------------------------------------------
            # Metadata
            # ---------------------------------------------------------

            if output_cfg.get(
                "save_metadata",
                True,
            ):

                metadata = {
                    "instance":
                        instance_name,

                    "source_sample":
                        sample["name"],

                    "source_spine_count":
                        len(sample["spines"]),

                    "rotation":
                        transform,

                    "center_xyz_nm":
                        [
                            float(v)
                            for v
                            in center_xyz
                        ],

                    "output_shape_zyx":
                        list(shape_zyx),

                    "xy_nm_per_px":
                        xy_nm,

                    "z_step_um":
                        z_um,

                    "image_bit_depth":
                        bit_depth,

                    "renderer":
                        config["renderer"],

                    "psf":
                        config["psf"],

                    "noise":
                        config["noise"],

                    "masks":
                        config["masks"],
                }

                with open(
                    instance_dir
                    / "metadata.json",
                    "w",
                    encoding="utf-8",
                ) as f:

                    json.dump(
                        metadata,
                        f,
                        indent=2,
                    )

            # ---------------------------------------------------------
            # GPU cleanup
            # ---------------------------------------------------------

            del (
                dendrite_volume,
                spine_volume,
                clean,
                noisy,
                dendrite_mask,
                spine_mask,
                combined_mask,
                psf,
            )

            if device.type == "cuda":
                torch.cuda.empty_cache()

        print(
            f"Completed: "
            f"{instance_name}"
        )

    print()
    print("=" * 60)

    print(
        f"Dataset completed: "
        f"{output_dir}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()