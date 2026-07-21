"""

This script reads a YAML config file, loads mesh geometry, builds an output
voxel grid, renders a synthetic microscopy stack, and saves image/mask/metadata.

Coordinate convention:
    - Mesh coordinates are XYZ in nanometres.
    - Output arrays/images are ZYX: [Z slices, Y pixels, X pixels].

Supported input modes:
    - single_mesh: one mesh is rendered and one object mask is saved.
    - labelled_components: dendrite and spine meshes are rendered separately,
      allowing separate dendrite and spine ground-truth masks.

Supported renderers:
    - voxel_grid
    - gaussian_splatting
"""

import argparse
import sys
from pathlib import Path

import torch

# Allow imports from the repository root when running:
# PYTHONPATH=. python scripts/render.py --config configs/default.yaml
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.settings import load_config, resolve_path
from src.mesh_io import (
    prepare_mesh_for_sim,
    get_combined_bbox_nm,
    find_labelled_component_paths,
    prepare_labelled_components_for_sim,
    get_mesh_anchor_nm,
)
from src.roi import compute_full_bbox, compute_roi_bbox, compute_voxel_grid
from src.psf import load_psf_zyx, make_gaussian_psf_matched_zyx
from src.voxel_renderer import (
    ensure_psf_odd_xy,
    render_single_mesh_voxel,
)
from src.splatting_renderer import render_single_mesh_splatting
from src.noise import apply_noise_if_enabled
from src.labels import make_object_mask_from_volume
from src.io_utils import (
    ensure_dir,
    tensor_to_u16_stack,
    binary_mask_to_u16,
    save_stack_imagej_zyx_u16,
    save_metadata_json,
)


def load_effective_psf(config, grid_cfg, device):
    """
    Load or generate the effective PSF used for image formation.

    The PSF represents the microscope blur. For Born-Wolf PSFs, the PSF is
    loaded from a TIFF file. For gaussian_2p, an analytical Gaussian PSF is
    generated from the config.

    Returns:
        psf_eff:
            Torch tensor PSF in ZYX order.
        psf_mode:
            Name of the PSF mode used.
    """
    psf_cfg = config["psf"]
    psf_mode = psf_cfg.get("mode", "bornwolf_2p")

    if psf_mode in ["bornwolf", "bornwolf_1p", "bornwolf_2p"]:
        psf_path = resolve_path(psf_cfg["path"])

        two_photon_like = bool(psf_cfg.get("two_photon_like", False))

        # Force correct behaviour from the selected named PSF mode.
        if psf_mode == "bornwolf_1p":
            two_photon_like = False

        if psf_mode == "bornwolf_2p":
            two_photon_like = True

        psf_eff = load_psf_zyx(
            str(psf_path),
            two_photon_like=two_photon_like,
            verbose=True,
        )

    elif psf_mode == "gaussian_2p":
        psf_eff = make_gaussian_psf_matched_zyx(
            shape_zyx=tuple(psf_cfg.get("shape_zyx", [13, 65, 65])),
            lambda_nm=float(psf_cfg.get("lambda_nm", 488.0)),
            na=float(psf_cfg.get("na", 1.0)),
            n=float(psf_cfg.get("refractive_index", 1.33)),
            xy_um_per_px=float(grid_cfg["xy_um_per_px"]),
            z_step_um=float(grid_cfg["z_step_um"]),
            sigma_scale_xy=float(psf_cfg.get("sigma_scale_xy", 1.0)),
            sigma_scale_z=float(psf_cfg.get("sigma_scale_z", 1.0)),
            two_photon_like=True,
            verbose=True,
        )

    else:
        raise ValueError(f"Unknown PSF mode: {psf_mode}")

    # Some convolution operations work more cleanly with odd XY dimensions.
    # This also keeps the PSF centre well defined.
    psf_eff = ensure_psf_odd_xy(psf_eff, renormalize=True, device=device)

    return psf_eff, psf_mode


def render_one_mesh(mesh_path, grid, psf_eff, config, device, tag):
    """
    Render one mesh using the renderer selected in the config.

    This function is a small dispatcher. It keeps the main script independent
    of the specific rendering implementation.
    """
    method = config["renderer"].get("method", "voxel_grid")

    if method == "voxel_grid":
        return render_single_mesh_voxel(
            mesh_path=mesh_path,
            grid=grid,
            psf_eff=psf_eff,
            config=config,
            device=device,
        )

    if method == "gaussian_splatting":
        return render_single_mesh_splatting(
            mesh_path=mesh_path,
            grid=grid,
            psf_eff=psf_eff,
            config=config,
            device=device,
            tag=tag,
        )

    raise ValueError(f"Unknown renderer method: {method}")


def save_volume(vol, out_dir, tag, grid_cfg):
    """
    Save a rendered floating-point volume as a uint16 ImageJ TIFF stack.

    The volume is expected in ZYX order.
    """
    path = save_stack_imagej_zyx_u16(
        out_dir=out_dir,
        tag=tag,
        stack_u16_zyx=tensor_to_u16_stack(vol),
        xy_um_per_px=float(grid_cfg["xy_um_per_px"]),
        z_step_um=float(grid_cfg["z_step_um"]),
    )
    print(f"  Saved: {path}")
    return path


def save_mask(mask, out_dir, tag, grid_cfg):
    """
    Save a binary mask as a uint16 ImageJ TIFF stack.

    The mask is expected in ZYX order.
    """
    path = save_stack_imagej_zyx_u16(
        out_dir=out_dir,
        tag=tag,
        stack_u16_zyx=binary_mask_to_u16(mask),
        xy_um_per_px=float(grid_cfg["xy_um_per_px"]),
        z_step_um=float(grid_cfg["z_step_um"]),
    )
    print(f"  Saved: {path}")
    return path


def threshold_mask(vol, rel_threshold):
    """
    Create a binary mask from a rendered component volume.

    The threshold is relative to the maximum intensity of that component.
    Example:
        rel_threshold = 0.1 means threshold = 10% of max intensity.
    """
    vmax = float(vol.max().item())
    threshold = rel_threshold * vmax if vmax > 0 else 0.0
    mask = (vol > threshold).to(torch.float32)
    print(f"  threshold={threshold:.4f}, voxels={int(mask.sum().item())}")
    return mask


def get_output_shape_from_config(grid_cfg):
    """
    Read the output shape mode from the config.

    shape_mode:
        auto:
            The output shape is computed from the mesh bbox or ROI.
        fixed:
            The output shape is taken from output_shape_zyx.

    Important:
        output_shape_zyx order is [Z, Y, X], matching the saved image array.
    """
    shape_mode = grid_cfg.get("shape_mode", "auto")

    if shape_mode == "auto":
        return None

    if shape_mode == "fixed":
        if "output_shape_zyx" not in grid_cfg:
            raise ValueError(
                "grid.output_shape_zyx must be set when grid.shape_mode is 'fixed'"
            )

        output_shape_zyx = tuple(int(v) for v in grid_cfg["output_shape_zyx"])

        if len(output_shape_zyx) != 3:
            raise ValueError(
                "grid.output_shape_zyx must have three values in order [Z, Y, X]"
            )

        return output_shape_zyx

    raise ValueError("grid.shape_mode must be 'auto' or 'fixed'")


def main():
    """
    Main rendering workflow.

    Steps:
        1. Load config.
        2. Prepare input meshes.
        3. Build the output voxel grid.
        4. Load/generate PSF.
        5. Render image stack.
        6. Generate masks.
        7. Save image, masks, and metadata.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    input_cfg = config["input"]
    output_cfg = config["output"]
    grid_cfg = config["grid"]
    renderer_cfg = config["renderer"]
    mask_cfg = config.get("masks", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_mode = input_cfg.get("mode", "single_mesh")
    method = renderer_cfg.get("method", "voxel_grid")

    output_dir = resolve_path(output_cfg["output_dir"])
    output_name = output_cfg.get("output_name", "render")
    ensure_dir(output_dir)

    print("=" * 60)
    print("Synthetic Microscopy Renderer")
    print("=" * 60)
    print(f"Device     : {device}")
    print(f"Input mode : {input_mode}")
    print(f"Renderer   : {method}")
    print(f"Output     : {output_dir}")
    print("=" * 60)

    # ------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------
    # single_mesh:
    #     Render one mesh and create one object mask.
    #
    # labelled_components:
    #     Render dendrite and spines separately so that separate GT masks
    #     can be generated for dendrite and spine classes.
    # ------------------------------------------------------------
    if input_mode == "single_mesh":
        mesh_path = resolve_path(input_cfg["mesh_path"])

        sim_mesh_path = prepare_mesh_for_sim(
            mesh_path=mesh_path,
            scale_to_nm=float(input_cfg.get("scale_to_nm", 1.0)),
            recenter=bool(input_cfg.get("recenter", False)),
        )

        all_sim_paths = [sim_mesh_path]

    elif input_mode == "labelled_components":
        labelled_dir = resolve_path(input_cfg["labelled_dir"])

        dendrite_path, spine_paths = find_labelled_component_paths(
            labelled_dir=labelled_dir,
            dendrite_pattern=input_cfg.get("dendrite_pattern", "dendrite*.ply"),
            spine_pattern=input_cfg.get("spine_pattern", "spine*.ply"),
        )

        sim_dendrite_path, sim_spine_paths = prepare_labelled_components_for_sim(
            dendrite_path=dendrite_path,
            spine_paths=spine_paths,
            scale_to_nm=float(input_cfg.get("scale_to_nm", 1.0)),
            recenter=bool(input_cfg.get("recenter", False)),
        )

        all_sim_paths = [sim_dendrite_path] + sim_spine_paths

    else:
        raise ValueError(
            "input.mode must be 'single_mesh' or 'labelled_components'"
        )

    # ------------------------------------------------------------
    # Grid / ROI creation
    # ------------------------------------------------------------
    # The combined bounding box defines the physical region that can be
    # rendered. The final render region can be the full bbox or a cropped ROI.
    # The voxel grid stores origin, voxel size, and output shape.
    # ------------------------------------------------------------
    bbox_dict = get_combined_bbox_nm(all_sim_paths)

    print("\nCombined bbox nm:")
    print(f"  X: [{bbox_dict['xmin']:.1f}, {bbox_dict['xmax']:.1f}]")
    print(f"  Y: [{bbox_dict['ymin']:.1f}, {bbox_dict['ymax']:.1f}]")
    print(f"  Z: [{bbox_dict['zmin']:.1f}, {bbox_dict['zmax']:.1f}]")

    if bool(grid_cfg.get("use_roi", False)):
        render_bbox = compute_roi_bbox(
            bbox_dict,
            roi_size_um_x=float(grid_cfg.get("roi_size_um_x", 200.0)),
            roi_size_um_y=float(grid_cfg.get("roi_size_um_y", 200.0)),
            margin=float(grid_cfg.get("margin", 0.05)),
        )
        print("Using ROI bbox.")
    else:
        render_bbox = compute_full_bbox(
            bbox_dict,
            margin=float(grid_cfg.get("margin", 0.05)),
        )
        print("Using full bbox.")

    output_shape_zyx = get_output_shape_from_config(grid_cfg)

    # A fixed patch must be centred on actual geometry. The centre of the
    # complete bounding box can lie in empty space, especially for curved or
    # branched dendrites, which produces an all-black fixed-size render.
    fixed_center_xyz_nm = None
    if output_shape_zyx is not None:
        if input_mode == "labelled_components":
            fixed_center_xyz_nm = get_mesh_anchor_nm(sim_dendrite_path)
        else:
            fixed_center_xyz_nm = get_mesh_anchor_nm(sim_mesh_path)

    grid = compute_voxel_grid(
        render_bbox,
        xy_um_per_px=float(grid_cfg["xy_um_per_px"]),
        z_step_um=float(grid_cfg["z_step_um"]),
        output_shape_zyx=output_shape_zyx,
        fixed_center_xyz_nm=fixed_center_xyz_nm,
    )

    # ------------------------------------------------------------
    # PSF loading
    # ------------------------------------------------------------
    # The PSF is applied during rendering as the microscope image-formation
    # step. This converts a density/splat volume into a microscopy-like stack.
    # ------------------------------------------------------------
    psf_eff, psf_mode = load_effective_psf(config, grid_cfg, device)

    # ------------------------------------------------------------
    # Render mode: single mesh
    # ------------------------------------------------------------
    if input_mode == "single_mesh":
        vol = render_one_mesh(
            sim_mesh_path,
            grid,
            psf_eff,
            config,
            device,
            tag="single_mesh",
        )

        # Noise can be disabled for clean data generation or enabled for noisy simulations.
        vol = apply_noise_if_enabled(vol, config)

        image_path = save_volume(
            vol,
            output_dir,
            f"{output_name}_{method}_{psf_mode}_image",
            grid_cfg,
        )

        # For single_mesh mode, only one binary object mask is produced.
        mask = make_object_mask_from_volume(
            vol,
            rel_threshold=float(mask_cfg.get("object_rel_threshold", 0.1)),
        )

        mask_path = save_mask(
            mask,
            output_dir,
            f"{output_name}_{method}_{psf_mode}_object_mask",
            grid_cfg,
        )

        metadata = {
            "input_mode": input_mode,
            "renderer": method,
            "labeling_mode": renderer_cfg.get("labeling_mode", "membrane"),
            "psf_mode": psf_mode,
            "splatting_apply_psf": bool(
                config.get("splatting", {}).get("apply_psf", False)
            ),
            "shape_mode": grid_cfg.get("shape_mode", "auto"),
            "output_shape_zyx": list(grid["shape_zyx"]),
            "device": str(device),
            "image_path": image_path,
            "mask_path": mask_path,
        }

    # ------------------------------------------------------------
    # Render mode: labelled components
    # ------------------------------------------------------------
    # Dendrite and spine components are rendered separately. This is slower,
    # but it gives separate clean volumes and separate GT masks.
    # ------------------------------------------------------------
    else:
        print("\n--- Rendering dendrite ---")

        vol_dendrite = render_one_mesh(
            sim_dendrite_path,
            grid,
            psf_eff,
            config,
            device,
            tag="dendrite",
        )

        print("\n--- Rendering spines combined ---")

        vol_spines = torch.zeros_like(vol_dendrite)

        for i, sp in enumerate(sim_spine_paths, start=1):
            print(f"\nSpine {i}/{len(sim_spine_paths)}")
            vol_sp = render_one_mesh(
                sp,
                grid,
                psf_eff,
                config,
                device,
                tag=f"spine_{i}",
            )

            vol_spines = vol_spines + vol_sp

            del vol_sp
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Combined clean image contains dendrite + spine signal.
        vol_all = vol_dendrite + vol_spines
        vol_all = apply_noise_if_enabled(vol_all, config)

        image_path = save_volume(
            vol_all,
            output_dir,
            f"{output_name}_{method}_{psf_mode}_image",
            grid_cfg,
        )

        # Optional clean component images help debugging mask alignment.
        if bool(mask_cfg.get("save_clean_component_images", True)):
            save_volume(
                vol_dendrite,
                output_dir,
                f"{output_name}_{method}_{psf_mode}_dendrite_clean",
                grid_cfg,
            )
            save_volume(
                vol_spines,
                output_dir,
                f"{output_name}_{method}_{psf_mode}_spines_clean",
                grid_cfg,
            )

        print("\n--- Creating masks ---")

        print("Spine mask:")
        spine_mask = threshold_mask(
            vol_spines,
            float(mask_cfg.get("spine_rel_threshold", 0.1)),
        )

        print("Dendrite mask:")
        dendrite_mask = threshold_mask(
            vol_dendrite,
            float(mask_cfg.get("dendrite_rel_threshold", 0.1)),
        )

        spine_mask_path = save_mask(
            spine_mask,
            output_dir,
            f"{output_name}_{method}_{psf_mode}_spine_mask",
            grid_cfg,
        )

        dendrite_mask_path = save_mask(
            dendrite_mask,
            output_dir,
            f"{output_name}_{method}_{psf_mode}_dendrite_mask",
            grid_cfg,
        )

        del vol_all, spine_mask, dendrite_mask

        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Optional: save one mask per individual spine.
        # This is useful for instance-level evaluation, but can take longer.
        if bool(mask_cfg.get("save_individual_spine_masks", True)):
            print("\n--- Saving individual spine masks ---")

            for i, sp in enumerate(sim_spine_paths, start=1):
                print(f"\nIndividual spine {i}/{len(sim_spine_paths)}")

                vol_sp = render_one_mesh(
                    sp,
                    grid,
                    psf_eff,
                    config,
                    device,
                    tag=f"spine_{i}_individual",
                )

                sp_mask = threshold_mask(
                    vol_sp,
                    float(mask_cfg.get("spine_rel_threshold", 0.1)),
                )

                save_mask(
                    sp_mask,
                    output_dir,
                    f"{output_name}_{method}_{psf_mode}_spine{i}_mask",
                    grid_cfg,
                )

                if bool(mask_cfg.get("save_clean_component_images", True)):
                    save_volume(
                        vol_sp,
                        output_dir,
                        f"{output_name}_{method}_{psf_mode}_spine{i}_clean",
                        grid_cfg,
                    )

                del vol_sp, sp_mask

                if device.type == "cuda":
                    torch.cuda.empty_cache()

        metadata = {
            "input_mode": input_mode,
            "renderer": method,
            "labeling_mode": renderer_cfg.get("labeling_mode", "membrane"),
            "psf_mode": psf_mode,
            "splatting_apply_psf": bool(
                config.get("splatting", {}).get("apply_psf", False)
            ),
            "shape_mode": grid_cfg.get("shape_mode", "auto"),
            "output_shape_zyx": list(grid["shape_zyx"]),
            "device": str(device),
            "num_spines": len(sim_spine_paths),
            "image_path": image_path,
            "spine_mask_path": spine_mask_path,
            "dendrite_mask_path": dendrite_mask_path,
        }

    # ------------------------------------------------------------
    # Save metadata
    # ------------------------------------------------------------
    # Metadata records the settings and output paths needed to reproduce
    # or inspect the generated image/mask stack.
    # ------------------------------------------------------------
    metadata_path = save_metadata_json(
        output_dir,
        tag=f"{output_name}_{method}_{psf_mode}",
        metadata=metadata,
    )

    print("\nSaved metadata:")
    print(f"  {metadata_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()