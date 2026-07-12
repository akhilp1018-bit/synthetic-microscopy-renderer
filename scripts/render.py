import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.settings import load_config, resolve_path
from src.mesh_io import (
    prepare_mesh_for_sim,
    get_combined_bbox_nm,
    find_labelled_component_paths,
    prepare_labelled_components_for_sim,
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
    psf_cfg = config["psf"]
    psf_mode = psf_cfg.get("mode", "bornwolf_2p")

    if psf_mode in ["bornwolf", "bornwolf_1p", "bornwolf_2p"]:
        psf_path = resolve_path(psf_cfg["path"])

        two_photon_like = bool(psf_cfg.get("two_photon_like", False))

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

    psf_eff = ensure_psf_odd_xy(psf_eff, renormalize=True, device=device)

    return psf_eff, psf_mode


def render_one_mesh(mesh_path, grid, psf_eff, config, device, tag):
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
    vmax = float(vol.max().item())
    threshold = rel_threshold * vmax if vmax > 0 else 0.0
    mask = (vol > threshold).to(torch.float32)
    print(f"  threshold={threshold:.4f}, voxels={int(mask.sum().item())}")
    return mask


def main():
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
    # Grid
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

    grid = compute_voxel_grid(
        render_bbox,
        xy_um_per_px=float(grid_cfg["xy_um_per_px"]),
        z_step_um=float(grid_cfg["z_step_um"]),
    )

    # ------------------------------------------------------------
    # PSF
    # ------------------------------------------------------------
    psf_eff, psf_mode = load_effective_psf(config, grid_cfg, device)

    # ------------------------------------------------------------
    # Render single mesh
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

        vol = apply_noise_if_enabled(vol, config)

        image_path = save_volume(
            vol,
            output_dir,
            f"{output_name}_{method}_{psf_mode}_image",
            grid_cfg,
        )

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
            "splatting_apply_psf": bool(config.get("splatting", {}).get("apply_psf", False)),
            "device": str(device),
            "image_path": image_path,
            "mask_path": mask_path,
            "shape_zyx": list(grid["shape_zyx"]),
        }

    # ------------------------------------------------------------
    # Render labelled components
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

        vol_all = vol_dendrite + vol_spines
        vol_all = apply_noise_if_enabled(vol_all, config)

        image_path = save_volume(
            vol_all,
            output_dir,
            f"{output_name}_{method}_{psf_mode}_image",
            grid_cfg,
        )

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

        # --------------------------------------------------------
        # Individual spine masks
        # --------------------------------------------------------
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
            "splatting_apply_psf": bool(config.get("splatting", {}).get("apply_psf", False)),
            "device": str(device),
            "num_spines": len(sim_spine_paths),
            "image_path": image_path,
            "spine_mask_path": spine_mask_path,
            "dendrite_mask_path": dendrite_mask_path,
            "shape_zyx": list(grid["shape_zyx"]),
        }

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