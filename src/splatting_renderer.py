import torch

from src.voxel_renderer import (
    mesh_to_density_zyx,
    mesh_pseudofilled_to_density_zyx,
    smooth_density_zyx,
    focal_stack_from_density,
)


def render_single_mesh_splatting(
    mesh_path,
    grid,
    psf_eff,
    config,
    device,
    tag="splatting",
):
    """
    Gaussian splatting renderer.

    Supports:
        labeling_mode: membrane
        labeling_mode: pseudofilled

    membrane:
        mesh surface -> sparse surface density -> Gaussian splats

    pseudofilled:
        mesh surface -> pseudofilled density -> Gaussian splats

    apply_psf:
        false = splatting blur only
        true  = splatting blur + PSF convolution
    """
    renderer_cfg = config.get("renderer", {})
    splat_cfg = config.get("splatting", {})

    labeling_mode = renderer_cfg.get("labeling_mode", "membrane")

    spacing_nm = float(splat_cfg.get("spacing_nm", renderer_cfg.get("spacing_nm", 100)))
    sigma_zyx = tuple(splat_cfg.get("sigma_zyx", [1.0, 2.0, 2.0]))
    apply_psf = bool(splat_cfg.get("apply_psf", False))

    batch_faces = int(renderer_cfg.get("batch_faces", 2048))
    pseudofill_sigma_zyx = tuple(
        renderer_cfg.get("pseudofill_sigma_zyx", [2.0, 2.5, 2.5])
    )

    print(f"\n{'=' * 50}")
    print(f"Gaussian splatting: {tag}")
    print(f"Mesh: {mesh_path}")
    print(f"labeling_mode={labeling_mode}")
    print(f"spacing_nm={spacing_nm}")
    print(f"sigma_zyx={sigma_zyx}")
    print(f"apply_psf={apply_psf}")
    print(f"{'=' * 50}")

    if labeling_mode == "membrane":
        rho = mesh_to_density_zyx(
            mesh_path=mesh_path,
            origin_nm=grid["origin_nm"],
            voxel_size_nm_xyz=grid["voxel_size_nm_xyz"],
            shape_zyx=grid["shape_zyx"],
            spacing_nm=spacing_nm,
            device=device,
            batch_faces=batch_faces,
        )

    elif labeling_mode == "pseudofilled":
        rho = mesh_pseudofilled_to_density_zyx(
            mesh_path=mesh_path,
            origin_nm=grid["origin_nm"],
            voxel_size_nm_xyz=grid["voxel_size_nm_xyz"],
            shape_zyx=grid["shape_zyx"],
            spacing_nm=spacing_nm,
            device=device,
            batch_faces=batch_faces,
            pseudofill_sigma_zyx=pseudofill_sigma_zyx,
        )

    else:
        raise ValueError(
            "labeling_mode must be 'membrane' or 'pseudofilled'"
        )

    vol = smooth_density_zyx(
        rho,
        sigma_zyx=sigma_zyx,
        normalize_sum=True,
        device=device,
    )

    del rho

    if device.type == "cuda":
        torch.cuda.empty_cache()

    if apply_psf:
        vol = focal_stack_from_density(vol, psf_eff, device=device)

    return vol