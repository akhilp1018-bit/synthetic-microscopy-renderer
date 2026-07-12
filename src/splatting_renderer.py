import torch

from src.voxel_renderer import (
    mesh_to_density_zyx,
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

    Concept:
        mesh surface
        -> deterministic surface samples into sparse grid
        -> Gaussian smoothing as splats
        -> optional PSF convolution

    This keeps the same mesh sampling logic as the old code,
    but changes the rendering interpretation to splatting.
    """
    splat_cfg = config.get("splatting", {})

    spacing_nm = float(splat_cfg.get("spacing_nm", 100))
    sigma_zyx = tuple(splat_cfg.get("sigma_zyx", [1.0, 2.0, 2.0]))
    apply_psf = bool(splat_cfg.get("apply_psf", False))

    print(f"\n{'=' * 50}")
    print(f"Gaussian splatting: {tag}")
    print(f"Mesh: {mesh_path}")
    print(f"spacing_nm={spacing_nm}")
    print(f"sigma_zyx={sigma_zyx}")
    print(f"apply_psf={apply_psf}")
    print(f"{'=' * 50}")

    rho = mesh_to_density_zyx(
        mesh_path=mesh_path,
        origin_nm=grid["origin_nm"],
        voxel_size_nm_xyz=grid["voxel_size_nm_xyz"],
        shape_zyx=grid["shape_zyx"],
        spacing_nm=spacing_nm,
        device=device,
        batch_faces=int(config["renderer"].get("batch_faces", 2048)),
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