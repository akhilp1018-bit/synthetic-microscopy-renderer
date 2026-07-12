import math
import numpy as np
import torch
import trimesh

from src.voxel_renderer import focal_stack_from_density


def _load_mesh(mesh_path):
    mesh = trimesh.load(mesh_path, force="mesh", process=False)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load mesh as Trimesh: {mesh_path}")

    return mesh


def _sample_surface_points(mesh, spacing_nm=100.0, seed=0):
    """
    Sample continuous points on the mesh surface.

    The number of points is estimated from the mesh surface area and the
    requested sampling spacing.
    """
    area_nm2 = float(mesh.area)
    n_points = max(1, int(math.ceil(area_nm2 / (spacing_nm ** 2))))

    print(f"Mesh surface area nm^2 : {area_nm2:.2f}")
    print(f"Sampling points        : {n_points}")

    points, face_idx = trimesh.sample.sample_surface(
        mesh,
        count=n_points,
        seed=seed,
    )

    return points.astype(np.float32)


def _splat_points_to_volume(
    points_xyz_nm,
    grid,
    sigma_zyx=(1.0, 2.0, 2.0),
    device=None,
    points_per_batch=50000,
):
    """
    Accumulate Gaussian splats from sampled mesh points into a ZYX volume.

    Each point is converted from physical XYZ coordinates into continuous
    voxel coordinates. A local 3D Gaussian kernel is then accumulated around
    the point position.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shape_zyx = tuple(grid["shape_zyx"])
    Z, Y, X = shape_zyx

    origin_nm = np.asarray(grid["origin_nm"], dtype=np.float32)
    voxel_size_nm_xyz = np.asarray(grid["voxel_size_nm_xyz"], dtype=np.float32)

    sigma_z, sigma_y, sigma_x = [float(v) for v in sigma_zyx]

    radius_z = max(1, int(math.ceil(3.0 * sigma_z)))
    radius_y = max(1, int(math.ceil(3.0 * sigma_y)))
    radius_x = max(1, int(math.ceil(3.0 * sigma_x)))

    dz = torch.arange(-radius_z, radius_z + 1, device=device)
    dy = torch.arange(-radius_y, radius_y + 1, device=device)
    dx = torch.arange(-radius_x, radius_x + 1, device=device)

    zz, yy, xx = torch.meshgrid(dz, dy, dx, indexing="ij")

    offsets = torch.stack(
        [zz.reshape(-1), yy.reshape(-1), xx.reshape(-1)],
        dim=1,
    ).long()

    print(f"Splat kernel radius ZYX : ({radius_z}, {radius_y}, {radius_x})")
    print(f"Splat kernel voxels     : {offsets.shape[0]}")

    vol_flat = torch.zeros(Z * Y * X, dtype=torch.float32, device=device)

    points = torch.as_tensor(points_xyz_nm, dtype=torch.float32, device=device)

    origin = torch.as_tensor(origin_nm, dtype=torch.float32, device=device)
    voxel_size = torch.as_tensor(
        voxel_size_nm_xyz,
        dtype=torch.float32,
        device=device,
    )

    points_vox_xyz = (points - origin) / voxel_size
    points_vox_z = points_vox_xyz[:, 2]
    points_vox_y = (Y - 1) - points_vox_xyz[:, 1]
    points_vox_x = points_vox_xyz[:, 0]  

    points_vox_zyx = torch.stack(
        [
            points_vox_z,
            points_vox_y,
            points_vox_x,
        ],
        dim=1,
    )

    n_points = points_vox_zyx.shape[0]

    for start in range(0, n_points, points_per_batch):
        end = min(start + points_per_batch, n_points)
        p = points_vox_zyx[start:end]

        center = torch.floor(p).long()
        loc = center[:, None, :] + offsets[None, :, :]

        z = loc[:, :, 0]
        y = loc[:, :, 1]
        x = loc[:, :, 2]

        valid = (
            (z >= 0) & (z < Z) &
            (y >= 0) & (y < Y) &
            (x >= 0) & (x < X)
        )

        dist = loc.float() - p[:, None, :]

        weights = torch.exp(
            -0.5 * (
                (dist[:, :, 0] / sigma_z) ** 2 +
                (dist[:, :, 1] / sigma_y) ** 2 +
                (dist[:, :, 2] / sigma_x) ** 2
            )
        )

        weights = weights * valid.float()

        flat_idx = z * (Y * X) + y * X + x
        flat_idx = flat_idx[valid]
        weights = weights[valid]

        vol_flat.scatter_add_(0, flat_idx, weights)

        print(f"  splatted points {start} - {end} / {n_points}")

    vol = vol_flat.reshape(Z, Y, X)

    total = vol.sum()
    if total > 0:
        vol = vol / total * float(n_points)

    return vol


def render_single_mesh_splatting(
    mesh_path,
    grid,
    psf_eff,
    config,
    device,
    tag="splatting",
):
    """
    Render a mesh using Gaussian splatting.

    Pipeline:
        mesh surface
        -> sampled surface points
        -> Gaussian kernel accumulation
        -> optional PSF convolution

    Currently supported label mode:
        membrane
    """
    renderer_cfg = config.get("renderer", {})
    splat_cfg = config.get("splatting", {})

    labeling_mode = renderer_cfg.get("labeling_mode", "membrane")

    if labeling_mode != "membrane":
        raise NotImplementedError(
            "Gaussian splatting currently supports labeling_mode='membrane'."
        )

    spacing_nm = float(splat_cfg.get("spacing_nm", renderer_cfg.get("spacing_nm", 100)))
    sigma_zyx = tuple(splat_cfg.get("sigma_zyx", [1.0, 2.0, 2.0]))
    apply_psf = bool(splat_cfg.get("apply_psf", True))
    seed = int(splat_cfg.get("seed", 0))
    points_per_batch = int(splat_cfg.get("points_per_batch", 50000))

    print(f"\n{'=' * 50}")
    print(f"Gaussian splatting: {tag}")
    print(f"Mesh: {mesh_path}")
    print(f"labeling_mode={labeling_mode}")
    print(f"spacing_nm={spacing_nm}")
    print(f"sigma_zyx={sigma_zyx}")
    print(f"apply_psf={apply_psf}")
    print(f"{'=' * 50}")

    mesh = _load_mesh(mesh_path)

    points_xyz_nm = _sample_surface_points(
        mesh,
        spacing_nm=spacing_nm,
        seed=seed,
    )

    vol = _splat_points_to_volume(
        points_xyz_nm=points_xyz_nm,
        grid=grid,
        sigma_zyx=sigma_zyx,
        device=device,
        points_per_batch=points_per_batch,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    if apply_psf:
        print("Applying PSF convolution...")
        vol = focal_stack_from_density(vol, psf_eff, device=device)

    return vol