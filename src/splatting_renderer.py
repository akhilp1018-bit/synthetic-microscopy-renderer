"""
Gaussian splatting renderer.

This renderer converts a mesh surface into a synthetic fluorescence volume by:
    1. sampling continuous points on the mesh surface
    2. accumulating a local 3D Gaussian kernel around each sampled point
    3. optionally applying PSF convolution

Coordinate convention:
    Mesh coordinates are XYZ in nanometres.
    Output volumes are ZYX:
        [Z slices, Y pixels, X pixels]

Note:
    Gaussian splatting supports membrane-style labelling only.

    Each sampled surface point retains the normal of its source triangle.
    The local Gaussian is anisotropic: its narrow axis follows the membrane
    normal and its tangential extent lies along the membrane surface.
"""

import gc
import math
import time

import numpy as np
import torch
import trimesh

from src.voxel_renderer import (
    _load_and_crop_mesh_geometry,
    focal_stack_from_density,
)


def _load_cropped_mesh_to_grid(mesh_path, grid, sigma_nm):
    """
    Load only mesh geometry that can contribute to the splatting volume.

    The crop includes a margin equal to the local Gaussian splat radius.
    For supported binary PLY files, geometry is streamed in chunks through
    the shared memory-safe ROI loader from the voxel renderer, so the full
    source mesh does not need to be constructed in RAM.
    """
    origin = np.asarray(grid["origin_nm"], dtype=np.float64)
    voxel_size = np.asarray(
        grid["voxel_size_nm_xyz"],
        dtype=np.float64,
    )

    Z, Y, X = [int(v) for v in grid["shape_zyx"]]
    sx, sy, sz = voxel_size
    x0, y0, z0 = origin

    sigma_nm = float(sigma_nm)
    if sigma_nm <= 0:
        raise ValueError("Gaussian sigma must be positive.")

    # Convert the physical support radius independently along each voxel axis.
    radius_x = max(1, int(math.ceil(3.0 * sigma_nm / sx)))
    radius_y = max(1, int(math.ceil(3.0 * sigma_nm / sy)))
    radius_z = max(1, int(math.ceil(3.0 * sigma_nm / sz)))

    roi_min = np.array(
        [
            x0 - radius_x * sx,
            y0 - radius_y * sy,
            z0 - radius_z * sz,
        ],
        dtype=np.float64,
    )
    roi_max = np.array(
        [
            x0 + X * sx + radius_x * sx,
            y0 + Y * sy + radius_y * sy,
            z0 + Z * sz + radius_z * sz,
        ],
        dtype=np.float64,
    )

    vertices, faces, total_faces = _load_and_crop_mesh_geometry(
        mesh_path=mesh_path,
        roi_min=roi_min,
        roi_max=roi_max,
        face_filter_batch=65536,
    )

    if vertices is None or faces is None:
        print(
            f"ROI face filter: kept 0 / {total_faces} faces "
            "(no mesh surface can contribute to the render grid)"
        )
        return None

    kept_faces = len(faces)
    print(
        f"ROI face filter: kept {kept_faces} / {total_faces} faces "
        f"({100.0 * kept_faces / total_faces:.4f}%)"
    )

    # The cropped geometry is small enough to construct as a Trimesh and use
    # with trimesh's surface sampler. Keep processing disabled so geometry is
    # not modified.
    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )


def _sample_surface_points(mesh, spacing_nm=100.0, seed=0):
    """
    Sample continuous points on the mesh surface.

    The number of sampled points is estimated from the mesh surface area:

        n_points approximately surface_area / spacing_nm^2

    Smaller spacing gives more points and usually a smoother result, but it is
    slower and uses more memory.
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

    # Retain the source-face normal so each Gaussian can be oriented relative
    # to the local membrane surface.
    normals = np.asarray(mesh.face_normals[face_idx], dtype=np.float32)

    return points.astype(np.float32), normals


def _splat_points_to_volume(
    points_xyz_nm,
    normals_xyz,
    grid,
    sigma_tangent_nm=100.0,
    sigma_normal_nm=50.0,
    device=None,
    points_per_batch=50000,
):
    """
    Accumulate normal-oriented anisotropic Gaussian splats into a ZYX volume.

    Gaussian widths and orientation are defined in physical nanometre space.
    The Gaussian is broader within the local membrane plane and narrower along
    the physical surface normal. The physical displacement from each sampled
    surface point is evaluated at voxel centres, so anisotropic voxel spacing
    does not distort the Gaussian orientation or covariance.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Z, Y, X = tuple(grid["shape_zyx"])

    origin_nm = np.asarray(grid["origin_nm"], dtype=np.float32)
    voxel_size_nm_xyz = np.asarray(
        grid["voxel_size_nm_xyz"],
        dtype=np.float32,
    )

    sigma_tangent_nm = float(sigma_tangent_nm)
    sigma_normal_nm = float(sigma_normal_nm)

    if sigma_tangent_nm <= 0 or sigma_normal_nm <= 0:
        raise ValueError("Gaussian sigma values must be positive.")

    sx, sy, sz = [float(v) for v in voxel_size_nm_xyz]
    max_sigma_nm = max(sigma_tangent_nm, sigma_normal_nm)

    radius_x = max(1, int(math.ceil(3.0 * max_sigma_nm / sx)))
    radius_y = max(1, int(math.ceil(3.0 * max_sigma_nm / sy)))
    radius_z = max(1, int(math.ceil(3.0 * max_sigma_nm / sz)))

    dz = torch.arange(-radius_z, radius_z + 1, device=device)
    dy = torch.arange(-radius_y, radius_y + 1, device=device)
    dx = torch.arange(-radius_x, radius_x + 1, device=device)

    zz, yy, xx = torch.meshgrid(dz, dy, dx, indexing="ij")
    offsets = torch.stack(
        [zz.reshape(-1), yy.reshape(-1), xx.reshape(-1)],
        dim=1,
    ).long()

    print(
        f"Splat kernel radius ZYX : "
        f"({radius_z}, {radius_y}, {radius_x})"
    )
    print(f"Splat kernel voxels     : {offsets.shape[0]}")

    vol_flat = torch.zeros(
        Z * Y * X,
        dtype=torch.float32,
        device=device,
    )

    points = torch.as_tensor(
        points_xyz_nm,
        dtype=torch.float32,
        device=device,
    )
    normals = torch.as_tensor(
        normals_xyz,
        dtype=torch.float32,
        device=device,
    )

    origin = torch.as_tensor(
        origin_nm,
        dtype=torch.float32,
        device=device,
    )
    voxel_size = torch.as_tensor(
        voxel_size_nm_xyz,
        dtype=torch.float32,
        device=device,
    )

    # Continuous voxel coordinates are used only to locate nearby grid cells.
    points_vox_xyz = (points - origin) / voxel_size

    source_y = points_vox_xyz[:, 1]
    source_y_floor = torch.floor(source_y)

    points_vox_z = points_vox_xyz[:, 2]
    points_vox_x = points_vox_xyz[:, 0]

    centers_vox_zyx = torch.stack(
        [
            torch.floor(points_vox_z),
            (Y - 1) - source_y_floor,
            torch.floor(points_vox_x),
        ],
        dim=1,
    ).long()

    # Convert physical XYZ normals to physical output ZYX coordinates.
    # Y changes sign because the output volume uses the flipped Y convention.
    normals_zyx = torch.stack(
        [
            normals[:, 2],
            -normals[:, 1],
            normals[:, 0],
        ],
        dim=1,
    )
    normals_zyx = normals_zyx / torch.clamp(
        torch.linalg.norm(normals_zyx, dim=1, keepdim=True),
        min=1e-8,
    )

    # Physical sampled-point coordinates in output ZYX convention.
    # The Y coordinate is expressed relative to the top of the output grid.
    x0, y0, z0 = origin
    points_phys_zyx = torch.stack(
        [
            points[:, 2] - z0,
            (y0 + Y * voxel_size[1]) - points[:, 1],
            points[:, 0] - x0,
        ],
        dim=1,
    )

    spacing_zyx = torch.as_tensor(
        [sz, sy, sx],
        dtype=torch.float32,
        device=device,
    )

    n_points = points.shape[0]

    inv_tangent2 = 1.0 / (sigma_tangent_nm ** 2)
    inv_normal2 = 1.0 / (sigma_normal_nm ** 2)

    for start in range(0, n_points, points_per_batch):
        end = min(start + points_per_batch, n_points)

        p_phys = points_phys_zyx[start:end]
        n = normals_zyx[start:end]
        center = centers_vox_zyx[start:end]

        loc = center[:, None, :] + offsets[None, :, :]

        z = loc[:, :, 0]
        y = loc[:, :, 1]
        x = loc[:, :, 2]

        valid = (
            (z >= 0) & (z < Z)
            & (y >= 0) & (y < Y)
            & (x >= 0) & (x < X)
        )

        # Evaluate the Gaussian at physical voxel-centre coordinates.
        loc_phys = (loc.float() + 0.5) * spacing_zyx[None, None, :]
        dist_nm = loc_phys - p_phys[:, None, :]

        normal_distance_nm = torch.sum(
            dist_nm * n[:, None, :],
            dim=2,
        )
        dist_squared_nm2 = torch.sum(dist_nm * dist_nm, dim=2)
        tangent_squared_nm2 = torch.clamp(
            dist_squared_nm2 - normal_distance_nm ** 2,
            min=0.0,
        )

        exponent = (
            tangent_squared_nm2 * inv_tangent2
            + normal_distance_nm ** 2 * inv_normal2
        )

        weights = torch.exp(-0.5 * exponent)
        weights = weights * valid.float()

        flat_idx = z * (Y * X) + y * X + x
        flat_idx = flat_idx[valid]
        weights = weights[valid]

        vol_flat.scatter_add_(0, flat_idx, weights)

        print(f"  splatted points {start} - {end} / {n_points}")

    vol = vol_flat.reshape(Z, Y, X)

    # Preserve a comparable total contribution across sampling densities.
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
    Render one mesh using Gaussian splatting.

    Pipeline:
        mesh surface
        -> sampled surface points with mesh normals
        -> normal-oriented anisotropic Gaussian accumulation
        -> optional PSF convolution

    Returns:
        Rendered volume in ZYX order.
    """
    renderer_cfg = config.get("renderer", {})
    splat_cfg = config.get("splatting", {})

    labeling_mode = renderer_cfg.get("labeling_mode", "membrane")

    if labeling_mode != "membrane":
        raise NotImplementedError(
            "Gaussian splatting currently supports labeling_mode='membrane'."
        )

    spacing_nm = float(splat_cfg.get("spacing_nm", renderer_cfg.get("spacing_nm", 100)))
    sigma_tangent_nm = float(splat_cfg.get("sigma_tangent_nm", 100.0))
    sigma_normal_nm = float(splat_cfg.get("sigma_normal_nm", 50.0))
    apply_psf = bool(splat_cfg.get("apply_psf", True))
    seed = int(splat_cfg.get("seed", 0))
    points_per_batch = int(splat_cfg.get("points_per_batch", 50000))

    print(f"\n{'=' * 50}")
    print(f"Gaussian splatting: {tag}")
    print(f"Mesh: {mesh_path}")
    print(f"labeling_mode={labeling_mode}")
    print(f"spacing_nm={spacing_nm}")
    print(f"sigma_tangent_nm={sigma_tangent_nm}")
    print(f"sigma_normal_nm={sigma_normal_nm}")
    print(f"apply_psf={apply_psf}")
    print(f"{'=' * 50}")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    total_start = time.perf_counter()

    mesh_load_start = time.perf_counter()
    mesh = _load_cropped_mesh_to_grid(
        mesh_path=mesh_path,
        grid=grid,
        sigma_nm=max(sigma_tangent_nm, sigma_normal_nm),
    )
    mesh_load_time = time.perf_counter() - mesh_load_start
    print(f"[{tag}] mesh crop/load time: {mesh_load_time:.6f}s")

    if mesh is None:
        return torch.zeros(
            tuple(grid["shape_zyx"]),
            dtype=torch.float32,
            device=device,
        )

    sampling_start = time.perf_counter()

    points_xyz_nm, normals_xyz = _sample_surface_points(
        mesh,
        spacing_nm=spacing_nm,
        seed=seed,
    )

    del mesh
    gc.collect()

    sampling_time = time.perf_counter() - sampling_start
    print(f"[{tag}] sampling time  : {sampling_time:.6f}s")

    if device.type == "cuda":
        torch.cuda.synchronize()

    splatting_start = time.perf_counter()

    vol = _splat_points_to_volume(
        points_xyz_nm=points_xyz_nm,
        normals_xyz=normals_xyz,
        grid=grid,
        sigma_tangent_nm=sigma_tangent_nm,
        sigma_normal_nm=sigma_normal_nm,
        device=device,
        points_per_batch=points_per_batch,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    splatting_time = time.perf_counter() - splatting_start
    print(f"[{tag}] splatting time : {splatting_time:.6f}s")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    if apply_psf:
        print("Applying PSF convolution...")

        if device.type == "cuda":
            torch.cuda.synchronize()

        psf_start = time.perf_counter()
        vol = focal_stack_from_density(vol, psf_eff, device=device)

        if device.type == "cuda":
            torch.cuda.synchronize()

        psf_time = time.perf_counter() - psf_start
        print(f"[{tag}] PSF time       : {psf_time:.6f}s")

    if device.type == "cuda":
        torch.cuda.synchronize()

    print(
        f"[{tag}] total renderer time: "
        f"{time.perf_counter() - total_start:.6f}s"
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

        peak_allocated_mb = (
            torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)
        )
        peak_reserved_mb = (
            torch.cuda.max_memory_reserved(device=device) / (1024 ** 2)
        )

        print(
            f"[{tag}] peak allocated GPU memory: "
            f"{peak_allocated_mb:.1f} MB"
        )
        print(
            f"[{tag}] peak reserved GPU memory : "
            f"{peak_reserved_mb:.1f} MB"
        )

    return vol