import math
import time

import numpy as np
import torch
import torch.nn.functional as F
import trimesh


def get_device(device=None):
    if device is not None:
        return torch.device(device)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def as_torch(x, device=None, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)

    return torch.as_tensor(x, device=device, dtype=dtype)


def focal_stack_from_density(rho_zyx, psf_zyx, device=None):
    """
    Full 3D convolution in PyTorch.

    rho_zyx: (Z, Y, X)
    psf_zyx: (Zp, Yp, Xp)
    returns: volume (Z, Y, X)
    """
    device = get_device(device)

    rho_zyx = as_torch(rho_zyx, device=device, dtype=torch.float32)
    psf_zyx = as_torch(psf_zyx, device=device, dtype=torch.float32)

    if rho_zyx.ndim != 3 or psf_zyx.ndim != 3:
        raise ValueError("rho_zyx and psf_zyx must both be 3D")

    kernel = torch.flip(psf_zyx, dims=(0, 1, 2))

    inp = rho_zyx.unsqueeze(0).unsqueeze(0)
    ker = kernel.unsqueeze(0).unsqueeze(0)

    Zp, Yp, Xp = psf_zyx.shape

    out = F.conv3d(inp, ker, padding=(Zp // 2, Yp // 2, Xp // 2))

    return out[0, 0]


def _triangle_barycentric_grid(m, device):
    if m <= 0:
        return torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=device)

    rows = []

    for i in range(m + 1):
        j = torch.arange(m + 1 - i, device=device, dtype=torch.float32)
        ii = torch.full_like(j, float(i))
        rows.append(torch.stack([ii, j], dim=1))

    ij = torch.cat(rows, dim=0)

    a = ij[:, 0] / m
    b = ij[:, 1] / m
    c = 1.0 - a - b

    return torch.stack([a, b, c], dim=1)


def mesh_to_density_zyx(
    mesh_path,
    origin_nm,
    voxel_size_nm_xyz,
    shape_zyx,
    spacing_nm=200.0,
    device=None,
    batch_faces=2048,
):
    """
    Deterministic mesh surface to density grid rho[z,y,x].
    This is the membrane/surface-label renderer from the old pipeline.
    """
    device = get_device(device)

    Z, Y, X = shape_zyx
    sx, sy, sz = voxel_size_nm_xyz
    x0, y0, z0 = origin_nm

    mesh = trimesh.load(mesh_path, process=False)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )

    vertices = torch.as_tensor(
        np.asarray(mesh.vertices, dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )

    faces = torch.as_tensor(
        np.asarray(mesh.faces, dtype=np.int64),
        dtype=torch.long,
        device=device,
    )

    rho = torch.zeros((Z, Y, X), dtype=torch.float32, device=device)
    rho_flat = rho.view(-1)

    tris = vertices[faces]

    v0 = tris[:, 0, :]
    v1 = tris[:, 1, :]
    v2 = tris[:, 2, :]

    areas = 0.5 * torch.linalg.norm(torch.cross(v1 - v0, v2 - v0, dim=1), dim=1)

    valid_faces = areas > 0

    if not torch.any(valid_faces):
        return rho

    tris = tris[valid_faces]
    areas = areas[valid_faces]

    n_per_face = torch.clamp(
        torch.ceil(areas / (spacing_nm ** 2)).long(),
        min=1,
    )

    m_per_face = torch.ceil(torch.sqrt(n_per_face.float())).long()
    unique_m = torch.unique(m_per_face)

    for m_val in unique_m.tolist():
        sel = m_per_face == m_val
        tris_m = tris[sel]

        if tris_m.shape[0] == 0:
            continue

        bary = _triangle_barycentric_grid(int(m_val), device=device)

        for start in range(0, tris_m.shape[0], batch_faces):
            tri_batch = tris_m[start:start + batch_faces]

            vb0 = tri_batch[:, 0, :]
            vb1 = tri_batch[:, 1, :]
            vb2 = tri_batch[:, 2, :]

            pts = (
                bary[None, :, 0:1] * vb0[:, None, :]
                + bary[None, :, 1:2] * vb1[:, None, :]
                + bary[None, :, 2:3] * vb2[:, None, :]
            )

            pts = pts.reshape(-1, 3)

            ix = torch.floor((pts[:, 0] - x0) / sx).long()
            iy = torch.floor((pts[:, 1] - y0) / sy).long()
            iz = torch.floor((pts[:, 2] - z0) / sz).long()

            # Keep same convention as old code
            iy = (Y - 1) - iy

            valid = (
                (ix >= 0) & (ix < X)
                & (iy >= 0) & (iy < Y)
                & (iz >= 0) & (iz < Z)
            )

            if not torch.any(valid):
                continue

            ix = ix[valid]
            iy = iy[valid]
            iz = iz[valid]

            flat_idx = iz * (Y * X) + iy * X + ix

            vals = torch.ones(flat_idx.shape[0], dtype=torch.float32, device=device)

            rho_flat.scatter_add_(0, flat_idx, vals)

    return rho


def gaussian_kernel1d_torch(sigma, truncate=3.0, device=None, dtype=torch.float32):
    device = get_device(device)

    if sigma <= 0:
        return torch.tensor([1.0], dtype=dtype, device=device)

    radius = int(math.ceil(truncate * sigma))

    x = torch.arange(-radius, radius + 1, dtype=dtype, device=device)

    k = torch.exp(-(x ** 2) / (2 * sigma * sigma))
    k = k / k.sum()

    return k


def smooth_density_zyx(
    rho_zyx,
    sigma_zyx=(0.6, 0.8, 0.8),
    normalize_sum=True,
    device=None,
):
    device = get_device(device)

    rho_zyx = as_torch(rho_zyx, device=device, dtype=torch.float32)

    s0 = rho_zyx.sum()

    sz, sy, sx = sigma_zyx

    kz = gaussian_kernel1d_torch(sz, device=device)
    ky = gaussian_kernel1d_torch(sy, device=device)
    kx = gaussian_kernel1d_torch(sx, device=device)

    x = rho_zyx.unsqueeze(0).unsqueeze(0)

    wz = kz.view(1, 1, -1, 1, 1)
    x = F.conv3d(x, wz, padding=(kz.numel() // 2, 0, 0))

    wy = ky.view(1, 1, 1, -1, 1)
    x = F.conv3d(x, wy, padding=(0, ky.numel() // 2, 0))

    wx = kx.view(1, 1, 1, 1, -1)
    x = F.conv3d(x, wx, padding=(0, 0, kx.numel() // 2))

    rho_s = x[0, 0]

    if normalize_sum:
        s1 = rho_s.sum()
        if float(s1) > 0.0:
            rho_s = rho_s * (s0 / s1)

    return rho_s


def ensure_psf_odd_xy(psf_zyx, renormalize=False, device=None):
    device = get_device(device)

    psf_zyx = as_torch(psf_zyx, device=device, dtype=torch.float32)

    Z, Y, X = psf_zyx.shape

    pad_y = 1 if (Y % 2 == 0) else 0
    pad_x = 1 if (X % 2 == 0) else 0

    if pad_y or pad_x:
        psf_zyx = F.pad(
            psf_zyx,
            (0, pad_x, 0, pad_y, 0, 0),
            mode="constant",
            value=0.0,
        )

    if renormalize:
        s = psf_zyx.sum()
        if float(s) > 0.0:
            psf_zyx = psf_zyx / s

    return psf_zyx


def build_density_for_mesh(
    mesh_path,
    tag,
    labeling_mode,
    spacing_nm,
    origin_nm,
    voxel_size_nm_xyz,
    shape_zyx,
    device,
    batch_faces=2048,
    pseudofill_sigma_zyx=(2.0, 2.5, 2.5),
    density_smooth_sigma_zyx=(0.6, 0.8, 0.8),
    density_normalize_sum=True,
):
    print(f"\n{'=' * 50}")
    print(f"Building density: {tag}")
    print(f"Mesh: {mesh_path}")
    print(f"{'=' * 50}")

    t0 = time.time()

    if labeling_mode == "membrane":
        rho = mesh_to_density_zyx(
            mesh_path=mesh_path,
            origin_nm=origin_nm,
            voxel_size_nm_xyz=voxel_size_nm_xyz,
            shape_zyx=shape_zyx,
            spacing_nm=spacing_nm,
            device=device,
            batch_faces=batch_faces,
        )
    else:
        raise ValueError(
            "Only labeling_mode='membrane' is implemented in the clean single-mesh renderer."
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    print(
        f"[{tag}] density time: {time.time() - t0:.1f}s "
        f"sum={float(rho.sum()):.2f} max={float(rho.max()):.4f}"
    )

    t0 = time.time()

    rho = smooth_density_zyx(
        rho,
        sigma_zyx=density_smooth_sigma_zyx,
        normalize_sum=density_normalize_sum,
        device=device,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    print(
        f"[{tag}] smooth time : {time.time() - t0:.1f}s "
        f"sum={float(rho.sum()):.2f} max={float(rho.max()):.4f}"
    )

    return rho


def render_density(rho, psf_eff, tag, device):
    t0 = time.time()

    vol = focal_stack_from_density(rho, psf_eff, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    print(
        f"[{tag}] render time: {time.time() - t0:.1f}s "
        f"min={float(vol.min()):.4f} max={float(vol.max()):.4f}"
    )

    return vol


def render_single_mesh_voxel(
    mesh_path,
    grid,
    psf_eff,
    config,
    device,
):
    renderer_cfg = config["renderer"]

    rho = build_density_for_mesh(
        mesh_path=mesh_path,
        tag="single_mesh",
        labeling_mode=renderer_cfg.get("labeling_mode", "membrane"),
        spacing_nm=float(renderer_cfg.get("spacing_nm", 100)),
        origin_nm=grid["origin_nm"],
        voxel_size_nm_xyz=grid["voxel_size_nm_xyz"],
        shape_zyx=grid["shape_zyx"],
        device=device,
        batch_faces=int(renderer_cfg.get("batch_faces", 2048)),
        pseudofill_sigma_zyx=tuple(renderer_cfg.get("pseudofill_sigma_zyx", [2.0, 2.5, 2.5])),
        density_smooth_sigma_zyx=tuple(renderer_cfg.get("density_smooth_sigma_zyx", [0.6, 0.8, 0.8])),
        density_normalize_sum=bool(renderer_cfg.get("density_normalize_sum", True)),
    )

    vol = render_density(rho, psf_eff, tag="single_mesh", device=device)

    del rho

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return vol