"""
Voxel-grid renderer.

This renderer converts mesh geometry into a voxel density volume and then
applies PSF convolution to create a synthetic microscopy stack.

Pipeline:
    mesh surface
    -> voxel density volume
    -> optional enclosed-volume filling
    -> PSF convolution
    -> rendered image volume

Coordinate convention:
    Mesh coordinates are XYZ in nanometres.
    Output volumes are ZYX:
        [Z slices, Y pixels, X pixels]

Supported label modes:
    membrane:
        Surface-area-based density rasterized from mesh faces.
    filled:
        Filled occupancy derived from the voxelized mesh surface. Small
        voxel-scale gaps can be closed before enclosed regions are filled.
"""

import gc
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from scipy.ndimage import binary_closing, binary_fill_holes


def get_device(device=None):
    """
    Return a Torch device.

    If a device is provided, use it. Otherwise use CUDA when available,
    falling back to CPU.
    """
    if device is not None:
        return torch.device(device)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def as_torch(x, device=None, dtype=torch.float32):
    """
    Convert input to a Torch tensor on the selected device and dtype.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)

    return torch.as_tensor(x, device=device, dtype=dtype)


def focal_stack_from_density(rho_zyx, psf_zyx, device=None):
    """
    Apply 3D PSF convolution to a density volume.

    Args:
        rho_zyx:
            Input density volume in ZYX order.
        psf_zyx:
            PSF kernel in ZYX order.
        device:
            Torch device.

    Returns:
        Rendered volume in ZYX order.
    """
    device = get_device(device)

    rho_zyx = as_torch(rho_zyx, device=device, dtype=torch.float32)
    psf_zyx = as_torch(psf_zyx, device=device, dtype=torch.float32)

    if rho_zyx.ndim != 3 or psf_zyx.ndim != 3:
        raise ValueError("rho_zyx and psf_zyx must both be 3D")

    # PyTorch conv3d performs cross-correlation, so flip the PSF to obtain
    # convolution behaviour.
    kernel = torch.flip(psf_zyx, dims=(0, 1, 2))

    inp = rho_zyx.unsqueeze(0).unsqueeze(0)
    ker = kernel.unsqueeze(0).unsqueeze(0)

    Zp, Yp, Xp = psf_zyx.shape

    out = F.conv3d(inp, ker, padding=(Zp // 2, Yp // 2, Xp // 2))

    return out[0, 0]


def _triangle_surface_samples(tri, spacing_nm, device):
    """
    Create deterministic samples over one triangle.

    Samples distribute the triangle's physical membrane area across the voxel
    grid. Their weights sum to the physical area of the triangle.
    """
    v0 = tri[0]
    v1 = tri[1]
    v2 = tri[2]

    area = 0.5 * torch.linalg.norm(
        torch.cross(v1 - v0, v2 - v0, dim=0)
    )

    if float(area) <= 0.0:
        return None, None

    n_samples = max(
        1,
        int(torch.ceil(area / (spacing_nm ** 2)).item()),
    )

    sample_index = torch.arange(
        n_samples,
        dtype=torch.float32,
        device=device,
    )

    # Deterministic low-discrepancy samples over triangle area.
    u = (sample_index + 0.5) / float(n_samples)
    v = torch.frac(
        (sample_index + 0.5) * 0.6180339887498949
    )

    sqrt_u = torch.sqrt(u)
    bary0 = 1.0 - sqrt_u
    bary1 = sqrt_u * (1.0 - v)
    bary2 = sqrt_u * v

    points = (
        bary0[:, None] * v0[None, :]
        + bary1[:, None] * v1[None, :]
        + bary2[:, None] * v2[None, :]
    )

    weights = torch.full(
        (n_samples,),
        float(area.item()) / float(n_samples),
        dtype=torch.float32,
        device=device,
    )

    return points, weights

def _ply_scalar_dtype(name, endian="<"):
    """Return a NumPy dtype for a scalar PLY property type."""
    mapping = {
        "char": "i1", "int8": "i1",
        "uchar": "u1", "uint8": "u1",
        "short": "i2", "int16": "i2",
        "ushort": "u2", "uint16": "u2",
        "int": "i4", "int32": "i4",
        "uint": "u4", "uint32": "u4",
        "float": "f4", "float32": "f4",
        "double": "f8", "float64": "f8",
    }
    if name not in mapping:
        raise ValueError(f"Unsupported PLY scalar type: {name}")
    code = mapping[name]
    return np.dtype(code if code.endswith("1") else endian + code)


def _read_binary_ply_header(mesh_path):
    """Read enough PLY metadata for memory-efficient triangle streaming."""
    with open(mesh_path, "rb") as file:
        first = file.readline().decode("ascii", errors="strict").strip()
        if first != "ply":
            raise ValueError("Not a PLY file")

        fmt = None
        elements = []
        current = None

        while True:
            raw = file.readline()
            if not raw:
                raise ValueError("Unexpected end of PLY header")

            line = raw.decode("ascii", errors="strict").strip()
            if not line or line.startswith("comment") or line.startswith("obj_info"):
                continue
            if line == "end_header":
                data_offset = file.tell()
                break

            parts = line.split()
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                current = {
                    "name": parts[1],
                    "count": int(parts[2]),
                    "properties": [],
                }
                elements.append(current)
            elif parts[0] == "property" and current is not None:
                if parts[1] == "list":
                    current["properties"].append(
                        ("list", parts[2], parts[3], parts[4])
                    )
                else:
                    current["properties"].append(
                        ("scalar", parts[1], parts[2])
                    )

    return fmt, elements, data_offset


def _load_roi_geometry_from_binary_ply(
    mesh_path,
    roi_min,
    roi_max,
    face_filter_batch,
):
    """Stream a binary triangle PLY and return only geometry near the ROI.

    Vertex data are memory-mapped and face records are read in chunks. This
    avoids constructing a full Trimesh object for very large source meshes.
    """
    fmt, elements, data_offset = _read_binary_ply_header(mesh_path)

    if fmt not in ("binary_little_endian", "binary_big_endian"):
        raise ValueError("Memory-efficient PLY path requires binary PLY")

    endian = "<" if fmt == "binary_little_endian" else ">"
    vertex_el = next((e for e in elements if e["name"] == "vertex"), None)
    face_el = next((e for e in elements if e["name"] == "face"), None)
    if vertex_el is None or face_el is None:
        raise ValueError("PLY must contain vertex and face elements")

    # This fast path supports ordinary triangle PLY files: scalar vertex
    # properties followed by one face list property containing vertex indices.
    vertex_fields = []
    for prop in vertex_el["properties"]:
        if prop[0] != "scalar":
            raise ValueError("List properties on PLY vertices are unsupported")
        _, type_name, prop_name = prop
        vertex_fields.append((prop_name, _ply_scalar_dtype(type_name, endian)))

    vertex_dtype = np.dtype(vertex_fields)
    for axis in ("x", "y", "z"):
        if axis not in vertex_dtype.names:
            raise ValueError(f"PLY vertex property '{axis}' is missing")

    face_props = face_el["properties"]
    if len(face_props) != 1 or face_props[0][0] != "list":
        raise ValueError(
            "Memory-efficient PLY path expects one face vertex-index list property"
        )

    _, count_type, index_type, _ = face_props[0]
    count_dtype = _ply_scalar_dtype(count_type, endian)
    index_dtype = _ply_scalar_dtype(index_type, endian)

    # For triangle meshes every face record is: count (=3) + three indices.
    face_dtype = np.dtype([
        ("count", count_dtype),
        ("indices", index_dtype, (3,)),
    ])

    vertex_count = int(vertex_el["count"])
    face_count = int(face_el["count"])
    vertex_bytes = vertex_count * vertex_dtype.itemsize
    face_offset = data_offset + vertex_bytes

    vertices = np.memmap(
        mesh_path,
        mode="r",
        dtype=vertex_dtype,
        offset=data_offset,
        shape=(vertex_count,),
    )

    selected_face_chunks = []

    with open(mesh_path, "rb") as file:
        file.seek(face_offset)
        remaining = face_count

        while remaining > 0:
            n = min(face_filter_batch, remaining)
            records = np.fromfile(file, dtype=face_dtype, count=n)
            if len(records) != n:
                raise ValueError("Unexpected end of PLY face data")
            if np.any(records["count"] != 3):
                raise ValueError(
                    "PLY contains non-triangle faces; fast ROI loader cannot stream it"
                )

            face_chunk = np.asarray(records["indices"], dtype=np.int64)
            ids = face_chunk.reshape(-1)

            # Gather only vertices referenced by this face chunk. No full Nx3
            # floating-point vertex array is ever created.
            tri_chunk = np.empty((n, 3, 3), dtype=np.float32)
            tri_chunk[..., 0] = np.asarray(vertices["x"][ids]).reshape(n, 3)
            tri_chunk[..., 1] = np.asarray(vertices["y"][ids]).reshape(n, 3)
            tri_chunk[..., 2] = np.asarray(vertices["z"][ids]).reshape(n, 3)

            tri_min = tri_chunk.min(axis=1)
            tri_max = tri_chunk.max(axis=1)
            keep = np.all(
                (tri_max >= roi_min[None, :])
                & (tri_min <= roi_max[None, :]),
                axis=1,
            )

            if np.any(keep):
                selected_face_chunks.append(face_chunk[keep].copy())

            remaining -= n

    if not selected_face_chunks:
        del vertices
        return None, None, face_count

    selected_faces = np.concatenate(selected_face_chunks, axis=0)
    used_vertex_ids, inverse = np.unique(
        selected_faces.reshape(-1),
        return_inverse=True,
    )

    selected_vertices = np.empty((len(used_vertex_ids), 3), dtype=np.float32)
    selected_vertices[:, 0] = np.asarray(vertices["x"][used_vertex_ids], dtype=np.float32)
    selected_vertices[:, 1] = np.asarray(vertices["y"][used_vertex_ids], dtype=np.float32)
    selected_vertices[:, 2] = np.asarray(vertices["z"][used_vertex_ids], dtype=np.float32)
    compact_faces = inverse.reshape(-1, 3).astype(np.int64, copy=False)

    del vertices, selected_face_chunks, selected_faces, used_vertex_ids, inverse
    gc.collect()

    return selected_vertices, compact_faces, face_count


def _load_and_crop_mesh_geometry(
    mesh_path,
    roi_min,
    roi_max,
    face_filter_batch,
):
    """Load only geometry intersecting the render ROI when possible."""
    if str(mesh_path).lower().endswith(".ply"):
        try:
            return _load_roi_geometry_from_binary_ply(
                mesh_path,
                roi_min,
                roi_max,
                face_filter_batch,
            )
        except (ValueError, OSError) as exc:
            print(
                "Memory-efficient PLY loader unavailable for this file; "
                f"falling back to trimesh ({exc})."
            )

    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        parts = [
            g for g in mesh.geometry.values()
            if isinstance(g, trimesh.Trimesh)
        ]
        if not parts:
            raise ValueError(f"No mesh geometry found in scene: {mesh_path}")
        mesh = parts[0] if len(parts) == 1 else trimesh.util.concatenate(parts)

    vertices_np = np.asarray(mesh.vertices)
    faces_np = np.asarray(mesh.faces)
    selected_face_chunks = []

    for start in range(0, len(faces_np), face_filter_batch):
        face_chunk = faces_np[start:start + face_filter_batch]
        tri_chunk = vertices_np[face_chunk]
        tri_min = tri_chunk.min(axis=1)
        tri_max = tri_chunk.max(axis=1)
        keep = np.all(
            (tri_max >= roi_min[None, :])
            & (tri_min <= roi_max[None, :]),
            axis=1,
        )
        if np.any(keep):
            selected_face_chunks.append(face_chunk[keep].copy())

    total_faces = len(faces_np)
    if not selected_face_chunks:
        del mesh, vertices_np, faces_np
        gc.collect()
        return None, None, total_faces

    selected_faces = np.concatenate(selected_face_chunks, axis=0)
    used_vertex_ids, inverse = np.unique(
        selected_faces.reshape(-1), return_inverse=True
    )
    selected_vertices = np.asarray(
        vertices_np[used_vertex_ids], dtype=np.float32
    )
    compact_faces = inverse.reshape(-1, 3).astype(np.int64, copy=False)

    del mesh, vertices_np, faces_np, selected_face_chunks
    del selected_faces, used_vertex_ids, inverse
    gc.collect()
    return selected_vertices, compact_faces, total_faces


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
    Rasterize mesh triangles into a ZYX membrane-density volume.

    Each valid triangle contributes its physical surface area to the voxel
    grid. Surface samples determine where that area is deposited, so total
    membrane density is not determined by mesh vertex count.
    """
    device = get_device(device)

    Z, Y, X = shape_zyx
    sx, sy, sz = voxel_size_nm_xyz
    x0, y0, z0 = origin_nm

    x1 = x0 + X * sx
    y1 = y0 + Y * sy
    z1 = z0 + Z * sz

    roi_min = np.array([x0 - sx, y0 - sy, z0 - sz], dtype=np.float64)
    roi_max = np.array([x1 + sx, y1 + sy, z1 + sz], dtype=np.float64)
    face_filter_batch = max(int(batch_faces) * 32, 65536)

    selected_vertices_np, compact_faces_np, total_faces = (
        _load_and_crop_mesh_geometry(
            mesh_path,
            roi_min,
            roi_max,
            face_filter_batch,
        )
    )

    if selected_vertices_np is None:
        print(
            f"ROI face filter: kept 0 / {total_faces} faces "
            "(no mesh surface intersects the render grid)"
        )
        return torch.zeros(
            (Z, Y, X), dtype=torch.float32, device=device
        )

    kept_faces = len(compact_faces_np)
    print(
        f"ROI face filter: kept {kept_faces} / {total_faces} faces "
        f"({100.0 * kept_faces / total_faces:.4f}%)"
    )

    vertices = torch.as_tensor(
        selected_vertices_np,
        dtype=torch.float32,
        device=device,
    )
    faces = torch.as_tensor(
        compact_faces_np,
        dtype=torch.long,
        device=device,
    )
    del selected_vertices_np, compact_faces_np

    rho = torch.zeros(
        (Z, Y, X),
        dtype=torch.float32,
        device=device,
    )
    rho_flat = rho.view(-1)

    tris = vertices[faces]
    del vertices, faces

    for start in range(0, tris.shape[0], batch_faces):
        tri_batch = tris[start:start + batch_faces]

        for tri in tri_batch:
            points, weights = _triangle_surface_samples(
                tri,
                spacing_nm=spacing_nm,
                device=device,
            )

            if points is None:
                continue

            ix = torch.floor(
                (points[:, 0] - x0) / sx
            ).long()
            iy_source = torch.floor(
                (points[:, 1] - y0) / sy
            ).long()
            iz = torch.floor(
                (points[:, 2] - z0) / sz
            ).long()

            iy = (Y - 1) - iy_source

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
            weights_valid = weights[valid]

            flat_idx = iz * (Y * X) + iy * X + ix

            rho_flat.scatter_add_(
                0,
                flat_idx,
                weights_valid,
            )

    return rho


def mesh_to_filled_density_zyx(
    mesh_path,
    origin_nm,
    voxel_size_nm_xyz,
    shape_zyx,
    spacing_nm=200.0,
    device=None,
    batch_faces=2048,
    closing_iterations=1,
):
    """
    Create a filled occupancy volume from the rasterized mesh surface.

    The mesh surface is first rasterized into the voxel grid. A controlled
    binary closing can bridge small voxel-scale gaps, after which enclosed
    regions are filled. This is intended for meshes that may contain small
    discretization gaps; large open boundaries are not repaired automatically.
    """
    device = get_device(device)

    surface_density = mesh_to_density_zyx(
        mesh_path=mesh_path,
        origin_nm=origin_nm,
        voxel_size_nm_xyz=voxel_size_nm_xyz,
        shape_zyx=shape_zyx,
        spacing_nm=spacing_nm,
        device=device,
        batch_faces=batch_faces,
    )

    surface = (
        surface_density.detach().cpu().numpy() > 0
    )

    if int(closing_iterations) > 0:
        structure = np.ones((3, 3, 3), dtype=bool)
        surface = binary_closing(
            surface,
            structure=structure,
            iterations=int(closing_iterations),
        )

    filled = binary_fill_holes(surface)
    interior = np.logical_and(filled, np.logical_not(surface))

    # Include both the enclosed interior and its boundary so the filled
    # object represents the complete occupied neuronal volume.
    filled_volume = np.logical_or(interior, surface).astype(np.float32)

    surface_voxels = int(surface.sum())
    filled_voxels = int(filled_volume.sum())

    print(
        f"Filled voxelization: surface={surface_voxels} voxels, "
        f"filled={filled_voxels} voxels"
    )

    if filled_voxels <= surface_voxels:
        print(
            "Warning: no enclosed interior was recovered. "
            "The mesh may contain an opening larger than the configured "
            "voxel-scale closing operation."
        )

    return torch.as_tensor(
        filled_volume,
        dtype=torch.float32,
        device=device,
    )

def ensure_psf_odd_xy(psf_zyx, renormalize=False, device=None):
    """
    Ensure PSF has odd Y and X dimensions.

    Odd XY dimensions make the PSF centre well defined for convolution.
    """
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
    filled_closing_iterations=1,
):
    """
    Build fluorescence density for one mesh.

    membrane:
        Physical mesh surface area is rasterized into the voxel grid.

    filled:
        The rasterized surface is closed at voxel scale and enclosed interior
        voxels are filled to create a filled occupancy volume.
    """
    print(f"\n{'=' * 50}")
    print(f"Building density: {tag}")
    print(f"Mesh: {mesh_path}")
    print(f"Labeling mode: {labeling_mode}")
    print(f"{'=' * 50}")

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()

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

    elif labeling_mode == "filled":
        rho = mesh_to_filled_density_zyx(
            mesh_path=mesh_path,
            origin_nm=origin_nm,
            voxel_size_nm_xyz=voxel_size_nm_xyz,
            shape_zyx=shape_zyx,
            spacing_nm=spacing_nm,
            device=device,
            batch_faces=batch_faces,
            closing_iterations=filled_closing_iterations,
        )

    else:
        raise ValueError(
            "labeling_mode must be 'membrane' or 'filled'"
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    print(
        f"[{tag}] density time: {time.perf_counter() - t0:.6f}s "
        f"sum={float(rho.sum()):.2f} max={float(rho.max()):.4f}"
    )

    return rho

def render_density(rho, psf_eff, tag, device):
    """
    Render a density volume by applying PSF convolution.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    vol = focal_stack_from_density(rho, psf_eff, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    print(
        f"[{tag}] render time: {time.perf_counter() - t0:.6f}s "
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
    """
    Render a single mesh with the voxel-grid renderer.

    The renderer configuration is read from config["renderer"]["voxel_grid"].

    Returns:
        Rendered image volume in ZYX order.
    """
    renderer_cfg = config["renderer"].get(
        "voxel_grid",
        {},
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    total_start = time.perf_counter()

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
        filled_closing_iterations=int(
            renderer_cfg.get("filled_closing_iterations", 1)
        ),
    )

    vol = render_density(rho, psf_eff, tag="single_mesh", device=device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    print(
        f"[single_mesh] total renderer time: "
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
            f"[single_mesh] peak allocated GPU memory: "
            f"{peak_allocated_mb:.1f} MB"
        )
        print(
            f"[single_mesh] peak reserved GPU memory : "
            f"{peak_reserved_mb:.1f} MB"
        )

    del rho

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return vol