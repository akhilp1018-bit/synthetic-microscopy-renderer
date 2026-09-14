"""
Mesh loading and preparation utilities.

This module handles:
    - loading PLY/mesh files with trimesh
    - optional coordinate scaling to nanometres
    - optional recentering
    - bounding-box extraction
    - locating labelled dendrite/spine component meshes
    - optional transformation into a mouse-selected ROI coordinate system

Coordinate convention:
    Mesh coordinates are expected as XYZ.
    After preparation, coordinates are interpreted in nanometres.
"""

from pathlib import Path
import gc
import tempfile

import numpy as np
import trimesh


def load_mesh(mesh_path: str | Path) -> trimesh.Trimesh:
    """Load a mesh file as a trimesh.Trimesh object."""
    mesh_path = Path(mesh_path)

    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    loaded = trimesh.load(mesh_path, process=False)

    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    elif isinstance(loaded, trimesh.Scene):
        parts = [
            geom for geom in loaded.geometry.values()
            if isinstance(geom, trimesh.Trimesh)
        ]
        if len(parts) == 0:
            raise ValueError(f"No mesh geometry found in scene: {mesh_path}")
        mesh = parts[0] if len(parts) == 1 else trimesh.util.concatenate(parts)
    else:
        raise TypeError(f"Loaded object is not a Trimesh or Scene: {type(loaded)}")

    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError(f"Mesh has no vertices: {mesh_path}")
    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no faces: {mesh_path}")

    return mesh


def _validate_transform(world_to_local_4x4):
    """Validate and return a homogeneous 4x4 transform matrix."""
    if world_to_local_4x4 is None:
        return None

    matrix = np.asarray(world_to_local_4x4, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("world_to_local_4x4 must be a 4x4 matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("world_to_local_4x4 contains non-finite values")
    return matrix


def _ply_scalar_dtype(name, endian="<"):
    mapping = {
        "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
        "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
        "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
        "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
    }
    if name not in mapping:
        raise ValueError(f"Unsupported PLY scalar type: {name}")
    code = mapping[name]
    return np.dtype(code if code.endswith("1") else endian + code)


def _read_binary_ply_header(mesh_path):
    """Read PLY metadata needed for memory-efficient triangle streaming."""
    with open(mesh_path, "rb") as file:
        if file.readline().decode("ascii", errors="strict").strip() != "ply":
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
                current = {"name": parts[1], "count": int(parts[2]), "properties": []}
                elements.append(current)
            elif parts[0] == "property" and current is not None:
                if parts[1] == "list":
                    current["properties"].append(("list", parts[2], parts[3], parts[4]))
                else:
                    current["properties"].append(("scalar", parts[1], parts[2]))

    return fmt, elements, data_offset


def _transform_points(points_xyz, matrix):
    """Apply a homogeneous transform to an Nx3 array without creating Nx4 data."""
    return points_xyz @ matrix[:3, :3].T + matrix[:3, 3]


def _prepare_oriented_roi_binary_ply(
    mesh_path,
    matrix,
    local_bounds_xyz_nm,
    scale_to_nm=1.0,
    face_filter_batch=65536,
):
    """
    Crop a large binary triangle PLY to an oriented ROI before transformation.

    Faces are streamed from disk. Only triangles intersecting the ROI after
    transformation into ROI-local coordinates are retained. This avoids
    loading, transforming, and exporting the complete source mesh.
    """
    fmt, elements, data_offset = _read_binary_ply_header(mesh_path)
    if fmt not in ("binary_little_endian", "binary_big_endian"):
        raise ValueError("Memory-efficient oriented ROI preparation requires binary PLY")

    endian = "<" if fmt == "binary_little_endian" else ">"
    vertex_el = next((e for e in elements if e["name"] == "vertex"), None)
    face_el = next((e for e in elements if e["name"] == "face"), None)
    if vertex_el is None or face_el is None:
        raise ValueError("PLY must contain vertex and face elements")

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
        raise ValueError("Expected one face vertex-index list property")

    _, count_type, index_type, _ = face_props[0]
    count_dtype = _ply_scalar_dtype(count_type, endian)
    index_dtype = _ply_scalar_dtype(index_type, endian)
    face_dtype = np.dtype([
        ("count", count_dtype),
        ("indices", index_dtype, (3,)),
    ])

    vertex_count = int(vertex_el["count"])
    face_count = int(face_el["count"])
    face_offset = data_offset + vertex_count * vertex_dtype.itemsize

    vertices = np.memmap(
        mesh_path,
        mode="r",
        dtype=vertex_dtype,
        offset=data_offset,
        shape=(vertex_count,),
    )

    xmin, xmax, ymin, ymax, zmin, zmax = [float(v) for v in local_bounds_xyz_nm]
    roi_min = np.array([xmin, ymin, zmin], dtype=np.float64)
    roi_max = np.array([xmax, ymax, zmax], dtype=np.float64)

    selected_face_chunks = []

    with open(mesh_path, "rb") as file:
        file.seek(face_offset)
        remaining = face_count

        while remaining > 0:
            n = min(int(face_filter_batch), remaining)
            records = np.fromfile(file, dtype=face_dtype, count=n)
            if len(records) != n:
                raise ValueError("Unexpected end of PLY face data")
            if np.any(records["count"] != 3):
                raise ValueError("PLY contains non-triangle faces")

            face_chunk = np.asarray(records["indices"], dtype=np.int64)
            ids = face_chunk.reshape(-1)

            tri_world = np.empty((n, 3, 3), dtype=np.float64)
            tri_world[..., 0] = np.asarray(vertices["x"][ids]).reshape(n, 3)
            tri_world[..., 1] = np.asarray(vertices["y"][ids]).reshape(n, 3)
            tri_world[..., 2] = np.asarray(vertices["z"][ids]).reshape(n, 3)
            tri_world *= float(scale_to_nm)

            tri_local = _transform_points(tri_world.reshape(-1, 3), matrix).reshape(n, 3, 3)
            tri_min = tri_local.min(axis=1)
            tri_max = tri_local.max(axis=1)

            keep = np.all(
                (tri_max >= roi_min[None, :]) &
                (tri_min <= roi_max[None, :]),
                axis=1,
            )
            if np.any(keep):
                selected_face_chunks.append(face_chunk[keep].copy())

            remaining -= n

    if not selected_face_chunks:
        del vertices
        gc.collect()
        raise ValueError("No mesh faces intersect the selected oriented rendering ROI")

    selected_faces = np.concatenate(selected_face_chunks, axis=0)
    used_vertex_ids, inverse = np.unique(selected_faces.reshape(-1), return_inverse=True)

    selected_vertices = np.empty((len(used_vertex_ids), 3), dtype=np.float64)
    selected_vertices[:, 0] = np.asarray(vertices["x"][used_vertex_ids], dtype=np.float64)
    selected_vertices[:, 1] = np.asarray(vertices["y"][used_vertex_ids], dtype=np.float64)
    selected_vertices[:, 2] = np.asarray(vertices["z"][used_vertex_ids], dtype=np.float64)
    selected_vertices *= float(scale_to_nm)
    selected_vertices = _transform_points(selected_vertices, matrix)

    compact_faces = inverse.reshape(-1, 3).astype(np.int64, copy=False)
    kept_faces = len(compact_faces)

    cropped = trimesh.Trimesh(
        vertices=selected_vertices,
        faces=compact_faces,
        process=False,
    )

    tmp = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
    tmp_path = tmp.name
    tmp.close()
    cropped.export(tmp_path)

    print(
        f"Oriented ROI pre-crop: kept {kept_faces} / {face_count} faces "
        f"({100.0 * kept_faces / face_count:.4f}%)",
        flush=True,
    )

    del vertices, selected_face_chunks, selected_faces, used_vertex_ids
    del inverse, selected_vertices, compact_faces, cropped
    gc.collect()

    return tmp_path


def prepare_mesh_for_sim(
    mesh_path: str | Path,
    scale_to_nm: float = 1.0,
    recenter: bool = False,
    world_to_local_4x4=None,
    roi_local_bounds_xyz_nm=None,
) -> str:
    """Prepare one mesh for simulation.

    When an oriented ROI and its local bounds are supplied for a binary PLY,
    the mesh is cropped while streaming from disk before the retained geometry
    is transformed and exported. This keeps large-mesh ROI rendering memory
    efficient.
    """
    mesh_path = Path(mesh_path)
    matrix = _validate_transform(world_to_local_4x4)

    need_preprocess = (
        float(scale_to_nm) != 1.0
        or bool(recenter)
        or matrix is not None
    )
    if not need_preprocess:
        return str(mesh_path)

    # GUI-oriented ROI path. Recentring changes the coordinate system using a
    # global vertex mean, so retain the general fallback when recenter=True.
    if (
        matrix is not None
        and roi_local_bounds_xyz_nm is not None
        and not recenter
        and mesh_path.suffix.lower() == ".ply"
    ):
        try:
            tmp_path = _prepare_oriented_roi_binary_ply(
                mesh_path=mesh_path,
                matrix=matrix,
                local_bounds_xyz_nm=roi_local_bounds_xyz_nm,
                scale_to_nm=scale_to_nm,
            )
            print(f"Prepared mesh: {mesh_path}", flush=True)
            print(f"  temp path     : {tmp_path}", flush=True)
            print(f"  scale_to_nm   : {scale_to_nm}", flush=True)
            print(f"  recenter      : {recenter}", flush=True)
            print("  ROI transform : True (memory-safe local crop)", flush=True)
            return tmp_path
        except (ValueError, OSError) as exc:
            print(
                "Memory-safe oriented ROI preparation unavailable; "
                f"falling back to full mesh preparation ({exc}).",
                flush=True,
            )

    mesh = load_mesh(mesh_path)
    vertices = mesh.vertices.astype(np.float64) * float(scale_to_nm)

    if recenter:
        vertices = vertices - vertices.mean(axis=0, keepdims=True)

    mesh.vertices = vertices
    if matrix is not None:
        mesh.apply_transform(matrix)

    tmp = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
    tmp_path = tmp.name
    tmp.close()
    mesh.export(tmp_path)

    del mesh, vertices
    gc.collect()

    print(f"Prepared mesh: {mesh_path}", flush=True)
    print(f"  temp path     : {tmp_path}", flush=True)
    print(f"  scale_to_nm   : {scale_to_nm}", flush=True)
    print(f"  recenter      : {recenter}", flush=True)
    print(f"  ROI transform : {matrix is not None}", flush=True)
    return tmp_path


def transform_mesh_to_local_coordinates(mesh_path: str | Path, world_to_local_4x4) -> str:
    return prepare_mesh_for_sim(
        mesh_path=mesh_path,
        scale_to_nm=1.0,
        recenter=False,
        world_to_local_4x4=world_to_local_4x4,
    )


def get_mesh_anchor_nm(mesh_path, scale_to_nm=1.0, recenter=False):
    """Return the mesh vertex nearest to the mean vertex position."""
    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            raise ValueError(f"No geometry found in mesh: {mesh_path}")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    vertices = np.asarray(mesh.vertices)
    if vertices.size == 0:
        raise ValueError(f"Mesh contains no vertices: {mesh_path}")

    mean_xyz = vertices.mean(axis=0, dtype=np.float64)
    best_index = None
    best_distance_sq = np.inf
    chunk_size = 250_000

    for start in range(0, len(vertices), chunk_size):
        end = min(start + chunk_size, len(vertices))
        chunk = vertices[start:end]
        diff = chunk.astype(np.float64, copy=False) - mean_xyz
        distances_sq = np.einsum("ij,ij->i", diff, diff)
        local_index = int(np.argmin(distances_sq))
        local_distance = float(distances_sq[local_index])
        if local_distance < best_distance_sq:
            best_distance_sq = local_distance
            best_index = start + local_index

    anchor = vertices[best_index].astype(np.float64) * float(scale_to_nm)
    return tuple(float(v) for v in anchor)


def load_bbox_nm(mesh_path: str | Path) -> tuple[float, float, float, float, float, float]:
    mesh = load_mesh(mesh_path)
    bounds = mesh.bounds
    return (
        float(bounds[0, 0]), float(bounds[0, 1]), float(bounds[0, 2]),
        float(bounds[1, 0]), float(bounds[1, 1]), float(bounds[1, 2]),
    )


def get_combined_bbox_nm(mesh_paths: list[str | Path]) -> dict:
    bboxes = [load_bbox_nm(p) for p in mesh_paths]
    return {
        "xmin": min(b[0] for b in bboxes),
        "ymin": min(b[1] for b in bboxes),
        "zmin": min(b[2] for b in bboxes),
        "xmax": max(b[3] for b in bboxes),
        "ymax": max(b[4] for b in bboxes),
        "zmax": max(b[5] for b in bboxes),
    }


def find_labelled_component_paths(
    labelled_dir: str | Path,
    dendrite_pattern: str = "dendrite*.ply",
    spine_pattern: str = "spine*.ply",
) -> tuple[Path, list[Path]]:
    labelled_dir = Path(labelled_dir)
    if not labelled_dir.exists():
        raise FileNotFoundError(f"Labelled mesh folder not found: {labelled_dir}")

    dendrite_paths = sorted(labelled_dir.glob(dendrite_pattern))
    spine_paths = sorted(labelled_dir.glob(spine_pattern))

    if len(dendrite_paths) == 0:
        raise FileNotFoundError(
            f"No dendrite mesh found in {labelled_dir} with pattern {dendrite_pattern}"
        )
    if len(dendrite_paths) > 1:
        print("Warning: multiple dendrite meshes found. Using first one:")
        for p in dendrite_paths:
            print(f"  {p}")
    if len(spine_paths) == 0:
        raise FileNotFoundError(
            f"No spine meshes found in {labelled_dir} with pattern {spine_pattern}"
        )

    dendrite_path = dendrite_paths[0]
    print("\nLabelled components:")
    print(f"  Dendrite : {dendrite_path}")
    print(f"  Spines   : {len(spine_paths)} found")
    print(f"  First few: {spine_paths[:3]}")
    return dendrite_path, spine_paths


def prepare_labelled_components_for_sim(
    dendrite_path: str | Path,
    spine_paths: list[str | Path],
    scale_to_nm: float = 1.0,
    recenter: bool = False,
    world_to_local_4x4=None,
    roi_local_bounds_xyz_nm=None,
) -> tuple[str, list[str]]:
    """Prepare dendrite and spine meshes with the same ROI transform."""
    sim_dendrite_path = prepare_mesh_for_sim(
        dendrite_path,
        scale_to_nm=scale_to_nm,
        recenter=recenter,
        world_to_local_4x4=world_to_local_4x4,
        roi_local_bounds_xyz_nm=roi_local_bounds_xyz_nm,
    )

    sim_spine_paths = [
        prepare_mesh_for_sim(
            p,
            scale_to_nm=scale_to_nm,
            recenter=recenter,
            world_to_local_4x4=world_to_local_4x4,
            roi_local_bounds_xyz_nm=roi_local_bounds_xyz_nm,
        )
        for p in spine_paths
    ]
    return sim_dendrite_path, sim_spine_paths
