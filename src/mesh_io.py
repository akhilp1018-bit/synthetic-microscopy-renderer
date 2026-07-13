from pathlib import Path
import tempfile
import numpy as np
import trimesh


def load_mesh(mesh_path: str | Path) -> trimesh.Trimesh:
    mesh_path = Path(mesh_path)

    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    mesh = trimesh.load(mesh_path, force="mesh", process=False)

    if isinstance(mesh, trimesh.Scene):
        parts = [
            geom for geom in mesh.geometry.values()
            if isinstance(geom, trimesh.Trimesh)
        ]

        if len(parts) == 0:
            raise ValueError(f"No mesh geometry found in scene: {mesh_path}")

        mesh = trimesh.util.concatenate(parts)

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Loaded object is not a Trimesh: {type(mesh)}")

    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError(f"Mesh has no vertices: {mesh_path}")

    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no faces: {mesh_path}")

    return mesh


def prepare_mesh_for_sim(
    mesh_path: str | Path,
    scale_to_nm: float = 1.0,
    recenter: bool = False,
) -> str:
    """
    Prepare one mesh for simulation.

    If no scaling/recentering is needed, returns the original path.
    If scaling/recentering is needed, writes a temporary PLY and returns that path.
    """
    mesh_path = Path(mesh_path)

    need_preprocess = (float(scale_to_nm) != 1.0) or bool(recenter)

    if not need_preprocess:
        return str(mesh_path)

    mesh = load_mesh(mesh_path)

    vertices = mesh.vertices.astype(np.float64) * float(scale_to_nm)

    if recenter:
        vertices = vertices - vertices.mean(axis=0, keepdims=True)

    mesh.vertices = vertices

    tmp = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
    tmp_path = tmp.name
    tmp.close()

    mesh.export(tmp_path)

    print(f"Prepared mesh: {mesh_path}")
    print(f"  temp path   : {tmp_path}")
    print(f"  scale_to_nm : {scale_to_nm}")
    print(f"  recenter    : {recenter}")

    return tmp_path


def load_bbox_nm(mesh_path: str | Path) -> tuple[float, float, float, float, float, float]:
    """
    Return mesh bounding box in nm:
    xmin, ymin, zmin, xmax, ymax, zmax
    """
    mesh = load_mesh(mesh_path)
    bounds = mesh.bounds

    return (
        float(bounds[0, 0]),
        float(bounds[0, 1]),
        float(bounds[0, 2]),
        float(bounds[1, 0]),
        float(bounds[1, 1]),
        float(bounds[1, 2]),
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
) -> tuple[str, list[str]]:
    sim_dendrite_path = prepare_mesh_for_sim(
        dendrite_path,
        scale_to_nm=scale_to_nm,
        recenter=recenter,
    )

    sim_spine_paths = [
        prepare_mesh_for_sim(
            p,
            scale_to_nm=scale_to_nm,
            recenter=recenter,
        )
        for p in spine_paths
    ]

    return sim_dendrite_path, sim_spine_paths