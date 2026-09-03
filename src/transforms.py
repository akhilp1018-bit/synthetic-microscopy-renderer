"""
Geometry transformations for synthetic dataset generation.

The dendrite and all associated spines receive the SAME rotation,
so their original spatial relationship is preserved.

Mesh coordinates are XYZ in nanometres after scaling.
"""

from pathlib import Path

import numpy as np
import trimesh


def random_rotation_matrix(rng=None):
    """Create a random 3D rotation matrix."""

    if rng is None:
        rng = np.random.default_rng()

    angles_deg = rng.uniform(0.0, 360.0, size=3)
    rx, ry, rz = np.deg2rad(angles_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx),  np.cos(rx)],
    ])

    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)],
    ])

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [0, 0, 1],
    ])

    R = Rz @ Ry @ Rx

    return R, angles_deg


def transform_mesh_vertices(mesh, rotation_matrix, origin_xyz):
    """Rotate one mesh around a shared XYZ origin."""

    transformed_mesh = mesh.copy()

    vertices = np.asarray(
        transformed_mesh.vertices,
        dtype=np.float64,
    )

    origin_xyz = np.asarray(
        origin_xyz,
        dtype=np.float64,
    )

    # Move to origin -> rotate -> move back
    vertices = vertices - origin_xyz
    vertices = vertices @ rotation_matrix.T
    vertices = vertices + origin_xyz

    transformed_mesh.vertices = vertices

    return transformed_mesh


def rotate_labelled_components(
    dendrite_path,
    spine_paths,
    output_dir,
    scale_to_nm=1.0,
    random_orientation=True,
):
    """
    Apply the SAME random rotation to dendrite and all spine meshes.

    This is important because rotating components independently would
    destroy the original dendrite-spine geometry.
    """

    dendrite_path = Path(dendrite_path)
    spine_paths = [Path(p) for p in spine_paths]
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Load and scale dendrite
    # ---------------------------------------------------------

    dendrite_mesh = trimesh.load(
        dendrite_path,
        force="mesh",
        process=False,
    )

    dendrite_mesh.vertices = (
        np.asarray(
            dendrite_mesh.vertices,
            dtype=np.float64,
        )
        * float(scale_to_nm)
    )

    # Common rotation centre for dendrite + all spines.
    origin_xyz = np.asarray(
        dendrite_mesh.vertices,
        dtype=np.float64,
    ).mean(axis=0)

    # ---------------------------------------------------------
    # Choose one rotation for the whole labelled structure
    # ---------------------------------------------------------

    if random_orientation:
        rotation_matrix, angles_deg = random_rotation_matrix()
    else:
        rotation_matrix = np.eye(3)
        angles_deg = np.zeros(3)

    # ---------------------------------------------------------
    # Rotate dendrite
    # ---------------------------------------------------------

    dendrite_rotated = transform_mesh_vertices(
        dendrite_mesh,
        rotation_matrix,
        origin_xyz,
    )

    rotated_dendrite_path = (
        output_dir / "dendrite00.ply"
    )

    dendrite_rotated.export(
        rotated_dendrite_path
    )

    # ---------------------------------------------------------
    # Rotate all spines using exactly the SAME transform
    # ---------------------------------------------------------

    rotated_spine_paths = []

    for spine_path in spine_paths:

        spine_mesh = trimesh.load(
            spine_path,
            force="mesh",
            process=False,
        )

        spine_mesh.vertices = (
            np.asarray(
                spine_mesh.vertices,
                dtype=np.float64,
            )
            * float(scale_to_nm)
        )

        spine_rotated = transform_mesh_vertices(
            spine_mesh,
            rotation_matrix,
            origin_xyz,
        )

        rotated_path = (
            output_dir / spine_path.name
        )

        spine_rotated.export(
            rotated_path
        )

        rotated_spine_paths.append(
            rotated_path
        )

    # Store the actual transformation for metadata.
    transform_metadata = {
        "random_orientation": bool(random_orientation),
        "rotation_angles_deg_xyz": [
            float(v) for v in angles_deg
        ],
        "rotation_origin_xyz_nm": [
            float(v) for v in origin_xyz
        ],
        "rotation_matrix": rotation_matrix.tolist(),
    }

    return (
        rotated_dendrite_path,
        rotated_spine_paths,
        transform_metadata,
    )


def select_random_geometry_center(dendrite_path):
    """
    Select a random FOV centre from actual dendrite geometry.

    A dendrite vertex is selected instead of a random location from
    the complete bounding box. This helps keep neuronal geometry
    inside the generated training instance.
    """

    dendrite_mesh = trimesh.load(
        dendrite_path,
        force="mesh",
        process=False,
    )

    vertices = np.asarray(
        dendrite_mesh.vertices,
        dtype=np.float64,
    )

    if len(vertices) == 0:
        raise ValueError(
            f"Dendrite mesh contains no vertices: {dendrite_path}"
        )

    rng = np.random.default_rng()

    vertex_index = int(
        rng.integers(
            0,
            len(vertices),
        )
    )

    center_xyz_nm = vertices[
        vertex_index
    ]

    return tuple(
        float(v)
        for v in center_xyz_nm
    )