import numpy as np


def compute_full_bbox(bbox_dict: dict, margin: float = 0.05) -> dict:
    xmin0 = bbox_dict["xmin"]
    xmax0 = bbox_dict["xmax"]
    ymin0 = bbox_dict["ymin"]
    ymax0 = bbox_dict["ymax"]

    xrange_nm = xmax0 - xmin0
    yrange_nm = ymax0 - ymin0

    return {
        "xmin": xmin0 - margin * xrange_nm,
        "xmax": xmax0 + margin * xrange_nm,
        "ymin": ymin0 - margin * yrange_nm,
        "ymax": ymax0 + margin * yrange_nm,
        "zmin": bbox_dict["zmin"],
        "zmax": bbox_dict["zmax"],
    }


def compute_roi_bbox(
    bbox_dict: dict,
    roi_size_um_x: float,
    roi_size_um_y: float,
    margin: float = 0.05,
) -> dict:
    full = compute_full_bbox(bbox_dict, margin=margin)

    cx_nm = 0.5 * (full["xmin"] + full["xmax"])
    cy_nm = 0.5 * (full["ymin"] + full["ymax"])

    half_x_nm = roi_size_um_x * 1000.0 * 0.5
    half_y_nm = roi_size_um_y * 1000.0 * 0.5

    return {
        "xmin": cx_nm - half_x_nm,
        "xmax": cx_nm + half_x_nm,
        "ymin": cy_nm - half_y_nm,
        "ymax": cy_nm + half_y_nm,
        "zmin": bbox_dict["zmin"],
        "zmax": bbox_dict["zmax"],
    }


def compute_voxel_grid(render_bbox: dict, xy_um_per_px: float, z_step_um: float) -> dict:
    xmin = render_bbox["xmin"]
    xmax = render_bbox["xmax"]
    ymin = render_bbox["ymin"]
    ymax = render_bbox["ymax"]
    zmin = render_bbox["zmin"]
    zmax = render_bbox["zmax"]

    voxel_x_nm = xy_um_per_px * 1000.0
    voxel_y_nm = xy_um_per_px * 1000.0
    voxel_z_nm = z_step_um * 1000.0

    xspan_um = (xmax - xmin) / 1000.0
    yspan_um = (ymax - ymin) / 1000.0

    W = int(np.ceil(xspan_um / xy_um_per_px)) + 1
    H = int(np.ceil(yspan_um / xy_um_per_px)) + 1
    NUM_SLICES = int(np.ceil((zmax - zmin) / voxel_z_nm)) + 1

    print(f"Voxel grid : W={W}, H={H}, Z={NUM_SLICES}")
    print(f"FOV        : {xspan_um:.2f} µm x {yspan_um:.2f} µm")
    print(f"Depth      : {(zmax - zmin) / 1000.0:.2f} µm")

    return {
        "W": W,
        "H": H,
        "NUM_SLICES": NUM_SLICES,
        "origin_nm": (xmin, ymin, zmin),
        "voxel_size_nm_xyz": (voxel_x_nm, voxel_y_nm, voxel_z_nm),
        "shape_zyx": (NUM_SLICES, H, W),
    }