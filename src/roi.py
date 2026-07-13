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


def compute_voxel_grid(
    render_bbox: dict,
    xy_um_per_px: float,
    z_step_um: float,
    output_shape_zyx=None,
) -> dict:
    xmin = float(render_bbox["xmin"])
    xmax = float(render_bbox["xmax"])
    ymin = float(render_bbox["ymin"])
    ymax = float(render_bbox["ymax"])
    zmin = float(render_bbox["zmin"])
    zmax = float(render_bbox["zmax"])

    voxel_x_nm = float(xy_um_per_px) * 1000.0
    voxel_y_nm = float(xy_um_per_px) * 1000.0
    voxel_z_nm = float(z_step_um) * 1000.0

    xspan_um = (xmax - xmin) / 1000.0
    yspan_um = (ymax - ymin) / 1000.0
    zspan_um = (zmax - zmin) / 1000.0

    if output_shape_zyx is None:
        W = int(np.ceil(xspan_um / xy_um_per_px)) + 1
        H = int(np.ceil(yspan_um / xy_um_per_px)) + 1
        Z = int(np.ceil((zmax - zmin) / voxel_z_nm)) + 1
        shape_zyx = (Z, H, W)
        shape_mode = "auto"
    else:
        shape_zyx = tuple(int(v) for v in output_shape_zyx)
        Z, H, W = shape_zyx
        shape_mode = "fixed"

        cx_nm = 0.5 * (xmin + xmax)
        cy_nm = 0.5 * (ymin + ymax)
        cz_nm = 0.5 * (zmin + zmax)

        xmin = cx_nm - 0.5 * (W - 1) * voxel_x_nm
        ymin = cy_nm - 0.5 * (H - 1) * voxel_y_nm
        zmin = cz_nm - 0.5 * (Z - 1) * voxel_z_nm

        xmax = xmin + (W - 1) * voxel_x_nm
        ymax = ymin + (H - 1) * voxel_y_nm
        zmax = zmin + (Z - 1) * voxel_z_nm

        xspan_um = (xmax - xmin) / 1000.0
        yspan_um = (ymax - ymin) / 1000.0
        zspan_um = (zmax - zmin) / 1000.0

    print(f"Grid mode  : {shape_mode}")
    print(f"Voxel grid : W={W}, H={H}, Z={Z}")
    print(f"FOV        : {xspan_um:.2f} µm x {yspan_um:.2f} µm")
    print(f"Depth      : {zspan_um:.2f} µm")

    return {
        "W": W,
        "H": H,
        "NUM_SLICES": Z,
        "origin_nm": (xmin, ymin, zmin),
        "voxel_size_nm_xyz": (voxel_x_nm, voxel_y_nm, voxel_z_nm),
        "shape_zyx": shape_zyx,
    }