# 3D Mesh Renderer GUI

The GUI provides an interactive interface for configuring synthetic microscopy renders from 3D mesh geometry.

## Installation

Install PyTorch using the appropriate CPU or CUDA build for the target computer, as described in the main repository README.

Then install the remaining project dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Run the GUI

From the repository root:

```bash
python gui/app.py
```

## Basic workflow

1. Load a mesh.
2. Rotate, zoom, and inspect the mesh in the 3D viewer.
3. Select the output image shape `[Z, Y, X]`.
4. Select the XY and Z sampling.
5. Add the rendering ROI.
6. Move or rotate the ROI to the desired position on the mesh.
7. Export the renderer configuration.
8. Run the render or add the configuration to the render queue.

## Rendering subvolume

For fixed-grid rendering, the physical extent of the selected subvolume is determined by the output shape and spatial sampling.

For example:

```yaml
output_shape_zyx: [64, 128, 128]
xy_um_per_px: 0.094
z_step_um: 0.5
```

Using the distance between the first and last voxel centres, this corresponds to:

```text
X = 127 × 0.094 = 11.938 µm
Y = 127 × 0.094 = 11.938 µm
Z = 63 × 0.5   = 31.500 µm
```

For an unrotated ROI, the selected position is exported as `grid.center_xyz_nm`.

If the ROI is rotated, the GUI exports the corresponding
`grid.roi_world_to_local_4x4` transformation so that the renderer can reproduce
the selected ROI orientation.

## Exported configuration

The GUI creates a complete YAML configuration that can also be used directly with the command-line renderer:

```bash
python scripts/render.py --config gui/configs/<exported_config.yaml>
```

## Render queue

Multiple rendering configurations can be added to the internal render queue.

Each queued job receives its own configuration file so that later GUI changes do not overwrite previously queued settings. Jobs are executed sequentially using the same renderer.

## Hardware

The GUI itself does not require a GPU.

The rendering backend uses PyTorch and can use CUDA when a compatible NVIDIA GPU and CUDA-enabled PyTorch installation are available. GPU execution is recommended for larger rendering tasks.