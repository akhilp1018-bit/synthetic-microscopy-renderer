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
6. Move the ROI to the desired position on the mesh.
7. Export the renderer configuration.
8. Run the render or add the configuration to the render queue.

## Rendering subvolume

For fixed-grid rendering, the physical size of the selected subvolume is determined by the output shape and spatial sampling.

For example:

```yaml
output_shape_zyx: [64, 128, 128]
xy_um_per_px: 0.094
z_step_um: 0.5
```

This corresponds to the nominal physical rendering volume:

```text
X = 128 × 0.094 = 12.032 µm
Y = 128 × 0.094 = 12.032 µm
Z = 64 × 0.5   = 32.000 µm
```

The selected position is exported as `grid.center_xyz_nm`.

The selected cube bounds are also stored in the exported configuration as `grid.selected_bounds_xyz_nm` for reproducibility.

## Exported configuration

The GUI creates a complete YAML configuration that can also be used directly with the command-line renderer:

```bash
python scripts/render.py --config gui/configs/<exported_config.yaml>
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH="."
python scripts/render.py --config gui/configs/gui_render.yaml
```

## Render queue

Multiple rendering configurations can be added to the internal render queue.

Each queued job receives its own configuration file so that later GUI changes do not overwrite previously queued settings. Jobs are executed sequentially using the same renderer.

## Hardware

The GUI itself does not require a GPU.

The rendering backend uses PyTorch and can use CUDA when a compatible NVIDIA GPU and CUDA-enabled PyTorch installation are available. GPU execution is recommended for larger rendering tasks.
