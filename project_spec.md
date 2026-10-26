# Project Specification

## Input

- Surface mesh files in PLY format.
- Mesh coordinates are converted to nanometres using `input.scale_to_nm`.
- Two input modes are supported:
  - `single_mesh`: one neuronal mesh.
  - `labelled_components`: separate dendrite and individual spine meshes.

## Coordinate convention

- Mesh coordinates use XYZ order.
- Output image and mask arrays use ZYX order.
- `xy_um_per_px` controls the X/Y pixel spacing.
- `z_step_um` controls the Z slice spacing.

## Output

- Synthetic microscopy image stack: uint16 ImageJ TIFF, ZYX.
- Ground-truth mask stack(s): uint16 ImageJ TIFF, ZYX.
- Rendering and configuration metadata: JSON.

## Renderers

Supported rendering methods:

- `voxel_grid`
- `gaussian_splatting`

The voxel-grid renderer supports both membrane and filled rendering.

Gaussian splatting represents the mesh surface using anisotropic Gaussian kernels and is used for membrane rendering.

## Label modes

### `membrane`

Represents fluorescence associated with the mesh surface.

For voxel-grid rendering, surface samples contribute physical surface-area weights to the voxel grid.

### `filled`

Produces an approximate filled-volume representation from the voxelized mesh surface using morphological closing and hole filling.

## PSF

The pipeline supports two PSF approaches:

- Born-Wolf PSFs generated using Fiji.
- An analytical Gaussian approximation of a two-photon PSF.

The analytical Gaussian two-photon approximation is used for most synthetic rendering and dataset-generation experiments.

The PSF is convolved with the rendered fluorescence distribution to produce the microscopy-like image.

PSF sampling is matched to the output voxel spacing.

## Noise

Optional image noise can be applied after PSF convolution.

Supported noise models include:

- Poisson photon-counting noise.
- Gaussian detector/readout noise.
- Combined Poisson-Gaussian noise.

Ground-truth masks are generated from the clean signal before noise is applied.

## Masks

For `single_mesh` input:

- object mask.

For `labelled_components` input:

- dendrite mask.
- combined spine mask.
- optional individual spine masks.

Background is represented implicitly by zero-valued voxels.

## Output grid and spatial sampling

The physical voxel spacing is controlled by:

```yaml
grid:
  xy_um_per_px: ...
  z_step_um: ...
```

Two grid modes are supported.

### Automatic shape

```yaml
grid:
  shape_mode: auto
```

The output shape is determined from the mesh bounds or an optional ROI.

### Fixed shape

```yaml
grid:
  shape_mode: fixed
  output_shape_zyx: [Z, Y, X]
```

The output dimensions are specified explicitly. The physical size of the rendered subvolume is determined by the output shape together with the configured XY and Z sampling.

The standard ROI size parameters are not used in fixed mode.