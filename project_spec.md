# Project Specification

Input:
- mesh file in PLY format
- mesh coordinates converted to nm

Coordinate convention:
- mesh coordinates are XYZ
- output arrays are ZYX
- `xy_um_per_px` controls X/Y pixel size
- `z_step_um` controls Z slice spacing
s
Output:
- image stack: uint16 TIFF, ZYX
- mask stack: uint16 TIFF, ZYX
- metadata: JSON

Renderers:
- `voxel_grid`
- `gaussian_splatting`

Label modes:
- `membrane`
- `pseudofilled`

PSF:
- supported modes: `bornwolf_1p`, `bornwolf_2p`, `gaussian_2p`
- PSF is used for microscopy-like image formation
- PSF sampling should match the output voxel size

Masks:
- `single_mesh`: object mask
- `labelled_components`: separate dendrite mask and spine mask
- background is implicit where masks are zero

Output resolution:
- default: output shape is computed from mesh bounding box or ROI
- optional fixed shape: `output_shape_zyx` can be used for fixed-size patches
- fixed-size patches require selecting an ROI center that contains mesh geometry