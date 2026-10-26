## Renderer configuration options

The renderer is controlled by YAML configuration files in the `configs/` folder.

```text
configs/
├── noise/
├── resolution_study/
├── dataset_v1.yaml
├── default.yaml
├── mesh_001.yaml
├── sample_001.yaml
├── sample_002.yaml
├── sample_003.yaml
├── sample_004.yaml
├── test_gaussian_splatting.yaml
├── test_noise.yaml
├── test_voxelgrid_filled.yaml
└── test_voxelgrid.yaml
```

### Default example configuration

`default.yaml` provides a general example configuration for the renderer.

Run it from the repository root with:

```bash
python scripts/render.py --config configs/default.yaml
```

If `python` is not available, use:

```bash
python3 scripts/render.py --config configs/default.yaml
```

### Sample-specific configurations

To render a specific mesh or labelled sample:

```bash
python scripts/render.py --config configs/mesh_001.yaml
python scripts/render.py --config configs/sample_001.yaml
python scripts/render.py --config configs/sample_002.yaml
python scripts/render.py --config configs/sample_003.yaml
python scripts/render.py --config configs/sample_004.yaml
```

`mesh_001.yaml` uses a single neuronal mesh.

`sample_001.yaml` to `sample_004.yaml` use manually labelled dendrite and individual spine meshes.

### Dataset generation configuration

`dataset_v1.yaml` contains the configuration used for synthetic dataset generation.

```bash
python scripts/generate_dataset.py --config configs/dataset_v1.yaml
```

### Parameter-study configurations

The `noise/` directory contains configurations used to study different photon-count and noise conditions.

The `resolution_study/` directory contains configurations used to study spatial sampling and image-resolution settings, including different XY sampling and Z spacing.

These configurations are intended for controlled parameter studies rather than the standard sample-rendering workflow.

### Test and renderer-comparison configurations

The following configurations are used for testing individual rendering and noise settings:

```text
test_voxelgrid.yaml
test_voxelgrid_filled.yaml
test_gaussian_splatting.yaml
test_noise.yaml
```

They are mainly used for renderer comparisons, debugging, and controlled experiments.

For example:

```bash
python scripts/render.py --config configs/test_voxelgrid.yaml
python scripts/render.py --config configs/test_voxelgrid_filled.yaml
python scripts/render.py --config configs/test_gaussian_splatting.yaml
python scripts/render.py --config configs/test_noise.yaml
```

### Mesh scale factors

The tested mesh-to-nanometre scale factors are:

```text
mesh_001   : 1.0
sample_001 : 1000000.0
sample_002 : 10000.0
sample_003 : 1000.0
sample_004 : 1.0
```

These values are already defined in the corresponding YAML files through `input.scale_to_nm`.

