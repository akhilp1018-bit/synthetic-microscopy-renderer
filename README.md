# Synthetic Microscopy Renderer

We provide a rendering pipeline for generating synthetic two-photon-like
microscopy images and corresponding ground-truth masks from EM-derived
neuronal surface meshes.

With this pipeline, you can:

1. Render either a single neuronal mesh or manually labelled dendrite and
   spine meshes using configurable imaging and simulation parameters.
2. Choose between voxel-based rendering and Gaussian surface splatting.
3. Generate synthetic microscopy datasets using configurable image-formation
   and noise parameters.
4. Use the generated images and ground truth for dendrite and spine analysis
   with DeepD3.

## Installation

The pipeline is written in Python.

Clone the repository:

```bash
git clone https://github.com/akhilp1018-bit/synthetic-microscopy-renderer.git
cd synthetic-microscopy-renderer
```

Create and activate a Python environment before installing the dependencies.

### Install PyTorch

PyTorch is not pinned in `requirements.txt` because the appropriate installation
depends on the available hardware and compute platform.

Use the official PyTorch installation selector to choose the appropriate
installation for your operating system and, if applicable, CUDA version:

https://pytorch.org/get-started/locally/

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The core rendering pipeline uses packages such as NumPy, SciPy, Trimesh,
tifffile, PyYAML, Matplotlib, and PyTorch.

## Data

The neuronal meshes used in this project were derived from the publicly available
H01 human cortex electron microscopy dataset.

The input mesh data are not included in the repository because of their size.
They can be downloaded from:

[Download the input mesh data](https://faubox.rrze.uni-erlangen.de/getlink/fi3WMHaMpdnQ24JGZc6LFE/)

After downloading, place the files in the project `data/` directory:

```text
data/
├── mesh_001/      # single neuronal mesh for single_mesh rendering
├── sample_001/    # labelled dendrite/spine mesh
├── sample_002/    # labelled dendrite/spine mesh
├── sample_003/    # labelled dendrite/spine mesh
└── sample_004/    # labelled dendrite/spine mesh
```

`mesh_001` is used for the `single_mesh` rendering mode, while
`sample_001` to `sample_004` contain the manually labelled neuronal meshes used for
`labelled_components` rendering, generation of dendrite and spine ground-truth
masks, and synthetic dataset generation.

The exact input paths are specified in the YAML configuration files in `configs/`.

Generated synthetic data and evaluation outputs are not included in the
repository. They can be reproduced using the provided scripts and
configuration files.

For DeepD3-based analysis, the required pre-trained model weights and the real
DeepD3 benchmark data must be obtained separately from the original DeepD3
project:

https://github.com/Beanavil/DeepD3

## Input Modes

The pipeline supports two main input modes.

### Single mesh

`single_mesh` renders a single neuronal mesh without separate dendrite and spine
labels.

This mode is intended for general synthetic microscopy generation and produces a
foreground object mask for the rendered structure.

Typical outputs include:

```text
*_image.tif
*_object_mask.tif
metadata_*.json
```

### Labelled components

`labelled_components` renders separately labelled dendrite and spine components.

This mode is intended for supervised dataset generation and produces separate
ground-truth masks for dendrites and spines.

Typical outputs include:

```text
*_image.tif
*_dendrite_mask.tif
*_spine_mask.tif
metadata_*.json
```

## Repository Structure

```text
synthetic-microscopy-renderer/
├── benchmarks/        # Renderer runtime benchmarks
├── configs/           # Rendering and dataset configuration files
├── deepd3/            # DeepD3 evaluation and visualization scripts
├── gui/               # Interactive mesh visualization utilities
├── psfs/              # Example PSFs
├── scripts/           # Main pipeline entry points: render.py and generate_dataset.py
├── src/               # Core rendering implementation
├── README.md
├── requirements.txt
└── project_spec.md
```

## Workflow

### 1. Prepare neuronal meshes

The pipeline uses neuronal surface meshes reconstructed from electron
microscopy data.

For `single_mesh` rendering, a neuronal mesh can be used directly without
separate dendrite and spine labels.

For `labelled_components` rendering, dendritic shafts and individual spines are
manually labelled before rendering. The labelled mesh geometry is then used both
for fluorescence rendering and for generation of corresponding ground-truth
masks.

### 2. Configure the rendering pipeline

Rendering parameters are specified using YAML configuration files stored in
`configs/`.

The configuration controls parameters such as

- input mesh path
- input mode
- rendering method
- surface sampling density
- output voxel dimensions
- XY and Z sampling
- PSF parameters
- photon count
- Gaussian readout noise
- ground-truth mask thresholds
- random seed and transformations where applicable

### 3. Graphical User Interface

The `gui/` directory provides an interactive interface for loading neuronal
meshes, adjusting rendering settings, and running the renderer.

Run the GUI from the repository root:

```bash
python gui/app.py
```

The same rendering pipeline can also be executed from the command line using
`render.py`.

### 4. Render a single synthetic microscopy volume

For an individual rendering experiment, use `render.py` with a YAML
configuration file.

Example:

```bash
python scripts/render.py --config configs/<config_file>.yaml
```

For example:

```bash
# Single mesh
python scripts/render.py --config configs/mesh_001.yaml

# Labelled dendrite and spine components
python scripts/render.py --config configs/sample_001.yaml
```

Depending on the selected input mode, the renderer generates

- a synthetic microscopy image
- an object mask for `single_mesh`, or
- separate dendrite and spine masks for `labelled_components`
- metadata describing the rendering configuration

Single rendering is useful for testing rendering parameters, inspecting the
effect of different imaging settings, and generating individual synthetic
microscopy volumes.

### 5. Generate a synthetic dataset

For batch generation of multiple transformed samples, use the dataset-generation
script.

Example:

```bash
python scripts/generate_dataset.py --config configs/dataset_v1.yaml
```

The dataset-generation pipeline repeatedly applies the rendering pipeline using
the configured samples, transformations, and random seeds to create multiple
synthetic microscopy volumes.

For labelled meshes, the generated data include

- synthetic microscopy image stacks
- dendrite ground-truth masks
- spine ground-truth masks
- metadata describing the generated samples

## DeepD3 Evaluation

DeepD3 is used as a downstream application to evaluate the generated synthetic
data for dendrite and spine analysis. It is not required for the core rendering
and dataset-generation pipeline.

Install DeepD3 following the installation instructions in the original
DeepD3 repository:

https://github.com/Beanavil/DeepD3

The required pre-trained DeepD3 model weights and real benchmark data should be
downloaded separately from the original DeepD3 project, as described in the Data
section above.

The `deepd3/` directory contains scripts for

- inference and evaluation on the synthetic dataset
- training DeepD3 on synthetic data
- inference and evaluation on the real DeepD3 benchmark
- qualitative visualization of predictions

## Reproducibility

The synthetic microscopy data can be regenerated using the provided configuration
files and pipeline scripts. Input mesh data are available through the link provided
in the Data section.

## Thesis

This repository accompanies the Master's thesis:

**In Silico Light Microscopy of Neuronal Ultrastructures**

DeepD3 is used as a downstream application to evaluate the generated synthetic
data.

## Related Work

DeepD3:

Fernholz, M. H. P., Guggiana Nilo, D. A., Bonhoeffer, T., and Kist, A. M.
“DeepD3, an open framework for automated quantification of dendritic spines.”
PLOS Computational Biology, 2024.

Repository: [DeepD3](https://github.com/Beanavil/DeepD3)