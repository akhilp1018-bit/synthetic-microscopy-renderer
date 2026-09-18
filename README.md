# Synthetic Microscopy Renderer

This repository generates synthetic microscopy image stacks from 3D neuronal mesh geometry.

The pipeline converts labelled or unlabelled mesh data into microscopy-like TIFF stacks using configurable rendering, point spread function (PSF), and noise models. For labelled neuronal meshes, corresponding dendrite and spine ground-truth masks can also be generated.

The synthetic data can be used for training and evaluating segmentation methods such as DeepD3.

## Pipeline

```text
3D neuronal mesh
       |
       v
Fluorescence-density rendering
       |
       v
PSF convolution
       |
       v
Microscopy noise
       |
       v
TIFF image stack + ground-truth masks + metadata
```

Two rendering approaches are available:

- Voxel-based fluorescence rendering
- Gaussian surface-splatting fluorescence rendering

The voxel renderer supports membrane and filled-volume labelling. Gaussian splatting represents fluorescence using normal-oriented Gaussian surface primitives.

## Repository structure

```text
configs/     YAML configuration files
data/        Mesh data (not stored in GitHub)
gui/         Graphical user interface
psfs/        Point spread function files
scripts/     Rendering and dataset-generation scripts
src/         Renderer source code
outputs/     Generated results (not stored in GitHub)
```

## Mesh data

The mesh dataset is not stored directly in this repository because of its size.

Download link:

> FAUbox download link will be added here.

After downloading, place the `data` directory in the repository root:

```text
synthetic-microscopy-renderer/
├── configs/
├── data/
│   ├── mesh_001/
│   ├── sample_001/
│   ├── sample_002/
│   ├── sample_003/
│   └── sample_004/
├── gui/
├── outputs/
├── psfs/
├── scripts/
└── src/
```

The required PSF files are included under `psfs/`.

## Installation on Windows

Python 3.12 is recommended.

Clone the repository and enter the project directory:

```powershell
git clone https://github.com/akhilp1018-bit/synthetic-microscopy-renderer.git
cd synthetic-microscopy-renderer
```

Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

### Install PyTorch

PyTorch is not pinned in `requirements.txt` because the appropriate installation depends on the available hardware and CUDA version.

Use the official PyTorch installation selector:

https://pytorch.org/get-started/locally/

For a CPU installation, for example:

```powershell
python -m pip install torch torchvision
```

For an NVIDIA GPU, use the installation command recommended by PyTorch for the CUDA configuration of the system.

Then install the remaining dependencies:

```powershell
python -m pip install -r requirements.txt
```

Verify the PyTorch installation:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

The renderer automatically uses CUDA when a compatible GPU and CUDA-enabled PyTorch installation are available; otherwise it runs on the CPU.

## Run the renderer

From the repository root in PowerShell:

```powershell
$env:PYTHONPATH="."
python scripts/render.py --config configs/default.yaml
```

Rendering parameters such as the input mesh, output resolution, renderer, PSF, noise model, and output directory are controlled through YAML configuration files.

To use another configuration:

```powershell
python scripts/render.py --config configs/<configuration-name>.yaml
```

## Input modes

### Single mesh

`single_mesh` renders one mesh and generates a foreground object mask.

Typical outputs are:

```text
*_image.tif
*_object_mask.tif
metadata_*.json
```

### Labelled components

`labelled_components` renders labelled dendrite and spine components and generates separate ground-truth masks.

Typical outputs are:

```text
*_image.tif
*_dendrite_mask.tif
*_spine_mask.tif
metadata_*.json
```

Ground-truth masks are generated from the clean rendered signal before microscopy noise is applied.

## Synthetic dataset generation

Synthetic datasets can be generated from the labelled neuronal meshes using:

```powershell
$env:PYTHONPATH="."
python scripts/generate_dataset.py --config configs/dataset_v1.yaml
```

The dataset generator supports random mesh orientations and random locations along the dendrite. Randomness is controlled by the configured seed for reproducibility.

A generated instance can contain:

```text
clean.tif
noisy.tif
dendrite_mask.tif
spine_mask.tif
combined_mask.tif
metadata.json
```

The number of instances, spatial sampling, renderer, PSF, noise model, mask settings, and other parameters are defined in `configs/dataset_v1.yaml`.

## Graphical user interface

A GUI is available for loading meshes, selecting rendering regions, exporting configurations, and launching renders.

Run:

```powershell
python gui/app.py
```

See `gui/README.md` for additional GUI instructions.

## Configuration

The main example configuration is:

```text
configs/default.yaml
```

Additional YAML files under `configs/` define specific rendering and experimental configurations.