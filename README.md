# Synthetic Microscopy Renderer

This repository renders synthetic microscopy image stacks from 3D mesh geometry.

The goal is to generate synthetic two-photon-like image stacks together with ground-truth masks and metadata. These outputs can be used for DeepD3 spine and dendrite segmentation experiments.

## Basic idea

```text
3D mesh + YAML configuration
            |
            v
TIFF image stack + ground-truth mask(s) + metadata
```

## Repository contents

```text
configs/      YAML configuration files
scripts/      Command-line scripts
src/          Renderer source code
psfs/         Point spread function files
data/         Mesh data; not stored in GitHub
outputs/      Generated results; not stored in GitHub
```

## Example data

Large mesh files are not stored directly in GitHub.

Download the mesh data from FAUbox:

> FAUbox download link will be added here.

After downloading, place the `data` folder in the repository root. The expected structure is:

```text
synthetic-microscopy-renderer/
├── configs/
├── data/
│   ├── mesh_001/
│   │   └── mesh_001.ply
│   ├── sample_001/
│   ├── sample_002/
│   ├── sample_003/
│   └── sample_004/
├── outputs/
├── psfs/
├── scripts/
└── src/
```

The PSF files are already included in the repository under:

```text
psfs/
```

## Installation

PyTorch is not pinned in `requirements.txt` because the correct installation depends on the operating system, Python version, GPU, and CUDA setup.

Use the official PyTorch installation selector to obtain the correct command for your computer:

[PyTorch installation selector](https://pytorch.org/get-started/locally/)

Install PyTorch first, then install the remaining project dependencies.

### Standard Linux or macOS installation

Clone the repository and enter its root directory:

```bash
git clone https://github.com/akhilp1018-bit/synthetic-microscopy-renderer.git
cd synthetic-microscopy-renderer
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the appropriate PyTorch build using the command provided by the official PyTorch installation selector.

For example, a CPU-only installation may use:

```bash
python -m pip install torch torchvision
```

For a CUDA-enabled installation, select the correct operating system, package manager, Python version, and CUDA version on the PyTorch website, then run the generated command.

Install the remaining project dependencies:

```bash
python -m pip install -r requirements.txt
```

Verify the installation:

```bash
python -c "import torch, numpy, trimesh, tifffile, yaml, matplotlib; print('Installation successful'); print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

On a computer without a compatible NVIDIA GPU, the renderer can run on the CPU, although large rendering tasks may be slow and require substantial memory.

### Windows installation

Clone the repository and enter its root directory:

```powershell
git clone https://github.com/akhilp1018-bit/synthetic-microscopy-renderer.git
cd synthetic-microscopy-renderer
```

Create and activate the environment in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Use the official PyTorch installation selector and choose Windows, Pip, Python, and the CUDA version supported by your system, or CPU if no compatible GPU is available.

Then run the generated PyTorch installation command.

Example only:

```powershell
python -m pip install torch torchvision
```

Install the remaining project dependencies:

```powershell
python -m pip install -r requirements.txt
```

Verify the installation:

```powershell
python -c "import torch, numpy, trimesh, tifffile, yaml, matplotlib; print('Installation successful'); print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

If `CUDA available` is `False` on a computer with an NVIDIA GPU, check that the installed PyTorch build matches the system's supported CUDA configuration.

## Run an example

From the repository root, run:

```bash
PYTHONPATH=. python scripts/render.py --config configs/default.yaml
```

On Windows PowerShell, use:

```powershell
$env:PYTHONPATH="."
python scripts/render.py --config configs/default.yaml
```

The rendering settings, input paths, output paths, microscope model, sampling resolution, and other parameters are defined in the YAML configuration file.

## Optional: FAU HPC installation

The FAU HPC can be used for larger rendering tasks that require a GPU or more memory.

The cluster-provided PyTorch module is recommended because it is configured for the HPC environment and available GPUs.

Load the module:

```bash
module purge
module load python/pytorch2.6py3.12
```

Create a virtual environment that can access the module-provided PyTorch installation:

```bash
python -m venv --system-site-packages .venv-hpc
source .venv-hpc/bin/activate
```

Upgrade pip and install the remaining project dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Do not install another PyTorch version inside this environment unless the cluster module is intentionally being replaced.

Verify GPU support on a GPU node:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); print('Architectures:', torch.cuda.get_arch_list())"
```

For a Tesla V100 GPU, the supported architecture list should include:

```text
sm_70
```

### Request an FAU HPC GPU node

An example interactive allocation is:

```bash
salloc --partition=v100 --gres=gpu:v100:1 --time=02:00:00 --cpus-per-task=8
```

After the allocation starts, load the module and activate the environment:

```bash
module purge
module load python/pytorch2.6py3.12
source .venv-hpc/bin/activate
```

Run the renderer from the repository root:

```bash
PYTHONPATH=. python scripts/render.py --config configs/default.yaml
```

The HPC instructions are optional. The renderer can also be used on a normal workstation or laptop, depending on the size of the selected mesh and rendering configuration.

## Expected outputs

The renderer writes its results to the output folder defined in the configuration file.

### Unlabelled single-mesh mode

Typical outputs for `single_mesh` mode are:

```text
zstack_*_image.tif
zstack_*_object_mask.tif
metadata_*.json
```

The `object_mask` is a binary foreground/background mask generated from the complete unlabelled mesh:

- foreground: voxels occupied by the mesh
- background: empty voxels

It does not distinguish dendrites from spines.

### Labelled-components mode

Typical outputs for `labelled_components` mode are:

```text
zstack_*_image.tif
zstack_*_dendrite_mask.tif
zstack_*_spine_mask.tif
metadata_*.json
```

In this mode, the renderer generates separate ground-truth masks for dendrites and spines in the same coordinate system as the rendered image.

Depending on the configuration, the renderer may also create preview images, overlays, or additional intermediate outputs.

## Configuration

The default example configuration is:

```text
configs/default.yaml
```

To run a different experiment, edit an existing YAML file or create a new configuration and pass its path with `--config`:

```bash
PYTHONPATH=. python scripts/render.py --config configs/<configuration-name>.yaml
```
