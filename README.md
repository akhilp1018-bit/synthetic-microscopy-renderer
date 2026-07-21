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

### Standard installation

These instructions are intended for a normal Linux or macOS computer.

Clone the repository and enter its root directory:

```bash
git clone <https://github.com/akhilp1018-bit/synthetic-microscopy-renderer>
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

Install PyTorch:

```bash
python -m pip install torch
```

Install the remaining project dependencies:

```bash
python -m pip install -r requirements.txt
```

Verify the installation:

```bash
python -c "import torch, numpy, trimesh, tifffile, yaml, matplotlib; print('Installation successful'); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

On a computer without a compatible NVIDIA GPU, the renderer can run on the CPU, although large rendering tasks may be slow and require substantial memory.

For a CUDA-enabled installation, install a PyTorch build that is compatible with the computer's GPU and CUDA setup.

### Windows installation

Create and activate the environment in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install torch
python -m pip install -r requirements.txt
```

Verify the installation:

```powershell
python -c "import torch, numpy, trimesh, tifffile, yaml, matplotlib; print('Installation successful'); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

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

The cluster-provided PyTorch module is recommended because it is compatible with the available Tesla V100 GPUs.

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

Verify GPU support on a GPU node:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); print('Architectures:', torch.cuda.get_arch_list())"
```

For the Tesla V100, the supported architecture list should include:

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

Typical outputs for `single_mesh` mode are:

```text
zstack_*_image.tif
zstack_*_object_mask.tif
metadata_*.json
```

Typical outputs for `labelled_components` mode are:

```text
zstack_*_image.tif
zstack_*_dendrite_mask.tif
zstack_*_spine_mask.tif
metadata_*.json
```

Depending on the configuration, the renderer may also create preview images or additional intermediate outputs.

## Configuration

The default example configuration is:

```text
configs/default.yaml
```

To run a different experiment, edit an existing YAML file or create a new configuration and pass its path with `--config`:

```bash
PYTHONPATH=. python scripts/render.py --config configs/<configuration-name>.yaml
```
