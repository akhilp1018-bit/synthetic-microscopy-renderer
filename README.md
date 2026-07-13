# Synthetic Microscopy Renderer

This repository renders synthetic microscopy image stacks from 3D mesh geometry.

The goal is to generate synthetic two-photon-like image stacks together with ground-truth masks and metadata. These outputs will later be used for DeepD3 spine and dendrite segmentation experiments.

## Basic idea

mesh + config -> TIFF image stack + mask + metadata


## Repository contents


configs/      YAML configuration files
scripts/      command-line scripts
src/          renderer source code
psfs/         point spread function files
data/         mesh data folder, not stored in GitHub
outputs/      generated outputs, not stored in GitHub


## Example data

Large mesh files are not stored directly in GitHub.

Download the mesh data from FAUbox:

https://faubox.rrze.uni-erlangen.de/getlink/fiPoqGFeM4ym6iUhvyyHpu/data

After downloading, place the `data` folder in the repository root so that the structure is:

synthetic-microscopy-renderer/
└── data/
    ├── mesh_001/
    │   └── mesh_001.ply
    ├── sample_001/
    ├── sample_002/
    ├── sample_003/
    └── sample_004/


The PSF files are already included in the repository under:


psfs/


## Installation

Create or activate a Python environment, then install the requirements:

pip install -r requirements.txt


## Run one example

From the repository root, run:


PYTHONPATH=. python scripts/render.py --config configs/default.yaml


On the HPC, the current environment can be used with:


PYTHONPATH=. /home/hpc/iwb3/iwb3119h/synthetic-spine-sim/thesis_env/bin/python scripts/render.py --config configs/default.yaml


## Expected outputs

The renderer writes outputs to the folder defined in the config file.

Typical outputs for `single_mesh` mode are:

zstack_*_image.tif
zstack_*_object_mask.tif
metadata_*.json


For `labelled_components` mode, the expected outputs are:

zstack_*_image.tif
zstack_*_dendrite_mask.tif
zstack_*_spine_mask.tif
metadata_*.json




