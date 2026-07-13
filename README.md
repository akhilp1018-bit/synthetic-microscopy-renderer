Synthetic Microscopy Renderer

This repository renders synthetic microscopy image stacks from mesh geometry.

Basic usage:
mesh + config → TIFF image stack + mask + metadata

Install:
pip install -r requirements.txt

Run:
PYTHONPATH=. python scripts/render.py --config configs/default.yaml

Expected outputs:
- image TIFF stack
- object mask TIFF stack
- metadata JSON