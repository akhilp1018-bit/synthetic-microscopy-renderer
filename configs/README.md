

## Renderer configuration options

The renderer is controlled by YAML configuration files in the `configs/` folder.

The main documented example is:

```bash
PYTHONPATH=. python scripts/render.py --config configs/default.yaml
```

If `python` is not available, use:

```bash
PYTHONPATH=. python3 scripts/render.py --config configs/default.yaml
```

### Available config files

```text
configs/default.yaml      # main documented example
configs/mesh_001.yaml     # single mesh example, scale_to_nm = 1.0
configs/sample_001.yaml   # labelled components, scale_to_nm = 1000000.0
configs/sample_002.yaml   # labelled components, scale_to_nm = 1000.0
configs/sample_003.yaml   # labelled components, scale_to_nm = 1000.0
configs/sample_004.yaml   # labelled components, scale_to_nm = 1.0
```

### Run a specific sample

```bash
PYTHONPATH=. python scripts/render.py --config configs/sample_001.yaml
PYTHONPATH=. python scripts/render.py --config configs/sample_002.yaml
PYTHONPATH=. python scripts/render.py --config configs/sample_003.yaml
PYTHONPATH=. python scripts/render.py --config configs/sample_004.yaml
```

### Create preview overlays

After rendering a sample, preview overlays can be generated with:

```bash
PYTHONPATH=. python scripts/make_overlay.py --output-dir outputs/sample_001/gaussian_2p_voxelgrid_membrane
```

Change the output folder for other samples, for example:

```bash
PYTHONPATH=. python scripts/make_overlay.py --output-dir outputs/sample_002/gaussian_2p_voxelgrid_membrane
```

### FAU HPC note

On the FAU HPC, I currently use my existing thesis environment:

```bash
PYTHONPATH=. /home/hpc/iwb3/iwb3119h/synthetic-spine-sim/thesis_env/bin/python scripts/render.py --config configs/sample_001.yaml
```

For preview overlays on the FAU HPC:

```bash
PYTHONPATH=. /home/hpc/iwb3/iwb3119h/synthetic-spine-sim/thesis_env/bin/python scripts/make_overlay.py --output-dir outputs/sample_001/gaussian_2p_voxelgrid_membrane
```
