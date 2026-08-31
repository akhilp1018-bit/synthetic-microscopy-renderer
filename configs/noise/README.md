# Noise experiment configurations

Each YAML file is a complete reproducible configuration. Comments inside each file describe
the specific condition and the parameters that are held fixed.

Run any configuration from the repository root:

```bash
python scripts/render.py --config configs/noise/<CONFIG>.yaml
```

Replace `<CONFIG>.yaml` with the desired configuration file, for example
`low_100_poisson.yaml`, `medium_500_poisson_gaussian.yaml`, or `seed1.yaml`.

The required Python environment and project dependencies must be installed before running.
