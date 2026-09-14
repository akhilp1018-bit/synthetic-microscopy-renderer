# Synthetic Microscopy Noise Configurations

Reproducible configurations for testing clean, Poisson, Gaussian, and combined Poisson-Gaussian image noise while keeping the mesh, renderer, sampling, and PSF settings fixed.

Run a configuration from the repository root:

```bash
python scripts/render.py --config configs/noise/<CONFIG>.yaml
```

## Configurations

- `clean.yaml` — clean reference without added noise.
- `high_2000_poisson.yaml`, `medium_500_poisson.yaml`, `low_100_poisson.yaml`, `very_low_20_poisson.yaml`, `extreme_low_5_poisson.yaml` — Poisson-only photon-count sweep.
- `gaussian_std5.yaml`, `gaussian_std20.yaml`, `gaussian_std50.yaml`, `gaussian_std100.yaml`, `gaussian_std250.yaml`, `gaussian_std500.yaml` — Gaussian-only noise sweep with the signal scale fixed at `peak_photons = 500`.
- `high_2000_poisson_gaussian.yaml`, `medium_500_poisson_gaussian.yaml`, `low_100_poisson_gaussian.yaml`, `strong_poisson_gaussian.yaml`, `very_strong_poisson_gaussian.yaml`, `noise_dominated_poisson_gaussian.yaml` — combined Poisson-Gaussian conditions.
- `seed0.yaml`, `seed1.yaml`, `seed2.yaml` — reproducibility examples using different random seeds.

`peak_photons` is a relative synthetic photon/signal budget, not an experimentally calibrated absolute photon count. For controlled comparisons, change only the intended noise parameter(s) and keep the remaining rendering settings unchanged.
