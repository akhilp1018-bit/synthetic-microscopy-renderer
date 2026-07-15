"""
Noise model utilities for synthetic microscopy images.

This module can add simple microscopy-like noise to a rendered volume.

Current noise model:
    1. Normalize clean rendered volume to [0, 1].
    2. Scale by peak photon count.
    3. Apply Poisson shot noise.
    4. Add Gaussian read noise.
    5. Clamp negative values to zero.

Coordinate convention:
    Input and output tensors are in ZYX order:
    [Z slices, Y pixels, X pixels].
"""

import torch


def add_microscopy_noise_torch(
    vol: torch.Tensor,
    peak_photons: float = 500.0,
    read_noise_std: float = 5.0,
    seed: int | None = 0,
    gaussian_chunk_slices: int = 16,
) -> torch.Tensor:
    """
    Add Poisson photon noise and Gaussian read noise to a volume.

    Args:
        vol:
            Clean rendered image volume in ZYX order.
        peak_photons:
            Maximum photon count after normalizing the clean volume.
            Higher values give cleaner images.
        read_noise_std:
            Standard deviation of additive Gaussian read noise.
        seed:
            Random seed for reproducible noise. Use None for non-deterministic noise.
        gaussian_chunk_slices:
            Number of Z slices processed at once when adding Gaussian noise.
            This helps reduce memory usage for large volumes.

    Returns:
        Noisy image volume as a Torch tensor in ZYX order.
    """
    if seed is not None:
        torch.manual_seed(seed)

    vol = vol.float()

    # Normalize the clean image and convert it to photon counts.
    vol = vol / (vol.max() + 1e-12)
    vol = vol * peak_photons

    # Photon shot noise.
    noisy = torch.poisson(vol)

    # Add Gaussian read noise in chunks to avoid unnecessary memory spikes.
    if read_noise_std > 0:
        Z = noisy.shape[0]

        for z0 in range(0, Z, gaussian_chunk_slices):
            z1 = min(z0 + gaussian_chunk_slices, Z)
            chunk = noisy[z0:z1]
            chunk.add_(torch.randn_like(chunk) * read_noise_std)

    noisy.clamp_(min=0.0)

    return noisy


def apply_noise_if_enabled(vol: torch.Tensor, config: dict) -> torch.Tensor:
    """
    Apply noise only when enabled in the config.

    Config example:

        noise:
          enabled: true
          peak_photons: 500.0
          read_noise_std: 5.0
          seed: 0
          gaussian_chunk_slices: 16

    If noise.enabled is false, the input volume is returned unchanged.
    """
    noise_cfg = config.get("noise", {})

    if not noise_cfg.get("enabled", False):
        return vol

    return add_microscopy_noise_torch(
        vol,
        peak_photons=float(noise_cfg.get("peak_photons", 500.0)),
        read_noise_std=float(noise_cfg.get("read_noise_std", 5.0)),
        seed=noise_cfg.get("seed", 0),
        gaussian_chunk_slices=int(noise_cfg.get("gaussian_chunk_slices", 16)),
    )