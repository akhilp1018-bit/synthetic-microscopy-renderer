"""
Noise model utilities for synthetic microscopy images.

This module adds simple microscopy-like noise to a rendered volume.

Noise pipeline:
    1. Normalize clean rendered volume to [0, 1].
    2. Scale by peak photon count.
    3. Apply Poisson shot noise.
    4. Optionally add Gaussian read/background noise.
    5. Clamp negative values to zero.

Supported noise modes:
    none
    poisson
    poisson_gaussian

Coordinate convention:
    Input and output tensors are in ZYX order:
    [Z slices, Y pixels, X pixels].

Note:
    peak_photons is a relative synthetic photon budget because the clean
    rendered volume is normalized by its maximum before photon scaling.
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
    Add Poisson photon noise and optional Gaussian read/background noise.

    Args:
        vol:
            Clean rendered image volume in ZYX order.

        peak_photons:
            Maximum expected photon count after normalizing the clean volume.
            Higher values produce lower relative Poisson noise.

        read_noise_std:
            Standard deviation of additive Gaussian noise in photon-count units.
            Use 0 for Poisson-only noise.

        seed:
            Random seed for reproducible noise.
            Use None for non-deterministic noise.

        gaussian_chunk_slices:
            Number of Z slices processed at once when adding Gaussian noise.

    Returns:
        Noisy image volume as a Torch tensor in ZYX order.
    """

    if peak_photons <= 0:
        raise ValueError("peak_photons must be > 0.")

    if read_noise_std < 0:
        raise ValueError("read_noise_std must be >= 0.")

    if gaussian_chunk_slices <= 0:
        raise ValueError("gaussian_chunk_slices must be > 0.")

    if seed is not None:
        torch.manual_seed(seed)

        if vol.is_cuda:
            torch.cuda.manual_seed_all(seed)

    vol = vol.float()

    max_value = vol.max()

    if max_value <= 0:
        return torch.zeros_like(vol)

    # Normalize clean image to [0, 1].
    normalized = vol / max_value

    # Convert normalized intensity to expected photon counts.
    expected_photons = normalized * peak_photons

    # Poisson shot noise.
    noisy = torch.poisson(expected_photons)

    # Optional additive Gaussian read/background noise.
    if read_noise_std > 0:
        Z = noisy.shape[0]

        for z0 in range(0, Z, gaussian_chunk_slices):
            z1 = min(
                z0 + gaussian_chunk_slices,
                Z,
            )

            chunk = noisy[z0:z1]

            gaussian_noise = (
                torch.randn_like(chunk)
                * read_noise_std
            )

            chunk.add_(gaussian_noise)

    # Photon/read noise cannot produce negative measured counts here.
    noisy.clamp_(min=0.0)

    return noisy


def apply_noise_if_enabled(
    vol: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    """
    Apply microscopy noise according to the configuration.

    Supported modes:

        none
            Return the clean rendered volume unchanged.

        poisson
            Apply photon scaling and Poisson shot noise only.

        poisson_gaussian
            Apply Poisson shot noise followed by additive Gaussian
            read/background noise.

    Example:

        noise:
          enabled: true
          mode: poisson_gaussian
          peak_photons: 500.0
          read_noise_std: 5.0
          seed: 0
          gaussian_chunk_slices: 16
    """

    noise_cfg = config.get(
        "noise",
        {},
    )

    if not noise_cfg.get(
        "enabled",
        False,
    ):
        return vol

    mode = str(
        noise_cfg.get(
            "mode",
            "poisson_gaussian",
        )
    ).lower()

    peak_photons = float(
        noise_cfg.get(
            "peak_photons",
            500.0,
        )
    )

    read_noise_std = float(
        noise_cfg.get(
            "read_noise_std",
            5.0,
        )
    )

    seed = noise_cfg.get(
        "seed",
        0,
    )

    gaussian_chunk_slices = int(
        noise_cfg.get(
            "gaussian_chunk_slices",
            16,
        )
    )

    if mode == "none":
        return vol

    if mode == "poisson":
        read_noise_std = 0.0

    elif mode == "poisson_gaussian":
        pass

    else:
        raise ValueError(
            f"Unknown noise mode: {mode}. "
            "Use 'none', 'poisson', or 'poisson_gaussian'."
        )

    return add_microscopy_noise_torch(
        vol=vol,
        peak_photons=peak_photons,
        read_noise_std=read_noise_std,
        seed=seed,
        gaussian_chunk_slices=gaussian_chunk_slices,
    )