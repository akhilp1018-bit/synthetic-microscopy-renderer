import torch


def add_microscopy_noise_torch(
    vol: torch.Tensor,
    peak_photons: float = 500.0,
    read_noise_std: float = 5.0,
    seed: int | None = 0,
    gaussian_chunk_slices: int = 16,
) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)

    vol = vol.float()

    vol = vol / (vol.max() + 1e-12)
    vol = vol * peak_photons

    noisy = torch.poisson(vol)

    if read_noise_std > 0:
        Z = noisy.shape[0]

        for z0 in range(0, Z, gaussian_chunk_slices):
            z1 = min(z0 + gaussian_chunk_slices, Z)
            chunk = noisy[z0:z1]
            chunk.add_(torch.randn_like(chunk) * read_noise_std)

    noisy.clamp_(min=0.0)

    return noisy


def apply_noise_if_enabled(vol: torch.Tensor, config: dict) -> torch.Tensor:
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