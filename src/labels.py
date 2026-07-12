import torch


def make_object_mask_from_volume(vol: torch.Tensor, rel_threshold: float = 0.1) -> torch.Tensor:
    vmax = float(vol.max().item())

    if vmax <= 0:
        return torch.zeros_like(vol, dtype=torch.float32)

    threshold = rel_threshold * vmax

    return (vol > threshold).to(torch.float32)