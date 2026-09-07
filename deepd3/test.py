import flammkuchen as fl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

path = (
    "outputs/synthetic_dataset_v1/test/"
    "instance_000017/deepd3_predictions/32F.prediction"
)

data = fl.load(path)
spines = np.asarray(data["spines"], dtype=np.float32)

out_dir = Path("debug_spine_slices_000017")
out_dir.mkdir(exist_ok=True)

for z in range(spines.shape[0]):
    plt.figure(figsize=(4, 4))
    plt.imshow(
        spines[z],
        cmap="hot",
        vmin=0,
        vmax=1,
    )
    plt.title(f"Z slice {z}")
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        out_dir / f"slice_{z:03d}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

print(f"Saved {spines.shape[0]} slices to {out_dir}")