import flammkuchen as fl
import numpy as np
import matplotlib.pyplot as plt

path = (
    "outputs/synthetic_dataset_v1/test/"
    "instance_000017/deepd3_predictions/32F.prediction"
)

data = fl.load(path)

spines = np.asarray(
    data["spines"],
    dtype=np.float32,
)

print("shape:", spines.shape)
print("min:", spines.min())
print("max:", spines.max())

mip = spines.max(axis=0)

plt.imshow(
    mip,
    cmap="hot",
    vmin=0,
    vmax=1,
)

plt.title("RAW DeepD3 spine probability MIP")
plt.axis("off")
plt.show()