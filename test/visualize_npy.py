import numpy as np
import matplotlib.pyplot as plt
import sys
import os

path = sys.argv[1] if len(sys.argv) > 1 else "test/0002_back_depth_raw.npy"
arr = np.load(path)

print(f"shape: {arr.shape}, dtype: {arr.dtype}, min: {arr.min():.4f}, max: {arr.max():.4f}")

valid = arr[arr > 0]
vmin = np.percentile(valid, 2) if valid.size else arr.min()
vmax = np.percentile(valid, 98) if valid.size else arr.max()

plt.figure(figsize=(8, 8))
plt.imshow(arr, cmap="plasma", vmin=vmin, vmax=vmax)
plt.colorbar(label="depth")
plt.title(os.path.basename(path))
plt.axis("off")

out = path.replace(".npy", "_vis.png")
plt.savefig(out, bbox_inches="tight", dpi=150)
print(f"saved: {out}")
plt.show()
