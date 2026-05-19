import numpy as np
import cv2
import glob

png_files = sorted(glob.glob("data/cubemap_faces_da2_png/*.png"))
print(f"Found {len(png_files)} PNG files")

for p in png_files[:3]:
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"{p}: failed to load")
        continue
    print(f"\n{p}")
    print(f"  dtype:        {img.dtype}")
    print(f"  shape:        {img.shape}")
    print(f"  min/max:      {img.min()} / {img.max()}")
    print(f"  non-zero px:  {(img > 0).sum()} / {img.size}")
