#!/usr/bin/env python3
"""
Combine roof (depth-based) and sky (MaSS13K) masks into a single mask.

Both input masks use the convention: 0 = masked, 255 = valid.
The combined mask marks a pixel as masked (0) if EITHER input mask
marks it as masked; valid (255) only if both agree.

Inputs
------
  --roof_dir   Roof masks   (default: output/masks_roof_depth)
  --sky_dir    Sky masks    (default: output/masks_sky_mass13k)

Outputs
-------
  <output_dir>/<image_name>.png
    Binary mask: 0 = masked (sky or roof), 255 = valid.

  <output_dir>/vis/<image_name>.png
    Original image with masked regions tinted red.

Usage
-----
  cd MyGaussianSplatting/gopro360/colmap
  python combine_roof_depth_sky_mas13k.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_ROOF   = str(_SCRIPT_DIR / "output" / "masks_roof_depth")
DEFAULT_SKY    = str(_SCRIPT_DIR / "output" / "masks_sky_mass13k")
DEFAULT_IMAGES = str(_SCRIPT_DIR / "output" / "images")
DEFAULT_OUTPUT = str(_SCRIPT_DIR / "output" / "masks_roof_depth_sky_mass13k")


def visualize_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vis = image_bgr.copy()
    invalid = mask == 0
    vis[invalid] = (vis[invalid] * 0.4 + np.array([0, 0, 200]) * 0.6).astype(np.uint8)
    return vis


def main():
    parser = argparse.ArgumentParser(
        description="Combine roof and sky masks into a single mask")
    parser.add_argument("--roof_dir",   type=str, default=DEFAULT_ROOF)
    parser.add_argument("--sky_dir",    type=str, default=DEFAULT_SKY)
    parser.add_argument("--images_dir", type=str, default=DEFAULT_IMAGES)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    roof_dir   = Path(args.roof_dir)
    sky_dir    = Path(args.sky_dir)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vis_dir = output_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Use sky masks as the reference file list
    sky_paths = sorted(sky_dir.glob("*.png"))
    if not sky_paths:
        print(f"[combine] No PNG masks found in {sky_dir}")
        return

    print(f"[combine] Combining {len(sky_paths)} masks …")
    stats = {"roof_only": 0, "sky_only": 0, "both": 0}

    for sky_path in tqdm(sky_paths, desc="Combining", unit="img"):
        name = sky_path.name
        roof_path = roof_dir / name

        sky_mask = cv2.imread(str(sky_path), cv2.IMREAD_GRAYSCALE)
        if sky_mask is None:
            tqdm.write(f"[combine] WARNING: cannot read {sky_path}, skipping")
            continue

        if roof_path.exists():
            roof_mask = cv2.imread(str(roof_path), cv2.IMREAD_GRAYSCALE)
        else:
            roof_mask = np.full_like(sky_mask, 255)

        # Sky mask convention:  255 = sky (to mask out), 0 = non-sky
        # Roof mask convention: 0 = roof (masked),     255 = valid
        # Combined: 0 = masked (sky or roof), 255 = valid
        is_sky  = sky_mask == 255
        is_roof = roof_mask == 0
        combined = np.where(is_sky | is_roof, 0, 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / name), combined)

        # Visualisation
        img_path = images_dir / name
        if img_path.exists():
            image_bgr = cv2.imread(str(img_path))
            cv2.imwrite(str(vis_dir / name), visualize_mask(image_bgr, combined))

    print(f"[combine] Combined masks saved to {output_dir}")
    print(f"[combine] Visualisations saved to {vis_dir}")


if __name__ == "__main__":
    main()
