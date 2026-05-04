"""
Sky removal for a single image with thin-branch preservation.

Pipeline
--------
1. Coarse colour-based sky detection in HSV (blue sky + bright clouds),
   restricted to the connected component touching the top edge.
2. Build a trimap:
       FG (non-sky) = coarse non-sky eroded
       BG (sky)     = coarse sky    eroded
       Unknown      = thick boundary band (where branches live)
3. Closed-form alpha matting (PyMatting) over the unknown band — this
   recovers thin tree branches that pure colour thresholds miss.

Outputs (next to the input image):
    <stem>_mask.png  : 8-bit alpha, 255 = non-sky, 0 = sky
    <stem>_nosky.png : RGBA with sky pixels made transparent
    <stem>_trimap.png: trimap used (debug)

Usage:
    python mask.py                       # processes 0001_back.jpg
    python mask.py path/to/image.jpg
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np


def coarse_sky(bgr: np.ndarray) -> np.ndarray:
    """Return uint8 mask where 255 = sky, 0 = non-sky."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    blue_sky = (H >= 90) & (H <= 135) & (S >= 25) & (V >= 110)
    cloud    = (S <= 35) & (V >= 200)
    sky = (blue_sky | cloud).astype(np.uint8) * 255

    # Keep only sky-coloured regions connected to the top edge.
    num, labels = cv2.connectedComponents(sky)
    keep = np.zeros(num, dtype=bool)
    for lbl in np.unique(labels[0, :]):
        if lbl != 0:
            keep[lbl] = True
    return (keep[labels].astype(np.uint8)) * 255


def build_trimap(sky: np.ndarray, band_px: int) -> np.ndarray:
    """255 = FG (non-sky), 0 = BG (sky), 128 = unknown."""
    non_sky = cv2.bitwise_not(sky)
    k = 2 * band_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    fg_sure = cv2.erode(non_sky, kernel, iterations=1)
    bg_sure = cv2.erode(sky,     kernel, iterations=1)

    trimap = np.full_like(sky, 128)
    trimap[fg_sure > 0] = 255
    trimap[bg_sure > 0] = 0
    return trimap


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    img_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "0001_back.jpg")
    if not os.path.isfile(img_path):
        sys.exit(f"ERROR: image not found: {img_path}")

    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        sys.exit(f"ERROR: failed to read image: {img_path}")
    h, w = bgr.shape[:2]

    sky = coarse_sky(bgr)

    band_px = max(8, min(h, w) // 80)
    trimap = build_trimap(sky, band_px=band_px)

    # Closed-form alpha matting recovers thin branches inside the band.
    from pymatting import estimate_alpha_cf
    rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    tri_f = trimap.astype(np.float64) / 255.0
    alpha = estimate_alpha_cf(rgb, tri_f)
    alpha = np.clip(alpha, 0.0, 1.0)
    mask  = (alpha * 255.0 + 0.5).astype(np.uint8)

    stem, _ = os.path.splitext(img_path)
    mask_path   = f"{stem}_mask.png"
    rgba_path   = f"{stem}_nosky.png"
    trimap_path = f"{stem}_trimap.png"

    cv2.imwrite(mask_path,   mask)
    cv2.imwrite(trimap_path, trimap)
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask
    cv2.imwrite(rgba_path, rgba)

    print(f"Saved mask    -> {mask_path}")
    print(f"Saved RGBA    -> {rgba_path}")
    print(f"Saved trimap  -> {trimap_path}")
    print(f"Band width    : {band_px}px")
    print(f"Sky pixels    : {(mask < 16).mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
