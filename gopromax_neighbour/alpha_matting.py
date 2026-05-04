"""
Batch sky-removal alpha mattes for cubemap face images.

Generates a soft alpha mask (255 = non-sky, 0 = sky) for every image in
``--in_dir`` using the same pipeline as ``test/mask.py``:

    1. Coarse HSV sky detection, kept only where connected to the top
       edge of the image.
    2. Trimap with FG (non-sky) / BG (sky) eroded, leaving an unknown
       band that covers thin tree branches.
    3. Closed-form alpha matting (PyMatting) inside the unknown band.

Outputs are written to ``--out_dir`` with the same basenames as the
inputs but as PNG, plus an optional RGBA copy and trimap for debug.

Usage
-----
    cd MyGaussianSplatting/gopromax_neighbour
    conda activate gopro_360
    python alpha_matting.py \
        --in_dir  data/cubemap_faces \
        --out_dir data/cubemap_faces_alpha_matting

    # also save RGBA + trimap for inspection
    python alpha_matting.py --save_rgba --save_trimap
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────
def process_one(args_tuple) -> tuple[str, str | None, float]:
    """Worker: compute alpha matte for a single image. Returns (name, error, sky_ratio)."""
    (img_path, out_dir, band_div, save_rgba, save_trimap) = args_tuple
    try:
        from pymatting import estimate_alpha_cf

        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            return (os.path.basename(img_path), "failed to read", 0.0)
        h, w = bgr.shape[:2]

        sky = coarse_sky(bgr)

        if sky.any() and (~sky.astype(bool)).any():
            band_px = max(8, min(h, w) // band_div)
            trimap = build_trimap(sky, band_px=band_px)
            rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
            tri_f = trimap.astype(np.float64) / 255.0
            alpha = np.clip(estimate_alpha_cf(rgb, tri_f), 0.0, 1.0)
            mask  = (alpha * 255.0 + 0.5).astype(np.uint8)
        else:
            # All sky or no sky → skip matting (avoids singular matrix).
            trimap = np.full_like(sky, 128)
            mask = cv2.bitwise_not(sky)

        stem = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(os.path.join(out_dir, f"{stem}_mask.png"), mask)
        if save_trimap:
            cv2.imwrite(os.path.join(out_dir, f"{stem}_trimap.png"), trimap)
        if save_rgba:
            rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = mask
            cv2.imwrite(os.path.join(out_dir, f"{stem}_nosky.png"), rgba)

        sky_ratio = float((mask < 16).mean())
        return (os.path.basename(img_path), None, sky_ratio)
    except Exception as e:  # noqa: BLE001
        return (os.path.basename(img_path), repr(e), 0.0)


# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Batch sky alpha matting.")
    ap.add_argument("--in_dir",  default="data/cubemap_faces",
                    help="Input image directory.")
    ap.add_argument("--out_dir", default="data/cubemap_faces_alpha_matting",
                    help="Output mask directory.")
    ap.add_argument("--band_div", type=int, default=80,
                    help="Unknown-band width = min(H,W)//band_div (smaller = wider band).")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2),
                    help="Parallel worker processes.")
    ap.add_argument("--save_rgba",   action="store_true", help="Also save RGBA cutouts.")
    ap.add_argument("--save_trimap", action="store_true", help="Also save trimaps.")
    ap.add_argument("--overwrite",   action="store_true", help="Recompute even if mask exists.")
    args = ap.parse_args()

    in_dir  = os.path.abspath(args.in_dir)
    out_dir = os.path.abspath(args.out_dir)
    if not os.path.isdir(in_dir):
        sys.exit(f"ERROR: input directory not found: {in_dir}")

    exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    img_paths = sorted({p for ext in exts for p in glob.glob(os.path.join(in_dir, ext))})
    if not img_paths:
        sys.exit(f"ERROR: no images found in {in_dir}")

    os.makedirs(out_dir, exist_ok=True)

    if not args.overwrite:
        before = len(img_paths)
        img_paths = [p for p in img_paths
                     if not os.path.exists(os.path.join(
                         out_dir, os.path.splitext(os.path.basename(p))[0] + "_mask.png"))]
        skipped = before - len(img_paths)
        if skipped:
            print(f"Skipping {skipped} images that already have a mask "
                  f"(use --overwrite to recompute).")

    if not img_paths:
        print("Nothing to do.")
        return

    jobs = [(p, out_dir, args.band_div, args.save_rgba, args.save_trimap)
            for p in img_paths]

    print(f"Input : {in_dir}")
    print(f"Output: {out_dir}")
    print(f"Images: {len(jobs)}   workers: {args.workers}")

    t0 = time.time()
    n_done = n_err = 0
    sky_ratios: list[float] = []

    if args.workers <= 1:
        for j in jobs:
            name, err, ratio = process_one(j)
            n_done += 1
            if err:
                n_err += 1
                print(f"  [{n_done}/{len(jobs)}] {name}  ERROR: {err}")
            else:
                sky_ratios.append(ratio)
                if n_done % 20 == 0 or n_done == len(jobs):
                    print(f"  [{n_done}/{len(jobs)}] {name}  sky={ratio*100:.1f}%")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(process_one, j) for j in jobs]
            for fut in as_completed(futures):
                name, err, ratio = fut.result()
                n_done += 1
                if err:
                    n_err += 1
                    print(f"  [{n_done}/{len(jobs)}] {name}  ERROR: {err}")
                else:
                    sky_ratios.append(ratio)
                    if n_done % 20 == 0 or n_done == len(jobs):
                        print(f"  [{n_done}/{len(jobs)}] {name}  sky={ratio*100:.1f}%")

    dt = time.time() - t0
    avg_sky = (sum(sky_ratios) / len(sky_ratios) * 100.0) if sky_ratios else 0.0
    print(f"\nDone: {n_done - n_err} ok, {n_err} failed in {dt:.1f}s "
          f"({dt / max(1, n_done):.2f}s/img).  mean sky={avg_sky:.1f}%")


if __name__ == "__main__":
    main()
