"""
Bake *_nosky.png (RGBA, soft alpha) into a hard-sky RGB image set.

For each input RGBA image, pixels with alpha >= --threshold (default 200)
are kept as-is, all other pixels are set to a uniform colour (default
black).  The result is saved as RGB PNG/JPG without an alpha channel,
suitable for COLMAP feature extraction and PatchMatch stereo.

Usage
-----
    cd MyGaussianSplatting/gopromax_neighbour_alpha_matting
    conda activate gopro_360
    python bake_nosky_rgb.py \
        --in_dir  data/cubemap_faces_alpha_matting \
        --pattern '*_nosky.png' \
        --out_dir data/cubemap_faces_nosky_rgb \
        --threshold 200
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np


def _sky_color_mask(bgr: np.ndarray, *, cloud: bool,
                    cloud_s_max: int, cloud_v_min: int) -> np.ndarray:
    """Return a bool mask where True = pixel looks like sky.

    The blue-sky band is always on.  The cloud / near-white band is
    optional and configurable, because pale roof tiles or building walls
    can otherwise get caught by it.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    blue_sky = (H >= 90) & (H <= 135) & (S >= 25) & (V >= 110)
    if cloud:
        cloud_mask = (S <= cloud_s_max) & (V >= cloud_v_min)
        return blue_sky | cloud_mask
    return blue_sky


def _sky_flood(alpha: np.ndarray, bgr: np.ndarray, *,
               seed_alpha: int, cloud: bool,
               cloud_s_max: int, cloud_v_min: int) -> np.ndarray:
    """Bool mask of pixels that are 'really sky'.

    A pixel is sky iff it is sky-coloured AND it lies in a connected
    component (4-connectivity over sky-colour | seed) that contains at
    least one seed pixel — i.e. one with alpha < ``seed_alpha``.  This
    keeps pale walls / ground (sky-coloured but isolated from real sky)
    while removing leaked sky islands inside foliage.
    """
    seed = (alpha < seed_alpha)
    sky_color = _sky_color_mask(bgr, cloud=cloud,
                                cloud_s_max=cloud_s_max,
                                cloud_v_min=cloud_v_min)
    region = (seed | sky_color).astype(np.uint8)
    if not region.any():
        return np.zeros_like(seed)

    n, labels = cv2.connectedComponents(region, connectivity=4)
    # Mark labels that contain any seed pixel.
    sky_labels = np.zeros(n, dtype=bool)
    seed_labels = np.unique(labels[seed])
    sky_labels[seed_labels[seed_labels != 0]] = True
    return sky_labels[labels]


def bake_one(in_path: str, out_path: str, threshold: int,
             fill_bgr: tuple[int, int, int], out_ext: str,
             sky_color_guard: bool, seed_alpha: int,
             guard_max_alpha: int, cloud: bool,
             cloud_s_max: int, cloud_v_min: int) -> tuple[str, str | None, float]:
    img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return (os.path.basename(in_path), "failed to read", 0.0)
    if img.ndim != 3 or img.shape[2] != 4:
        return (os.path.basename(in_path), f"expected RGBA, got shape={img.shape}", 0.0)

    bgr = img[:, :, :3].copy()
    alpha = img[:, :, 3]
    keep = alpha >= threshold
    if sky_color_guard:
        sky = _sky_flood(alpha, bgr, seed_alpha=seed_alpha, cloud=cloud,
                         cloud_s_max=cloud_s_max, cloud_v_min=cloud_v_min)
        sky &= (alpha < guard_max_alpha)
        keep &= ~sky
    bgr[~keep] = fill_bgr

    if out_ext == ".jpg":
        ok = cv2.imwrite(out_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        ok = cv2.imwrite(out_path, bgr)
    if not ok:
        return (os.path.basename(in_path), f"failed to write {out_path}", 0.0)
    return (os.path.basename(in_path), None, float((~keep).mean()))


def main() -> None:
    ap = argparse.ArgumentParser(description="Bake *_nosky.png to hard-sky RGB.")
    ap.add_argument("--in_dir",  default="data/cubemap_faces_alpha_matting",
                    help="Directory containing the RGBA mattes.")
    ap.add_argument("--pattern", default="*_nosky.png",
                    help="Glob (relative to --in_dir) selecting RGBA inputs.")
    ap.add_argument("--out_dir", default="data/cubemap_faces_nosky_rgb",
                    help="Output directory for baked RGB images.")
    ap.add_argument("--threshold", type=int, default=30,
                    help="Keep pixels with alpha >= threshold (0-255). "
                         "Lower preserves more thin branches.")
    ap.add_argument("--sky_color_guard", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Drop sky-coloured pixels only if they are in a connected "
                         "component touching a real sky seed (alpha < --seed_alpha). "
                         "Keeps pale walls/ground; removes leaked sky islands. "
                         "Use --no-sky_color_guard to disable.")
    ap.add_argument("--seed_alpha", type=int, default=10,
                    help="Pixels with alpha < this are treated as definite sky seeds "
                         "for the connectivity guard.")
    ap.add_argument("--guard_max_alpha", type=int, default=200,
                    help="The sky-colour guard only removes pixels with alpha < this. "
                         "Opaque pixels (alpha >= guard_max_alpha) are always kept, so "
                         "solid branches survive even if blue-tinted.")
    ap.add_argument("--cloud", action=argparse.BooleanOptionalAction, default=False,
                    help="Include the bright/low-saturation 'cloud' band in the sky "
                         "colour test.  Off by default because pale roofs / walls often "
                         "match it.  Turn on for overcast scenes.")
    ap.add_argument("--cloud_s_max", type=int, default=20,
                    help="Max HSV saturation considered cloud-like (only if --cloud).")
    ap.add_argument("--cloud_v_min", type=int, default=235,
                    help="Min HSV value considered cloud-like (only if --cloud).")
    ap.add_argument("--fill", default="0,0,0",
                    help="Sky fill colour as 'R,G,B' (default black).")
    ap.add_argument("--ext", default="png", choices=["png", "jpg"],
                    help="Output image format.")
    ap.add_argument("--strip_suffix", default="_nosky",
                    help="Suffix to strip from input stem when naming output "
                         "(so 0001_back_nosky.png -> 0001_back.png). "
                         "Pass '' to keep the original stem.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite outputs if they already exist.")
    args = ap.parse_args()

    in_dir  = os.path.abspath(args.in_dir)
    out_dir = os.path.abspath(args.out_dir)
    if not os.path.isdir(in_dir):
        sys.exit(f"ERROR: input directory not found: {in_dir}")
    os.makedirs(out_dir, exist_ok=True)

    try:
        r, g, b = (int(x) for x in args.fill.split(","))
        fill_bgr = (b, g, r)  # OpenCV uses BGR
    except Exception:
        sys.exit(f"ERROR: --fill must be 'R,G,B' integers, got {args.fill!r}")

    inputs = sorted(glob.glob(os.path.join(in_dir, args.pattern)))
    if not inputs:
        sys.exit(f"ERROR: no files match {args.pattern} in {in_dir}")

    out_ext = "." + args.ext
    jobs: list[tuple[str, str]] = []
    for p in inputs:
        stem = os.path.splitext(os.path.basename(p))[0]
        if args.strip_suffix and stem.endswith(args.strip_suffix):
            stem = stem[: -len(args.strip_suffix)]
        outp = os.path.join(out_dir, stem + out_ext)
        if not args.overwrite and os.path.exists(outp):
            continue
        jobs.append((p, outp))

    skipped = len(inputs) - len(jobs)
    print(f"In  : {in_dir}  pattern={args.pattern}  ({len(inputs)} files)")
    print(f"Out : {out_dir}  ext={out_ext}  threshold={args.threshold}  "
          f"fill(BGR)={fill_bgr}  sky_color_guard={args.sky_color_guard}")
    if skipped:
        print(f"Skipping {skipped} existing outputs (use --overwrite).")
    if not jobs:
        print("Nothing to do.")
        return

    t0 = time.time()
    n_err = 0
    sky_ratios: list[float] = []
    for i, (ip, op) in enumerate(jobs, 1):
        name, err, ratio = bake_one(ip, op, args.threshold, fill_bgr, out_ext,
                                    args.sky_color_guard, args.seed_alpha,
                                    args.guard_max_alpha, args.cloud,
                                    args.cloud_s_max, args.cloud_v_min)
        if err:
            n_err += 1
            print(f"  [{i}/{len(jobs)}] {name}  ERROR: {err}")
        else:
            sky_ratios.append(ratio)
            if i % 50 == 0 or i == len(jobs):
                print(f"  [{i}/{len(jobs)}] {name}  sky={ratio*100:.1f}%")

    dt = time.time() - t0
    avg = (sum(sky_ratios) / len(sky_ratios) * 100.0) if sky_ratios else 0.0
    print(f"\nDone: {len(jobs) - n_err} ok, {n_err} failed in {dt:.1f}s.  "
          f"mean sky={avg:.1f}%")


if __name__ == "__main__":
    main()
