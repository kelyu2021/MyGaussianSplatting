"""
Refine sky masks via real alpha matting at the sky/object boundary.

Why
---
The hard sky mask labels a few pixels of trees / antennae / building edges
as "sky".  In training those pixels are supervised toward sky colour and
their Gaussian opacity is pushed to 0, leaving a blue fringe around the
silhouettes in the renders.

A morphological erosion fixes the fringe but eats thin branches.  This
script instead runs a closed-form alpha matting solver (PyMatting) on a
trimap derived from the existing hard mask, producing a per-pixel soft
alpha that follows the actual RGB boundary — branches survive, the
fringe disappears.

Outputs
-------
* ``--out_dir``  : soft alpha mask, ``255 = non-sky, 0 = sky``,
                   smooth values in the boundary band.
                   Drop-in for the existing ``mask_dir`` slot in YAML
                   (note: train.py currently binarises with .bool();
                   that is fine — pixels with alpha > 0 act as non-sky
                   and the fringe pixels are then no longer labelled sky).
* ``--hard_dir`` : optional thresholded copy (255 / 0 only) for code
                   paths that strictly need a hard mask.
* ``<out_dir>/vis/<name>.png`` : overlay if ``--save_overlay`` is set.

Mask convention (matches existing pipeline):
    255 = valid / non-sky
    0   = sky (excluded)

Usage
-----
    cd MyGaussianSplatting/gopromax_neighbour
    python refine_sky_mask.py \\
        --in_dir    data/cubemap_faces_mass13k_manual \\
        --image_dir data/cubemap_faces \\
        --out_dir   data/cubemap_faces_mass13k_manual_matted \\
        --save_overlay
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────
#  Trimap construction
# ─────────────────────────────────────────────────────────────────────
def _build_trimap(mask: np.ndarray,
                  fg_erode_px: int,
                  bg_erode_px: int) -> np.ndarray:
    """Build a trimap from a hard mask.

    Convention:
        255 = definitely foreground (non-sky)
        0   = definitely background (sky)
        128 = unknown (band around the boundary)

    `fg_erode_px` shrinks the non-sky region; the eroded interior is
    "definitely non-sky".  `bg_erode_px` shrinks the sky region likewise.
    Everything else becomes unknown.
    """
    fg = (mask > 127).astype(np.uint8) * 255
    bg = (mask < 128).astype(np.uint8) * 255

    if fg_erode_px > 0:
        k = 2 * fg_erode_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        fg = cv2.erode(fg, kernel, iterations=1)
    if bg_erode_px > 0:
        k = 2 * bg_erode_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bg = cv2.erode(bg, kernel, iterations=1)

    trimap = np.full(mask.shape, 128, dtype=np.uint8)
    trimap[fg > 0] = 255
    trimap[bg > 0] = 0
    return trimap


# ─────────────────────────────────────────────────────────────────────
#  Alpha matting
# ─────────────────────────────────────────────────────────────────────
def _matte(img_bgr: np.ndarray, trimap: np.ndarray,
           method: str, max_size: int) -> np.ndarray:
    """Run alpha matting and return a (H, W) uint8 alpha mask.

    Returns alpha in the same convention as the input mask:
    255 = non-sky (foreground), 0 = sky (background).
    """
    # PyMatting expects RGB float in [0, 1] and trimap in [0, 1].
    H, W = trimap.shape[:2]
    if img_bgr.shape[:2] != (H, W):
        img_bgr = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_AREA)

    # Optional downscale for speed; matting is O(N) memory-heavy.
    scale = 1.0
    if max_size > 0 and max(H, W) > max_size:
        scale = max_size / max(H, W)
        new_w, new_h = int(round(W * scale)), int(round(H * scale))
        img_bgr_s = cv2.resize(img_bgr, (new_w, new_h),
                               interpolation=cv2.INTER_AREA)
        trimap_s = cv2.resize(trimap, (new_w, new_h),
                              interpolation=cv2.INTER_NEAREST)
    else:
        img_bgr_s = img_bgr
        trimap_s = trimap

    img_rgb_f = cv2.cvtColor(img_bgr_s, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    trimap_f = trimap_s.astype(np.float64) / 255.0

    if method == "cf":
        from pymatting import estimate_alpha_cf as _solve
    elif method == "knn":
        from pymatting import estimate_alpha_knn as _solve
    elif method == "lkm":
        from pymatting import estimate_alpha_lkm as _solve
    else:
        raise ValueError(f"Unknown matting method: {method}")

    alpha = _solve(img_rgb_f, trimap_f)        # (H, W) float in [0,1]
    alpha = np.clip(alpha, 0.0, 1.0)

    if scale != 1.0:
        alpha = cv2.resize(alpha.astype(np.float32), (W, H),
                           interpolation=cv2.INTER_LINEAR)

    return (alpha * 255.0 + 0.5).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────────────────────────────
def _save_overlay(img_bgr: np.ndarray, alpha: np.ndarray, out_path: str,
                  opacity: float = 0.5) -> None:
    if img_bgr.shape[:2] != alpha.shape[:2]:
        img_bgr = cv2.resize(img_bgr, (alpha.shape[1], alpha.shape[0]),
                             interpolation=cv2.INTER_AREA)
    a = alpha.astype(np.float32) / 255.0           # 1 = non-sky
    sky_w = (1.0 - a)[..., None]                   # 1 = sky
    red = np.array([0, 0, 255], dtype=np.float32)  # BGR red
    overlay = img_bgr.astype(np.float32) * (1 - sky_w * opacity) \
        + red * (sky_w * opacity)
    cv2.imwrite(out_path, np.clip(overlay, 0, 255).astype(np.uint8))


# ─────────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Refine hard sky masks with real alpha matting "
                    "(PyMatting) so the boundary follows actual RGB edges.")
    ap.add_argument("--in_dir", required=True,
                    help="Directory of input hard masks (255=non-sky, 0=sky).")
    ap.add_argument("--image_dir", required=True,
                    help="Directory of corresponding RGB images "
                         "(same basenames; .jpg/.png).")
    ap.add_argument("--out_dir", required=True,
                    help="Output directory for SOFT alpha masks.")
    ap.add_argument("--hard_dir", default=None,
                    help="Optional: also write a thresholded hard copy here.")
    ap.add_argument("--hard_thr", type=int, default=128,
                    help="Threshold for hard copy (default 128).")

    ap.add_argument("--fg_erode_px", type=int, default=4,
                    help="Erosion of the non-sky region for the trimap "
                         "(default 4).  Larger = wider unknown band.")
    ap.add_argument("--bg_erode_px", type=int, default=8,
                    help="Erosion of the sky region for the trimap "
                         "(default 8).  Larger = matter has more freedom "
                         "near the boundary.")
    ap.add_argument("--method", choices=("cf", "knn", "lkm"), default="cf",
                    help="PyMatting solver: cf (closed-form, default), "
                         "knn (faster, slightly worse), lkm (large-kernel).")
    ap.add_argument("--max_size", type=int, default=1024,
                    help="Downscale longest side to this for matting "
                         "(default 1024; 0 = no downscale).  Alpha is then "
                         "upsampled back to original size.")

    ap.add_argument("--save_overlay", action="store_true",
                    help="Write overlays into <out_dir>/vis/.")
    args = ap.parse_args()

    in_dir = os.path.abspath(args.in_dir)
    img_dir = os.path.abspath(args.image_dir)
    out_dir = os.path.abspath(args.out_dir)
    hard_dir = os.path.abspath(args.hard_dir) if args.hard_dir else None

    if not os.path.isdir(in_dir):
        sys.exit(f"ERROR: Input mask directory not found: {in_dir}")
    if not os.path.isdir(img_dir):
        sys.exit(f"ERROR: Image directory not found: {img_dir}")

    exts = ("*.png", "*.jpg", "*.jpeg")
    paths = sorted([p for ext in exts
                    for p in glob.glob(os.path.join(in_dir, ext))])
    if not paths:
        sys.exit(f"ERROR: No masks found in {in_dir}")

    os.makedirs(out_dir, exist_ok=True)
    if hard_dir:
        os.makedirs(hard_dir, exist_ok=True)
    vis_dir = os.path.join(out_dir, "vis") if args.save_overlay else None
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    print(f"[matte] {len(paths)} masks  in='{in_dir}'  out='{out_dir}'")
    print(f"[matte] method={args.method}  "
          f"fg_erode={args.fg_erode_px}  bg_erode={args.bg_erode_px}  "
          f"max_size={args.max_size}")

    for i, p in enumerate(paths):
        basename = os.path.basename(p)
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            print(f"  [{i+1}/{len(paths)}] {basename}: unreadable, skip")
            continue

        # Find matching RGB image
        stem = os.path.splitext(basename)[0]
        img_path = None
        for ext in (".jpg", ".png", ".jpeg"):
            cand = os.path.join(img_dir, stem + ext)
            if os.path.exists(cand):
                img_path = cand
                break
        if img_path is None:
            print(f"  [{i+1}/{len(paths)}] {basename}: no RGB found, skip")
            continue
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"  [{i+1}/{len(paths)}] {basename}: RGB unreadable, skip")
            continue
        if img_bgr.shape[:2] != m.shape[:2]:
            img_bgr = cv2.resize(img_bgr, (m.shape[1], m.shape[0]),
                                 interpolation=cv2.INTER_AREA)

        trimap = _build_trimap(m, args.fg_erode_px, args.bg_erode_px)

        # Skip matting if the trimap has no unknown band (e.g. all-sky face).
        if (trimap == 128).sum() == 0:
            alpha = m.copy()
        else:
            try:
                alpha = _matte(img_bgr, trimap,
                               method=args.method, max_size=args.max_size)
            except Exception as e:
                print(f"  [{i+1}/{len(paths)}] {basename}: matting failed "
                      f"({e}), falling back to input mask")
                alpha = m.copy()

        cv2.imwrite(os.path.join(out_dir, basename), alpha)

        if hard_dir:
            hard = np.where(alpha >= args.hard_thr, 255, 0).astype(np.uint8)
            cv2.imwrite(os.path.join(hard_dir, basename), hard)

        if vis_dir:
            _save_overlay(img_bgr, alpha,
                          os.path.join(vis_dir, basename))

        sky_before = int((m < 128).sum())
        sky_after_soft = float((255 - alpha).sum()) / 255.0
        pct_before = 100.0 * sky_before / m.size
        pct_after = 100.0 * sky_after_soft / m.size
        print(f"  [{i+1}/{len(paths)}] {basename}: "
              f"sky {pct_before:5.1f}% → {pct_after:5.1f}% (soft)")

    print(f"\n[matte] Done. Soft alpha masks in {out_dir}")
    if hard_dir:
        print(f"[matte] Hard copies in {hard_dir}")


if __name__ == "__main__":
    main()
