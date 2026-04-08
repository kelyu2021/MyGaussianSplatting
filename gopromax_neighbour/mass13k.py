"""
Generate binary masks for cubemap face images using MaSS-Former (MaSS13K).

Masks out specified semantic classes (e.g., person, sky) to produce
COLMAP-compatible masks: 0 = masked (excluded), 255 = valid.

MaSS13K classes:
    0: background
    1: person
    2: building
    3: tree
    4: ground
    5: sky
    6: water

Usage:
    cd /path/to/MaSS13K/mmsegmentation
    conda run -n massformer python /path/to/mass13k.py \
        --image_dir ../gopromax_neighbour/data/cubemap_faces \
        --out_dir ../gopromax_neighbour/data/cubemap_faces_masks \
        --exclude_classes 1 5
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate semantic masks using MaSS-Former (MaSS13K).",
    )
    ap.add_argument("--image_dir", required=True,
                    help="Directory containing input images.")
    ap.add_argument("--out_dir", required=True,
                    help="Output directory for binary masks.")
    ap.add_argument("--exclude_classes", type=int, nargs="+", default=[1, 5],
                    help="Class IDs to mask out (default: 1=person, 5=sky).")
    ap.add_argument("--config",
                    default="configs/massformer/massformer_r50_8xb2-90k_mass13k-1024x1024.py",
                    help="MaSS-Former config file.")
    ap.add_argument("--checkpoint", default="model/iter_80000.pth",
                    help="MaSS-Former checkpoint file.")
    ap.add_argument("--device", default="cuda:0",
                    help="Device for inference.")
    ap.add_argument("--save_overlay", action="store_true",
                    help="Also save overlay visualizations.")
    ap.add_argument("--opacity", type=float, default=0.5,
                    help="Opacity for overlay visualization.")
    args = ap.parse_args()

    # Lazy import so --help works without mmseg installed
    from mmseg.apis import inference_model, init_model

    image_dir = os.path.abspath(args.image_dir)
    out_dir = os.path.abspath(args.out_dir)

    if not os.path.isdir(image_dir):
        sys.exit(f"ERROR: Image directory not found: {image_dir}")

    exts = ("*.png", "*.jpg", "*.jpeg")
    img_paths = sorted([p for ext in exts for p in glob.glob(os.path.join(image_dir, ext))])
    if not img_paths:
        sys.exit(f"ERROR: No images found in {image_dir}")

    os.makedirs(out_dir, exist_ok=True)
    vis_dir = os.path.join(out_dir, "vis")
    if args.save_overlay:
        os.makedirs(vis_dir, exist_ok=True)

    class_names = {0: "background", 1: "person", 2: "building",
                   3: "tree", 4: "ground", 5: "sky", 6: "water"}
    exclude_names = [class_names.get(c, str(c)) for c in args.exclude_classes]
    print(f"Masking out classes: {args.exclude_classes} ({', '.join(exclude_names)})")
    print(f"Processing {len(img_paths)} images from {image_dir}")
    print(f"Output: {out_dir}")

    model = init_model(args.config, args.checkpoint, device=args.device)

    for i, img_path in enumerate(img_paths):
        basename = os.path.basename(img_path)
        result = inference_model(model, img_path)
        pred = result.pred_sem_seg.data.cpu().numpy()[0]  # (H, W)

        # Build binary mask: 255 = valid, 0 = masked out
        mask = np.full(pred.shape, 255, dtype=np.uint8)
        for cls_id in args.exclude_classes:
            mask[pred == cls_id] = 0

        n_masked = (mask == 0).sum()
        n_total = mask.size
        pct = 100.0 * n_masked / n_total

        # Save mask (same filename as input for COLMAP compatibility)
        out_file = os.path.join(out_dir, basename)
        Image.fromarray(mask).save(out_file)

        print(f"  [{i+1}/{len(img_paths)}] {basename} — "
              f"masked {n_masked}/{n_total} pixels ({pct:.1f}%)")

        # Optional overlay
        if args.save_overlay:
            orig = np.array(Image.open(img_path).convert("RGB").resize(
                (pred.shape[1], pred.shape[0])))
            overlay = orig.copy()
            masked_pixels = mask == 0
            overlay[masked_pixels] = (
                orig[masked_pixels] * (1 - args.opacity)
                + np.array([255, 0, 0]) * args.opacity
            ).astype(np.uint8)
            Image.fromarray(overlay).save(os.path.join(vis_dir, basename))

    print(f"\nDone. {len(img_paths)} masks saved to {out_dir}")


if __name__ == "__main__":
    main()
