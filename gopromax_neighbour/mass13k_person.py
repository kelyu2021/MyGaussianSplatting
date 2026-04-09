"""
Extract person masks from cubemap face images using MaSS-Former (MaSS13K).

Produces binary masks where: 255 = person, 0 = not person.

MaSS13K classes:
    0: background, 1: person, 2: building, 3: tree,
    4: ground, 5: sky, 6: water

Usage:
    cd /path/to/MaSS13K/mmsegmentation
    conda run -n massformer python /abs/path/to/mass13k_person.py \
        --image_dir ../../gopromax_neighbour/data/cubemap_faces \
        --out_dir ../../gopromax_neighbour/data/cubemap_faces_mass13k_human
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image

PERSON_CLASS_ID = 1


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract person masks using MaSS-Former (MaSS13K).",
    )
    ap.add_argument("--image_dir", required=True,
                    help="Directory containing input images.")
    ap.add_argument("--out_dir", required=True,
                    help="Output directory for person masks.")
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

    print(f"Extracting person (class {PERSON_CLASS_ID}) masks")
    print(f"Processing {len(img_paths)} images from {image_dir}")
    print(f"Output: {out_dir}")

    model = init_model(args.config, args.checkpoint, device=args.device)

    n_with_person = 0
    for i, img_path in enumerate(img_paths):
        basename = os.path.basename(img_path)
        result = inference_model(model, img_path)
        pred = result.pred_sem_seg.data.cpu().numpy()[0]  # (H, W)

        # Binary mask: 255 = person, 0 = not person
        mask = np.where(pred == PERSON_CLASS_ID, 255, 0).astype(np.uint8)

        n_person = (mask == 255).sum()
        n_total = mask.size
        pct = 100.0 * n_person / n_total
        if n_person > 0:
            n_with_person += 1

        out_file = os.path.join(out_dir, basename)
        Image.fromarray(mask).save(out_file)

        print(f"  [{i+1}/{len(img_paths)}] {basename} — "
              f"person {n_person}/{n_total} pixels ({pct:.1f}%)")

        if args.save_overlay:
            orig = np.array(Image.open(img_path).convert("RGB").resize(
                (pred.shape[1], pred.shape[0])))
            overlay = orig.copy()
            person_pixels = mask == 255
            overlay[person_pixels] = (
                orig[person_pixels] * (1 - args.opacity)
                + np.array([255, 0, 0]) * args.opacity
            ).astype(np.uint8)
            Image.fromarray(overlay).save(os.path.join(vis_dir, basename))

    print(f"\nDone. {len(img_paths)} masks saved to {out_dir}")
    print(f"  {n_with_person}/{len(img_paths)} images contain person pixels.")


if __name__ == "__main__":
    main()
