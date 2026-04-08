"""
Generate binary masks using Mask2Former (Cityscapes) to exclude
pedestrians, vehicles, and sky from cubemap face images.

Cityscapes classes (0-indexed in mmseg):
    0: road         5: pole         10: sky          15: bus
    1: sidewalk     6: traffic light 11: person      16: train
    2: building     7: traffic sign  12: rider       17: motorcycle
    3: wall         8: vegetation    13: car         18: bicycle
    4: fence        9: terrain       14: truck

Output masks: 0 = masked (excluded), 255 = valid.

Usage:
    cd MaSS13K/mmsegmentation
    conda run -n massformer python <this_script> \
        --image_dir <input_images> \
        --out_dir <output_masks> \
        --exclude_classes 10 11 12 13 14 15 16 17 18
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image


# Cityscapes class names (0-indexed as used by mmseg)
CITYSCAPES_CLASSES = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
    5: "pole", 6: "traffic light", 7: "traffic sign", 8: "vegetation",
    9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
    14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle",
}

# Default: pedestrians + vehicles + sky
DEFAULT_EXCLUDE = [10, 11, 12, 13, 14, 15, 16, 17, 18]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate semantic masks using Mask2Former (Cityscapes).",
    )
    ap.add_argument("--image_dir", required=True,
                    help="Directory containing input images.")
    ap.add_argument("--out_dir", required=True,
                    help="Output directory for binary masks.")
    ap.add_argument("--exclude_classes", type=int, nargs="+",
                    default=DEFAULT_EXCLUDE,
                    help="Class IDs to mask out (default: sky + person + vehicles).")
    ap.add_argument("--config",
                    default="configs/mask2former/mask2former_swin-b-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024.py",
                    help="Mask2Former config file.")
    ap.add_argument("--checkpoint",
                    default="model/mask2former_swin-b_cityscapes.pth",
                    help="Mask2Former checkpoint file.")
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

    exclude_names = [CITYSCAPES_CLASSES.get(c, str(c)) for c in args.exclude_classes]
    print(f"Masking out classes: {args.exclude_classes}")
    print(f"  ({', '.join(exclude_names)})")
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

        out_file = os.path.join(out_dir, basename)
        Image.fromarray(mask).save(out_file)

        print(f"  [{i+1}/{len(img_paths)}] {basename} — "
              f"masked {n_masked}/{n_total} pixels ({pct:.1f}%)")

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
