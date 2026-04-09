"""
Extract person + vehicle masks from cubemap face images using
Mask2Former (Swin-B) trained on Cityscapes.

Produces binary masks where: 255 = person or vehicle, 0 = other.

Cityscapes classes (19 total, 0-indexed):
    0: road,  1: sidewalk,  2: building,  3: wall,  4: fence,
    5: pole,  6: traffic light,  7: traffic sign,  8: vegetation,
    9: terrain, 10: sky, 11: person, 12: rider,
    13: car,  14: truck,  15: bus,  16: train,  17: motorcycle,  18: bicycle

Usage:
    cd /path/to/MaSS13K/mmsegmentation
    conda run -n massformer python /abs/path/to/MaSSFormer.py
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image

# Cityscapes class indices
PERSON_IDS = [11, 12]            # person, rider
VEHICLE_IDS = [13, 14, 15, 16, 17, 18]  # car, truck, bus, train, motorcycle, bicycle
TARGET_IDS = set(PERSON_IDS + VEHICLE_IDS)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract person+vehicle masks using Mask2Former (Cityscapes).",
    )
    ap.add_argument("--image_dir", required=True,
                    help="Directory containing input images.")
    ap.add_argument("--out_dir", required=True,
                    help="Output directory for masks.")
    ap.add_argument("--config",
                    default="configs/mask2former/mask2former_swin-b-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024.py",
                    help="Mask2Former Cityscapes config file.")
    ap.add_argument("--checkpoint",
                    default="model/mask2former_swin-b_cityscapes.pth",
                    help="Mask2Former Cityscapes checkpoint file.")
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
    img_paths = sorted(
        [p for ext in exts for p in glob.glob(os.path.join(image_dir, ext))]
    )
    if not img_paths:
        sys.exit(f"ERROR: No images found in {image_dir}")

    os.makedirs(out_dir, exist_ok=True)
    vis_dir = os.path.join(out_dir, "vis")
    if args.save_overlay:
        os.makedirs(vis_dir, exist_ok=True)

    target_names = (
        "person, rider, car, truck, bus, train, motorcycle, bicycle"
    )
    print(f"Extracting masks for: {target_names}")
    print(f"Processing {len(img_paths)} images from {image_dir}")
    print(f"Output: {out_dir}")

    model = init_model(args.config, args.checkpoint, device=args.device)

    n_with_target = 0
    for i, img_path in enumerate(img_paths):
        basename = os.path.basename(img_path)
        result = inference_model(model, img_path)
        pred = result.pred_sem_seg.data.cpu().numpy()[0]  # (H, W)

        # Binary mask: 255 = person/vehicle, 0 = other
        mask = np.isin(pred, list(TARGET_IDS)).astype(np.uint8) * 255

        n_target = (mask == 255).sum()
        n_total = mask.size
        pct = 100.0 * n_target / n_total
        if n_target > 0:
            n_with_target += 1

        out_file = os.path.join(out_dir, basename)
        Image.fromarray(mask).save(out_file)

        print(
            f"  [{i+1}/{len(img_paths)}] {basename} — "
            f"person+vehicle {n_target}/{n_total} pixels ({pct:.1f}%)"
        )

        if args.save_overlay:
            orig = np.array(
                Image.open(img_path)
                .convert("RGB")
                .resize((pred.shape[1], pred.shape[0]))
            )
            overlay = orig.copy()
            # Person pixels → red, vehicle pixels → blue
            person_mask = np.isin(pred, PERSON_IDS)
            vehicle_mask = np.isin(pred, VEHICLE_IDS)
            overlay[person_mask] = (
                orig[person_mask] * (1 - args.opacity)
                + np.array([255, 0, 0]) * args.opacity
            ).astype(np.uint8)
            overlay[vehicle_mask] = (
                orig[vehicle_mask] * (1 - args.opacity)
                + np.array([0, 0, 255]) * args.opacity
            ).astype(np.uint8)
            Image.fromarray(overlay).save(os.path.join(vis_dir, basename))

    print(f"\nDone. {len(img_paths)} masks saved to {out_dir}")
    print(
        f"  {n_with_target}/{len(img_paths)} images contain person/vehicle pixels."
    )


if __name__ == "__main__":
    main()
