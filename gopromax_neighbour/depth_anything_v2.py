"""
Generate monocular depth maps for cubemap face images using Depth Anything V2.

Sky pixels (mask == 0) are zeroed out in the saved depth maps so they can be
excluded from depth supervision during Gaussian-splatting training.

Outputs
-------
For each input ``NNNN_face.jpg`` two files are written:

* ``NNNN_face_depth_raw.npy``  – float32 (H, W) inverse-depth in model units
* ``NNNN_face_depth_vis.png``  – uint8 colormapped visualisation

Usage
-----
    cd MyGaussianSplatting/gopromax_neighbour
    python depth_anything_v2.py \
        --image_dir  data/cubemap_faces \
        --mask_dir   data/cubemap_faces_mass13k \
        --output_dir data/cubemap_faces_da2
"""

import argparse
import os
import glob

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def main():
    parser = argparse.ArgumentParser(
        description="Depth Anything V2 – cubemap faces with sky mask")
    parser.add_argument(
        "--image_dir", type=str,
        default="data/cubemap_faces",
        help="Directory with cubemap face images (*.jpg)")
    parser.add_argument(
        "--mask_dir", type=str,
        default="data/cubemap_faces_mass13k",
        help="Directory with sky masks (0=sky, 255=valid)")
    parser.add_argument(
        "--output_dir", type=str,
        default="data/cubemap_faces_depth",
        help="Output directory for depth maps")
    parser.add_argument(
        "--model_name", type=str,
        default="depth-anything/Depth-Anything-V2-Large-hf",
        help="HuggingFace model name")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_name} on {device} ...")
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModelForDepthEstimation.from_pretrained(
        args.model_name).to(device)
    model.eval()

    # ── Gather input images (sorted) ─────────────────────────────────
    exts = ("*.png", "*.jpg", "*.jpeg")
    paths = sorted(
        p for ext in exts
        for p in glob.glob(os.path.join(args.image_dir, ext))
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {args.image_dir}")
    print(f"Found {len(paths)} images")

    # ── Inference ─────────────────────────────────────────────────────
    for i, img_path in enumerate(paths):
        stem = os.path.splitext(os.path.basename(img_path))[0]

        pil_img = Image.open(img_path).convert("RGB")
        h, w = pil_img.size[1], pil_img.size[0]

        inputs = processor(images=pil_img, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth  # (1, Hm, Wm)

        # Resize to original resolution
        depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy().astype(np.float32)

        # ── Load sky mask and zero out sky pixels ─────────────────────
        mask_path = os.path.join(args.mask_dir, stem + ".jpg")
        if not os.path.isfile(mask_path):
            # Try .png extension
            mask_path = os.path.join(args.mask_dir, stem + ".png")

        if os.path.isfile(mask_path):
            mask = np.array(Image.open(mask_path).convert("L"))
            # Resize mask if needed
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h),
                                  interpolation=cv2.INTER_NEAREST)
            # 0 = sky → set depth to 0 there
            sky = mask < 128
            depth[sky] = 0.0
        else:
            print(f"  WARNING: no mask found for {stem}, saving unmasked depth")

        # ── Save raw depth (float32 .npy) ─────────────────────────────
        npy_path = os.path.join(args.output_dir, f"{stem}_depth_raw.npy")
        np.save(npy_path, depth)

        # ── Save colormapped visualisation ────────────────────────────
        valid = depth[depth > 0]
        if valid.size > 0:
            d_min, d_max = valid.min(), valid.max()
            depth_norm = np.clip(
                (depth - d_min) / (d_max - d_min + 1e-8), 0.0, 1.0)
        else:
            depth_norm = np.zeros_like(depth)

        # Sky stays black
        depth_norm[depth == 0] = 0.0
        depth_u8 = (depth_norm * 255).astype(np.uint8)

        vis_path = os.path.join(args.output_dir, f"{stem}_depth_vis.png")
        cv2.imwrite(vis_path, depth_u8)

        print(f"  [{i+1}/{len(paths)}] {stem}  "
              f"depth range [{depth[depth > 0].min():.2f}, "
              f"{depth[depth > 0].max():.2f}]"
              if depth[depth > 0].size > 0 else
              f"  [{i+1}/{len(paths)}] {stem}  (all sky)")

    print(f"\nDone. Raw depths saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
