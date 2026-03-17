"""
Generate depth maps for cubemap face images using Depth Anything V2.

Reads every image from  gopro360/colmap/output/images/
and writes a raw float32 .npy depth map (plus a colourised .png preview)
to  gopro360/colmap/output/depth/.

Usage
-----
    cd MyGaussianSplatting/gopro360/colmap
    python depthAnythingV2.py                          # defaults
    python depthAnythingV2.py --input_dir  <path>      # override
                              --output_dir <path>
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

_SCRIPT_DIR = Path(__file__).resolve().parent  # gopro360/colmap/

DEFAULT_INPUT  = str(_SCRIPT_DIR / "output" / "images")
DEFAULT_OUTPUT = str(_SCRIPT_DIR / "output" / "depth")


def generate_depth_maps(input_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "depth-anything/Depth-Anything-V2-Large-hf"
    print(f"Loading {model_name} on {device} ...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device)
    model.eval()

    # ── Gather input images (sorted) ──────────────────────────────────
    exts = ("*.png", "*.jpg", "*.jpeg")
    paths = sorted(
        p for ext in exts for p in glob.glob(os.path.join(input_dir, ext))
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {input_dir}")
    print(f"Found {len(paths)} images in {input_dir}")

    # ── Inference loop ────────────────────────────────────────────────
    for i, img_path in enumerate(paths):
        stem = Path(img_path).stem  # e.g. frame_000000_front

        pil_img = Image.open(img_path).convert("RGB")
        inputs = processor(images=pil_img, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth  # (1, H', W')

        # Interpolate to original resolution
        h, w = pil_img.size[1], pil_img.size[0]
        depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

        # Save raw float32 depth (metric-relative) as .npy
        npy_path = os.path.join(output_dir, f"{stem}.npy")
        np.save(npy_path, depth.astype(np.float32))

        # Save colourised preview as .png
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_u8 = (depth_norm * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
        png_path = os.path.join(output_dir, f"{stem}.png")
        cv2.imwrite(png_path, depth_color)

        print(f"  [{i + 1}/{len(paths)}] {stem}  "
              f"depth range [{depth.min():.2f}, {depth.max():.2f}]")

    print(f"\nDone – saved {len(paths)} depth maps to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate depth maps with Depth Anything V2"
    )
    parser.add_argument("--input_dir",  default=DEFAULT_INPUT,
                        help="Directory of input images")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT,
                        help="Directory for output depth maps")
    args = parser.parse_args()

    generate_depth_maps(args.input_dir, args.output_dir)
