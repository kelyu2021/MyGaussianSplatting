"""
Generate depth maps for cubemap face images using Depth Anything V3.

Reads every image from  gopro360/colmap/output/images/
and writes a raw float32 .npy depth map (plus a grayscale .png)
to  gopro360/colmap/output/depth_anything_v3/.

Usage
-----
    cd MyGaussianSplatting/gopro360/colmap
    python depthAnythingV3.py                          # defaults
    python depthAnythingV3.py --input_dir  <path>      # override
                              --output_dir <path>
                              --model_name da3-large
                              --device cuda:0

Model names can be HuggingFace repo IDs (e.g. depth-anything/DA3NESTED-GIANT-LARGE)
or local preset names (da3-large, da3-giant, etc.).  Use from_pretrained repo IDs
to load actual pretrained weights.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Add Depth-Anything-3 source to path
_SCRIPT_DIR = Path(__file__).resolve().parent  # gopro360/colmap/
_DA3_ROOT = _SCRIPT_DIR.parent.parent / "Depth-Anything-3" / "src"
sys.path.insert(0, str(_DA3_ROOT))

from depth_anything_3.api import DepthAnything3  # noqa: E402

DEFAULT_INPUT  = str(_SCRIPT_DIR / "output" / "images")
DEFAULT_OUTPUT = str(_SCRIPT_DIR / "output" / "depth_anything_v3")


def generate_depth_maps(input_dir: str, output_dir: str,
                        model_name: str, device: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # ── Gather input images (sorted) ──────────────────────────────────
    exts = ("*.png", "*.jpg", "*.jpeg")
    paths = sorted(
        str(p)
        for ext in exts
        for p in Path(input_dir).glob(ext)
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {input_dir}")
    print(f"Found {len(paths)} images in {input_dir}")

    # ── Load model ────────────────────────────────────────────────────
    print(f"Loading DepthAnything3 ({model_name}) on {device} ...")
    model = DepthAnything3.from_pretrained(model_name).to(device)
    model.eval()

    # ── Inference (one image at a time to avoid multi-view mode) ─────
    total = len(paths)
    for idx, img_path in enumerate(paths, 1):
        stem = Path(img_path).stem

        prediction = model.inference(
            image=[img_path],
            process_res=504,
        )

        # prediction.depth: (1, H, W) numpy array
        depth = prediction.depth[0]  # H, W

        # Resize to original resolution if needed
        orig = cv2.imread(img_path)
        if orig is not None:
            h_orig, w_orig = orig.shape[:2]
            if depth.shape != (h_orig, w_orig):
                depth = cv2.resize(depth, (w_orig, h_orig),
                                   interpolation=cv2.INTER_LINEAR)

        # Save raw float32 depth as .npy
        npy_path = os.path.join(output_dir, f"{stem}.npy")
        np.save(npy_path, depth.astype(np.float32))

        # Save grayscale depth as .png (inverse depth: closer=brighter)
        inv_depth = 1.0 / (depth + 1e-8)
        d_min, d_max = inv_depth.min(), inv_depth.max()
        depth_norm = (inv_depth - d_min) / (d_max - d_min + 1e-8)
        depth_u8 = (depth_norm * 255).astype(np.uint8)
        png_path = os.path.join(output_dir, f"{stem}.png")
        cv2.imwrite(png_path, depth_u8)

        # Save colorized depth as .png (INFERNO colormap)
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
        color_path = os.path.join(output_dir, f"{stem}_color.png")
        cv2.imwrite(color_path, depth_color)

        print(f"  [{idx}/{total}] {stem}  "
              f"depth range [{depth.min():.2f}, {depth.max():.2f}]")

    print(f"\nDone – saved {total} depth maps to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate depth maps with Depth Anything V3"
    )
    parser.add_argument("--input_dir",  default=DEFAULT_INPUT,
                        help="Directory of input images")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT,
                        help="Directory for output depth maps")
    parser.add_argument("--model_name", default="depth-anything/DA3NESTED-GIANT-LARGE",
                        help="HuggingFace model ID (e.g. depth-anything/DA3NESTED-GIANT-LARGE)")
    parser.add_argument("--device",     default="cuda:0",
                        help="Device to run on (cuda:0, cpu, etc.)")
    args = parser.parse_args()

    generate_depth_maps(args.input_dir, args.output_dir,
                        args.model_name, args.device)
