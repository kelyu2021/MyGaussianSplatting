#!/usr/bin/env python3
"""
Mask out **sky** and **car-roof** regions from perspective cubemap images.

  - **Sky**: per-image SegFormer (ADE20K class 2).
  - **Roof**: per-image Depth Anything V2.  The car roof is the closest
    surface to the camera, so it has the highest inverse-depth values.
    Otsu thresholding (scaled by --depth_scale) separates the near-field
    roof from the far-field scene.  Only the connected region touching
    the bottom image edge is kept.

Outputs
-------
  <output_dir>/<image_name>.png
    Binary mask: 0 = masked (sky or roof), 255 = valid region.

Usage
-----
  python maskSkyRoof_depth.py --images_dir output/images --output_dir output/masks --visualize --device cuda:4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

ADE20K_SKY = 2


# ── Model loading ─────────────────────────────────────────────────────

def load_segformer(device: torch.device):
    model_name = "nvidia/segformer-b5-finetuned-ade-640-640"
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.to(device).eval()
    return processor, model


def load_depth_model(device: torch.device):
    model_name = "depth-anything/Depth-Anything-V2-Large-hf"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model.to(device).eval()
    return processor, model


# ── Inference helpers ─────────────────────────────────────────────────

@torch.no_grad()
def predict_sky(image_bgr, seg_processor, seg_model, device):
    """Return boolean sky mask from SegFormer."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    inputs = seg_processor(images=image_rgb, return_tensors="pt").to(device)
    logits = seg_model(**inputs).logits
    h, w = image_bgr.shape[:2]
    up = torch.nn.functional.interpolate(
        logits, size=(h, w), mode="bilinear", align_corners=False,
    )
    seg_map = up.argmax(dim=1).squeeze(0).cpu().numpy()
    return seg_map == ADE20K_SKY


@torch.no_grad()
def predict_depth(image_bgr, depth_processor, depth_model, device):
    """Return inverse-depth map (higher = closer to camera)."""
    pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    inputs = depth_processor(images=pil_img, return_tensors="pt").to(device)
    depth = depth_model(**inputs).predicted_depth
    h, w = image_bgr.shape[:2]
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False,
    ).squeeze().cpu().numpy()
    return depth


# ── Utilities ─────────────────────────────────────────────────────────

def _keep_bottom_connected(mask: np.ndarray) -> np.ndarray:
    """Keep only connected components that touch the bottom edge."""
    if not mask.any():
        return mask
    labels_uint8 = mask.astype(np.uint8)
    _, label_map = cv2.connectedComponents(labels_uint8, connectivity=8)
    bottom_labels = set(label_map[-1, :][label_map[-1, :] > 0])
    if not bottom_labels:
        return np.zeros_like(mask)
    out = np.zeros_like(mask)
    for lbl in bottom_labels:
        out |= label_map == lbl
    return out


def depth_to_roof_mask(depth_map, depth_scale):
    """Threshold inverse-depth to get roof mask (near pixels)."""
    d_min, d_max = depth_map.min(), depth_map.max()
    d_norm = ((depth_map - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(d_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    effective_thresh = int(otsu_val * depth_scale)
    roof_mask = d_norm > effective_thresh
    roof_mask = _keep_bottom_connected(roof_mask)
    return roof_mask


def visualize_mask(image_bgr, mask):
    vis = image_bgr.copy()
    invalid = mask == 0
    vis[invalid] = (vis[invalid] * 0.4 + np.array([0, 0, 200]) * 0.6).astype(np.uint8)
    return vis


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Mask sky + car-roof (depth-based)")
    parser.add_argument("--images_dir", type=str, default="output/images")
    parser.add_argument("--output_dir", type=str, default="output/masks")
    parser.add_argument("--depth_scale", type=float, default=0.7,
                        help="Scale factor for Otsu depth threshold (default: 0.7)")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.visualize:
        vis_dir = output_dir / "vis"
        vis_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mask] Using device: {device}")

    print("[mask] Loading SegFormer-B5 (ADE20K) …")
    seg_processor, seg_model = load_segformer(device)
    print("[mask] Loading Depth Anything V2 Large …")
    depth_processor, depth_model = load_depth_model(device)

    image_paths = sorted(images_dir.glob("*.png"))
    if not image_paths:
        print(f"[mask] No PNG images found in {images_dir}")
        return

    print(f"[mask] Processing {len(image_paths)} images (per-frame sky + depth roof) …")
    roof_pcts: list[float] = []

    for img_path in tqdm(image_paths, desc="Masking", unit="img"):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            tqdm.write(f"[mask] WARNING: cannot read {img_path}, skipping")
            continue

        # Sky mask (SegFormer)
        sky_mask = predict_sky(image_bgr, seg_processor, seg_model, device)

        # Roof mask (Depth Anything V2 — near pixels)
        depth_map = predict_depth(image_bgr, depth_processor, depth_model, device)
        roof_mask = depth_to_roof_mask(depth_map, args.depth_scale)
        roof_pcts.append(100 * roof_mask.sum() / roof_mask.size)

        # Combine: 0 = masked, 255 = valid
        out = np.where(sky_mask | roof_mask, 0, 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / img_path.name), out)

        if args.visualize:
            cv2.imwrite(str(vis_dir / img_path.name), visualize_mask(image_bgr, out))

    if roof_pcts:
        print(f"  Roof coverage: mean={np.mean(roof_pcts):.1f}%, "
              f"min={np.min(roof_pcts):.1f}%, max={np.max(roof_pcts):.1f}%")

    del depth_model, depth_processor, seg_model, seg_processor
    torch.cuda.empty_cache()

    print(f"[mask] Masks saved to {output_dir}")
    if args.visualize:
        print(f"[mask] Visualisations saved to {vis_dir}")


if __name__ == "__main__":
    main()
