#!/usr/bin/env python3
"""
Mask out the **car-roof** region from perspective cubemap images using
pre-computed Depth Anything V2 depth maps.

The GoPro 360 is mounted on the car roof, so the roof is the nearest
surface to the camera — it has the highest inverse-depth values in each
frame.  Otsu thresholding (scaled by --depth_scale) separates the
near-field roof from the far-field scene.  Only the connected component
touching the bottom image edge is kept to avoid stray near-field noise.

Inputs
------
  --images_dir   RGB cubemap face images   (default: output/images)
  --depth_dir    Pre-computed depth .npy    (default: output/depth)

Outputs
-------
  <output_dir>/<image_name>.png
    Binary mask: 0 = masked (roof), 255 = valid region.

  <output_dir>/vis/<image_name>.png   (with --visualize)
    Original image with masked regions tinted red.

Usage
-----
  cd MyGaussianSplatting/gopro360/colmap
  python maskRoofBasedOnDepth.py
  python maskRoofBasedOnDepth.py --visualize --device cuda:0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent  # gopro360/colmap/

DEFAULT_IMAGES = str(_SCRIPT_DIR / "output" / "images")
DEFAULT_DEPTH  = str(_SCRIPT_DIR / "output" / "depth")
DEFAULT_OUTPUT = str(_SCRIPT_DIR / "output" / "masks_roof")

# Per-face Otsu scale factors (tuned visually)
FACE_NAMES = ("front", "right", "back", "left")
FACE_DEPTH_SCALE = {
    "front": 0.5,
    "back":  1.0,
    "left":  0.7,
    "right": 0.5,
}


# ── Depth-based roof detection ───────────────────────────────────────

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


def depth_to_roof_mask(depth_map: np.ndarray, depth_scale: float,
                       erode_px: int = 0) -> np.ndarray:
    """Threshold inverse-depth to get roof mask (near pixels).

    Higher depth values = closer to camera = roof.
    Otsu threshold is scaled by depth_scale — higher values give a
    tighter (smaller) mask; lower values give a looser (larger) mask.
    An optional erosion step (erode_px) further shrinks the mask to
    remove boundary over-estimation.
    """
    d_min, d_max = depth_map.min(), depth_map.max()
    d_norm = ((depth_map - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(d_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    effective_thresh = int(otsu_val * depth_scale)
    roof_mask = d_norm > effective_thresh
    roof_mask = _keep_bottom_connected(roof_mask)

    # Erode to trim boundary over-estimation
    if erode_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (2 * erode_px + 1, 2 * erode_px + 1))
        roof_mask = cv2.erode(roof_mask.astype(np.uint8), kernel).astype(bool)

    return roof_mask


# ── Visualisation ─────────────────────────────────────────────────────

def visualize_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay red tint on masked (invalid = 0) regions."""
    vis = image_bgr.copy()
    invalid = mask == 0
    vis[invalid] = (vis[invalid] * 0.4 + np.array([0, 0, 200]) * 0.6).astype(np.uint8)
    return vis


# ── Utilities ─────────────────────────────────────────────────────────

def face_name_from_path(img_path: Path) -> str:
    """Extract face direction from filename like frame_000000_back.png."""
    stem = img_path.stem
    for face in FACE_NAMES:
        if stem.endswith(f"_{face}"):
            return face
    return "unknown"


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Mask car-roof using pre-computed depth maps")
    parser.add_argument("--images_dir",  type=str, default=DEFAULT_IMAGES)
    parser.add_argument("--depth_dir",   type=str, default=DEFAULT_DEPTH)
    parser.add_argument("--output_dir",  type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--depth_scale", type=float, default=None,
                        help="Override depth scale for ALL faces. "
                             "If not set, uses per-face defaults: "
                             "front=0.5, back=1.0, left=0.7, right=0.5.")
    parser.add_argument("--erode_px",    type=int, default=0,
                        help="Erosion radius in pixels to shrink mask boundary (default: 0). "
                             "Set 0 to disable.")
    parser.add_argument("--visualize",   action="store_true")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    depth_dir  = Path(args.depth_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vis_dir = output_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # ── Gather images ─────────────────────────────────────────────────
    image_paths = sorted(images_dir.glob("*.png"))
    if not image_paths:
        print(f"[mask] No PNG images found in {images_dir}")
        return

    # Per-face scale map
    if args.depth_scale is not None:
        scale_map = {f: args.depth_scale for f in FACE_NAMES}
        print(f"[mask] Using uniform depth_scale={args.depth_scale} for all faces")
    else:
        scale_map = dict(FACE_DEPTH_SCALE)
        print(f"[mask] Per-face depth scales: {scale_map}")

    print(f"[mask] Processing {len(image_paths)} images …")
    roof_pcts: list[float] = []
    missing_depth = 0

    for img_path in tqdm(image_paths, desc="Masking roof", unit="img"):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            tqdm.write(f"[mask] WARNING: cannot read {img_path}, skipping")
            continue

        # Load pre-computed depth
        depth_path = depth_dir / f"{img_path.stem}.npy"
        if not depth_path.exists():
            missing_depth += 1
            if missing_depth <= 3:
                tqdm.write(f"[mask] WARNING: no depth for {img_path.stem}, skipping")
            # No roof info — mark everything as valid
            out = np.full(image_bgr.shape[:2], 255, dtype=np.uint8)
        else:
            depth_map = np.load(str(depth_path))
            h, w = image_bgr.shape[:2]
            if depth_map.shape != (h, w):
                depth_map = cv2.resize(depth_map, (w, h),
                                       interpolation=cv2.INTER_LINEAR)
            face = face_name_from_path(img_path)
            scale = scale_map.get(face, 1.0)
            roof_mask = depth_to_roof_mask(depth_map, scale, args.erode_px)
            roof_pcts.append(100 * roof_mask.sum() / roof_mask.size)

            # 0 = masked (roof), 255 = valid
            out = np.where(roof_mask, 0, 255).astype(np.uint8)

        cv2.imwrite(str(output_dir / img_path.name), out)

        # Always write visualisation
        cv2.imwrite(str(vis_dir / img_path.name),
                    visualize_mask(image_bgr, out))

    # ── Summary ───────────────────────────────────────────────────────
    if roof_pcts:
        print(f"  Roof coverage: mean={np.mean(roof_pcts):.1f}%, "
              f"min={np.min(roof_pcts):.1f}%, max={np.max(roof_pcts):.1f}%")
    if missing_depth:
        print(f"  WARNING: {missing_depth} images had no matching depth file")

    print(f"[mask] Masks saved to {output_dir}")
    print(f"[mask] Visualisations saved to {vis_dir}")


if __name__ == "__main__":
    main()
