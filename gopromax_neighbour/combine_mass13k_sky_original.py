"""
Overlay sky masks from cubemap_faces_mass13k_manual on top of original
cubemap_faces images and save visualizations.

Sky mask images are binary (0 = not sky, ~255 = sky, with JPEG artifacts).
A threshold of 128 is used to binarize.

The visualization tints sky regions with a semi-transparent color overlay.
"""

import argparse
import glob
import os

import numpy as np
from PIL import Image


def overlay_mask_on_image(
    original: np.ndarray,
    mask: np.ndarray,
    color: tuple = (255, 0, 0),
    alpha: float = 0.5,
    threshold: int = 128,
) -> np.ndarray:
    """Blend a colored overlay on sky regions of the original image."""
    # Sky pixels are < threshold (dark in mask), non-sky are bright
    sky_mask = mask < threshold
    result = original.copy()
    for c in range(3):
        result[:, :, c] = np.where(
            sky_mask,
            np.clip(original[:, :, c] * (1 - alpha) + color[c] * alpha, 0, 255).astype(np.uint8),
            original[:, :, c],
        )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Overlay sky masks on original images.")
    ap.add_argument(
        "--mask_dir",
        default="data/cubemap_faces_mass13k_manual",
        help="Directory containing sky mask images.",
    )
    ap.add_argument(
        "--image_dir",
        default="data/cubemap_faces",
        help="Directory containing original images.",
    )
    ap.add_argument(
        "--vis_dir",
        default="data/cubemap_faces_mass13k_manual/vis",
        help="Output directory for visualization images.",
    )
    ap.add_argument("--alpha", type=float, default=0.5, help="Overlay opacity.")
    ap.add_argument("--threshold", type=int, default=128, help="Mask binarization threshold.")
    args = ap.parse_args()

    mask_dir = os.path.abspath(args.mask_dir)
    image_dir = os.path.abspath(args.image_dir)
    vis_dir = os.path.abspath(args.vis_dir)

    os.makedirs(vis_dir, exist_ok=True)

    exts = ("*.jpg", "*.jpeg", "*.png")
    mask_paths = sorted(p for ext in exts for p in glob.glob(os.path.join(mask_dir, ext)))
    if not mask_paths:
        print(f"No mask images found in {mask_dir}")
        return

    print(f"Found {len(mask_paths)} mask images")
    print(f"Original images: {image_dir}")
    print(f"Output: {vis_dir}")

    for mask_path in mask_paths:
        fname = os.path.basename(mask_path)
        orig_path = os.path.join(image_dir, fname)

        if not os.path.exists(orig_path):
            print(f"  SKIP {fname} (no matching original)")
            continue

        mask = np.array(Image.open(mask_path).convert("L"))
        original = np.array(Image.open(orig_path).convert("RGB"))

        if mask.shape[:2] != original.shape[:2]:
            mask_img = Image.fromarray(mask)
            mask_img = mask_img.resize((original.shape[1], original.shape[0]), Image.NEAREST)
            mask = np.array(mask_img)

        vis = overlay_mask_on_image(original, mask, alpha=args.alpha, threshold=args.threshold)
        Image.fromarray(vis).save(os.path.join(vis_dir, fname))

    print(f"Done. Saved {len(mask_paths)} visualizations to {vis_dir}")


if __name__ == "__main__":
    main()
