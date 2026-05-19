#!/usr/bin/env python3
"""
Generate a COLMAP dense point cloud with mask support.

Same MVS pipeline as densePointCloud.py, but after image_undistorter the
undistorted images are masked (sky / car-roof set to black) so that
patch_match_stereo ignores those regions.

Pipeline
--------
  1. image_undistorter  — prepare undistorted images + sparse model
  2. Apply masks        — zero out sky/roof in undistorted images
  3. patch_match_stereo — compute dense depth/normal maps on GPU
  4. stereo_fusion      — fuse depth maps into a dense point cloud

Usage
-----
  python densePointCloudWithMask.py
  python densePointCloudWithMask.py \
      --sparse_dir output/colmap_ws \
      --images_dir output/images \
      --masks_dir  output/masks_depth \
      --output     output/dense_mask
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def run_cmd(cmd: list[str], desc: str = "") -> None:
    """Run a shell command, printing it and raising on failure."""
    print(f"\n{'=' * 60}")
    if desc:
        print(f"[COLMAP] {desc}")
    print(f"  $ {' '.join(cmd)}")
    print("=" * 60)
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {' '.join(cmd)}"
        )


def apply_masks(undistorted_images_dir: Path, masks_dir: Path) -> None:
    """
    For each undistorted image, find its corresponding mask and zero out
    masked pixels (mask==0 → pixel set to black).
    """
    image_paths = sorted(undistorted_images_dir.glob("*.png"))
    if not image_paths:
        print("WARNING: No PNG images found in undistorted directory", file=sys.stderr)
        return

    matched, skipped = 0, 0
    for img_path in tqdm(image_paths, desc="Applying masks", unit="img"):
        mask_path = masks_dir / img_path.name
        if not mask_path.exists():
            tqdm.write(f"  [skip] No mask for {img_path.name}")
            skipped += 1
            continue

        image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            tqdm.write(f"  [skip] Cannot read {img_path.name} or its mask")
            skipped += 1
            continue

        # Resize mask to match undistorted image size if needed
        h, w = image.shape[:2]
        mh, mw = mask.shape[:2]
        if (mh, mw) != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Apply: set masked pixels (mask == 0) to black
        invalid = mask == 0
        if image.ndim == 3:
            image[invalid] = 0
        else:
            image[invalid] = 0

        cv2.imwrite(str(img_path), image)
        matched += 1

    print(f"  Masks applied: {matched}, skipped: {skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate COLMAP dense point cloud with mask support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    default_base = Path(__file__).resolve().parent / "output"
    parser.add_argument(
        "--sparse_dir", type=str,
        default=str(default_base / "colmap_ws"),
        help="Path to COLMAP workspace with sparse/ and database.db",
    )
    parser.add_argument(
        "--images_dir", type=str,
        default=str(default_base / "images"),
        help="Path to the images used in reconstruction",
    )
    parser.add_argument(
        "--masks_dir", type=str,
        default=str(default_base / "masks_depth"),
        help="Path to binary masks (0=masked, 255=valid)",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(default_base / "dense_mask"),
        help="Output directory for dense reconstruction",
    )
    parser.add_argument(
        "--max_image_size", type=int, default=1024,
        help="Maximum image size for patch match stereo",
    )
    parser.add_argument(
        "--gpu_index", type=str, default="0,1",
        help="Comma-separated GPU indices for patch match stereo",
    )
    parser.add_argument(
        "--fusion_min_num_pixels", type=int, default=5,
        help="Min views a point must be seen in to survive fusion (default 5, try 3)",
    )
    parser.add_argument(
        "--fusion_max_reproj_error", type=float, default=2.0,
        help="Max reprojection error for fusion (default 2.0, try 3-4)",
    )
    parser.add_argument(
        "--pm_filter_min_ncc", type=float, default=0.1,
        help="PatchMatch min NCC filter (default 0.1, try 0.05 for more points)",
    )
    args = parser.parse_args()

    sparse_ws = Path(args.sparse_dir).resolve()
    images_dir = Path(args.images_dir).resolve()
    masks_dir = Path(args.masks_dir).resolve()
    dense_dir = Path(args.output).resolve()
    dense_dir.mkdir(parents=True, exist_ok=True)

    # Find the sparse model (usually sparse/0)
    sparse_model = sparse_ws / "sparse"
    recon_dirs = sorted(sparse_model.iterdir()) if sparse_model.exists() else []
    if not recon_dirs:
        print(f"ERROR: No sparse reconstruction found in {sparse_model}",
              file=sys.stderr)
        sys.exit(1)
    sparse_recon = recon_dirs[0]

    if not masks_dir.exists():
        print(f"ERROR: Masks directory not found: {masks_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Sparse workspace: {sparse_ws}")
    print(f"Sparse model:     {sparse_recon}")
    print(f"Images:           {images_dir}")
    print(f"Masks:            {masks_dir}")
    print(f"Dense output:     {dense_dir}")
    print(f"Max image size:   {args.max_image_size}")
    print(f"GPU index:        {args.gpu_index}")
    print(f"Fusion min views: {args.fusion_min_num_pixels}")
    print(f"Fusion max repr:  {args.fusion_max_reproj_error}")
    print(f"PM filter NCC:    {args.pm_filter_min_ncc}")
    print()

    # ── 1. Image undistorter ──────────────────────────────────────────
    run_cmd([
        "colmap", "image_undistorter",
        "--image_path", str(images_dir),
        "--input_path", str(sparse_recon),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP",
        "--max_image_size", str(args.max_image_size),
    ], desc="Image undistorter")

    # ── 2. Apply masks to undistorted images ──────────────────────────
    undistorted_images = dense_dir / "images"
    if not undistorted_images.exists():
        print(f"ERROR: Undistorted images not found at {undistorted_images}",
              file=sys.stderr)
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("[MASK] Applying masks to undistorted images …")
    print("=" * 60)
    apply_masks(undistorted_images, masks_dir)

    # ── 3. Patch match stereo (GPU) ──────────────────────────────────
    run_cmd([
        "colmap", "patch_match_stereo",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true",
        "--PatchMatchStereo.max_image_size", str(args.max_image_size),
        "--PatchMatchStereo.gpu_index", args.gpu_index,
        "--PatchMatchStereo.filter_min_ncc", str(args.pm_filter_min_ncc),
    ], desc="Patch match stereo (GPU)")

    # ── 4. Stereo fusion ─────────────────────────────────────────────
    dense_ply = dense_dir / "fused.ply"
    run_cmd([
        "colmap", "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(dense_ply),
        "--StereoFusion.min_num_pixels", str(args.fusion_min_num_pixels),
        "--StereoFusion.max_reproj_error", str(args.fusion_max_reproj_error),
    ], desc="Stereo fusion → dense PLY")

    # Check result
    if dense_ply.exists():
        size_mb = dense_ply.stat().st_size / (1024 * 1024)
        vertex_count = 0
        with open(dense_ply, "rb") as f:
            for line in f:
                decoded = line.decode("ascii", errors="ignore").strip()
                if decoded.startswith("element vertex"):
                    vertex_count = int(decoded.split()[-1])
                if decoded == "end_header":
                    break

        print(f"\n{'=' * 60}")
        print("DONE!")
        print(f"  Dense point cloud: {dense_ply}")
        print(f"  Size:              {size_mb:.1f} MB")
        print(f"  Vertices:          {vertex_count:,}")
        print(f"  Dense workspace:   {dense_dir}")
        print("=" * 60)
    else:
        print("WARNING: Dense point cloud was not created.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
