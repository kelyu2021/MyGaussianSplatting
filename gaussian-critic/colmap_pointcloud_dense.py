"""
Generate COLMAP dense reconstruction from an existing sparse model.

Pipeline: image_undistorter → patch_match_stereo → stereo_fusion → PLY export.

Usage:
    python colmap_pointcloud_dense.py \
        --sparse_dir data/colmap_pointcloud_sparse/sparse/1 \
        --image_dir data/cubemap_faces \
        --output_dir data/colmap_pointcloud_dense
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def populate_dense_masks(output_dir: Path, sky_mask_dir: Path) -> int:
    """
    After image_undistorter, write inverted sky masks into <output_dir>/masks/
    so that patch_match_stereo and stereo_fusion ignore sky pixels.

    COLMAP dense convention: mask pixel == 0  →  pixel is masked (ignored).
    Input sky masks assumed sky=255 (white) → inverted to 0 for COLMAP.

    Matches undistorted images in <output_dir>/images/ to sky masks by stem.
    Returns number of masks written.
    """
    images_dir = output_dir / "images"
    if not images_dir.exists():
        print("  WARNING: undistorted images dir not found, skipping masks.")
        return 0

    # Index all available sky masks by lowercase stem
    sky_index = {
        p.stem.lower(): p
        for p in sky_mask_dir.iterdir()
        if p.suffix.lower() == ".png"
    }
    if not sky_index:
        print(f"  WARNING: no .png masks found in {sky_mask_dir}")
        return 0

    masks_dir = output_dir / "masks"
    masks_dir.mkdir(exist_ok=True)

    count = 0
    for img_path in sorted(images_dir.rglob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        mask_src = sky_index.get(img_path.stem.lower())
        if mask_src is None:
            continue
        # Mirror the images/ subdirectory structure inside masks/
        rel = img_path.relative_to(images_dir)
        mask_dst = masks_dir / rel.parent / (img_path.stem + ".png")
        mask_dst.parent.mkdir(parents=True, exist_ok=True)
        mask = Image.open(mask_src).convert("L")
        Image.fromarray(255 - np.array(mask)).save(mask_dst)
        count += 1

    print(f"  Written {count} dense mask(s) to {masks_dir}")
    if count == 0:
        print(f"  (image stems: {list(sky_index.keys())[:5]} …)")
    return count


def run_colmap(cmd: list[str], desc: str) -> None:
    """Run a COLMAP CLI command; abort on failure."""
    print(f"\n{'─' * 60}\n  {desc}\n  $ {' '.join(cmd)}\n{'─' * 60}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        for ln in proc.stdout.strip().splitlines()[-15:]:
            print(f"  {ln}")
    if proc.returncode != 0:
        print(f"\n  *** COLMAP FAILED (exit {proc.returncode}) ***")
        if proc.stderr:
            for ln in proc.stderr.strip().splitlines()[-20:]:
                print(f"  {ln}")
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="COLMAP dense reconstruction from an existing sparse model.",
    )
    ap.add_argument("--sparse_dir", default="data/colmap_pointcloud_sparse/sparse/1",
                    help="Path to the sparse model directory (e.g. sparse/1).")
    ap.add_argument("--image_dir", default="data/cubemap_faces",
                    help="Directory containing input images.")
    ap.add_argument("--output_dir", default="data/colmap_pointcloud_dense",
                    help="Output directory for dense reconstruction.")
    ap.add_argument("--colmap_exe", default="colmap",
                    help="Path to the COLMAP executable.")
    ap.add_argument("--use_gpu", type=int, default=1, choices=[0, 1],
                    help="Use GPU for PatchMatch stereo (1=yes, 0=no).")
    ap.add_argument("--sky_mask_dir", default=None,
                    help="Directory of sky masks (sky=white/255). "
                         "Masked pixels are excluded from depth estimation.")
    args = ap.parse_args()

    script_dir = Path(__file__).parent.resolve()

    sparse_dir = Path(args.sparse_dir)
    if not sparse_dir.is_absolute():
        sparse_dir = (script_dir / sparse_dir).resolve()
    image_dir = Path(args.image_dir)
    if not image_dir.is_absolute():
        image_dir = (script_dir / image_dir).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (script_dir / output_dir).resolve()

    for p, tag in [(sparse_dir, "Sparse model"), (image_dir, "Image directory")]:
        if not p.exists():
            sys.exit(f"ERROR: {tag} not found: {p}")

    output_dir.mkdir(parents=True, exist_ok=True)
    colmap = args.colmap_exe
    gpu = str(args.use_gpu)

    # Resolve sky_mask_dir early so we can report errors before COLMAP runs
    sky_mask_dir = None
    if args.sky_mask_dir:
        sky_mask_dir = Path(args.sky_mask_dir)
        if not sky_mask_dir.is_absolute():
            sky_mask_dir = (script_dir / sky_mask_dir).resolve()
        if not sky_mask_dir.exists():
            sys.exit(f"ERROR: Sky mask directory not found: {sky_mask_dir}")

    # Step 1: Undistort images
    run_colmap([
        colmap, "image_undistorter",
        "--image_path", str(image_dir),
        "--input_path", str(sparse_dir),
        "--output_path", str(output_dir),
        "--output_type", "COLMAP",
    ], "Image undistortion")

    # Step 1b: Populate <output_dir>/masks/ from sky masks so that
    # patch_match_stereo and stereo_fusion skip sky pixels.
    if sky_mask_dir:
        n = populate_dense_masks(output_dir, sky_mask_dir)
        if n == 0:
            print("  WARNING: 0 masks written — sky masking will be skipped.")

    # Step 2: PatchMatch stereo (depth & normal maps)
    run_colmap([
        colmap, "patch_match_stereo",
        "--workspace_path", str(output_dir),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true",
        "--PatchMatchStereo.gpu_index", "0" if args.use_gpu else "-1",
    ], "PatchMatch stereo (depth map estimation)")

    # Step 3: Stereo fusion → dense point cloud
    fused_ply = output_dir / "fused.ply"
    run_colmap([
        colmap, "stereo_fusion",
        "--workspace_path", str(output_dir),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(fused_ply),
    ], "Stereo fusion → dense point cloud")

    # Summary
    print(f"\n{'═' * 60}")
    print(f"  DONE")
    print(f"  Dense point cloud : {fused_ply}")
    print(f"  Workspace         : {output_dir}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
