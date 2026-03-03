#!/usr/bin/env python3
"""
Generate a COLMAP dense point cloud from an existing sparse reconstruction.

Uses COLMAP's MVS pipeline (GPU-accelerated via CUDA):
  1. image_undistorter  — prepare undistorted images + sparse model
  2. patch_match_stereo — compute dense depth/normal maps on GPU
  3. stereo_fusion      — fuse depth maps into a dense point cloud

Usage
-----
  python densePointCloud.py \\
      --sparse_dir ../colmap_output/colmap_ws \\
      --images_dir ../colmap_output/images \\
      --output     ../colmap_output/dense

  Or use defaults (auto-detect from pointCLoud.py output):
  python densePointCloud.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate COLMAP dense point cloud from sparse reconstruction",
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
        "--output", type=str,
        default=str(default_base / "dense"),
        help="Output directory for dense reconstruction",
    )
    parser.add_argument(
        "--max_image_size", type=int, default=1024,
        help="Maximum image size for patch match stereo",
    )
    args = parser.parse_args()

    sparse_ws = Path(args.sparse_dir).resolve()
    images_dir = Path(args.images_dir).resolve()
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

    print(f"Sparse workspace: {sparse_ws}")
    print(f"Sparse model:     {sparse_recon}")
    print(f"Images:           {images_dir}")
    print(f"Dense output:     {dense_dir}")
    print(f"Max image size:   {args.max_image_size}")
    print()

    # ── 1. Image undistorter ──────────────────────────────────────────
    #   Prepares undistorted images and converts sparse model for MVS
    run_cmd([
        "colmap", "image_undistorter",
        "--image_path", str(images_dir),
        "--input_path", str(sparse_recon),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP",
        "--max_image_size", str(args.max_image_size),
    ], desc="Image undistorter")

    # ── 2. Patch match stereo (GPU) ──────────────────────────────────
    #   Computes dense depth and normal maps using CUDA
    run_cmd([
        "colmap", "patch_match_stereo",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true",
        "--PatchMatchStereo.max_image_size", str(args.max_image_size),
    ], desc="Patch match stereo (GPU)")

    # ── 3. Stereo fusion ─────────────────────────────────────────────
    #   Fuses depth maps into a dense point cloud
    dense_ply = dense_dir / "fused.ply"
    run_cmd([
        "colmap", "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(dense_ply),
    ], desc="Stereo fusion → dense PLY")

    # Check result
    if dense_ply.exists():
        size_mb = dense_ply.stat().st_size / (1024 * 1024)
        # Read vertex count from PLY header
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
