"""
Generate COLMAP sparse reconstruction from GoPro Max cubemap face images.

Pipeline: feature extraction → exhaustive matching → mapper → PLY export.
No known poses required — COLMAP estimates everything via SfM.

Usage:
    python colmap_pointcloud_sparse.py \
        --image_dir data/cubemap_faces_alpha_matting \
        --image_pattern '*_nosky.png' \
        --output_dir data/colmap_pointcloud_sparse
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


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
        description="COLMAP sparse reconstruction from cubemap face images.",
    )
    ap.add_argument("--image_dir", default="data/cubemap_faces_alpha_matting",
                    help="Directory containing input images.")
    ap.add_argument("--image_pattern", default="*_nosky.png",
                    help="Glob (relative to --image_dir) selecting which files to use. "
                         "Pass '*' to use everything.")
    ap.add_argument("--output_dir", default="data/colmap_pointcloud_sparse",
                    help="Output directory for sparse model and point cloud.")
    ap.add_argument("--colmap_exe", default="colmap",
                    help="Path to the COLMAP executable.")
    ap.add_argument("--use_gpu", type=int, default=1, choices=[0, 1],
                    help="Use GPU for SIFT & matching (1=yes, 0=no).")
    ap.add_argument("--matcher", default="exhaustive",
                    choices=["exhaustive", "sequential"],
                    help="Feature matching strategy.")
    args = ap.parse_args()

    script_dir = Path(__file__).parent.resolve()
    image_dir = Path(args.image_dir)
    if not image_dir.is_absolute():
        image_dir = (script_dir / image_dir).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (script_dir / output_dir).resolve()

    if not image_dir.exists():
        sys.exit(f"ERROR: Image directory not found: {image_dir}")

    # Filter the directory by the requested glob and write an image_list.txt
    # that COLMAP will honour (relative paths to --image_path).
    selected = sorted(image_dir.glob(args.image_pattern))
    if not selected:
        sys.exit(f"ERROR: No images in {image_dir} match pattern '{args.image_pattern}'.")
    print(f"Found {len(selected)} images in {image_dir} matching '{args.image_pattern}'")

    # Create workspace directories
    sparse_dir = output_dir / "sparse" / "0"
    db = output_dir / "database.db"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    image_list = output_dir / "image_list.txt"
    image_list.write_text("\n".join(p.name for p in selected) + "\n")

    if db.exists():
        db.unlink()
        print(f"Removed existing database: {db}")

    colmap = args.colmap_exe
    gpu = str(args.use_gpu)

    # Step 1: Feature extraction
    run_colmap([
        colmap, "feature_extractor",
        "--database_path", str(db),
        "--image_path", str(image_dir),
        "--image_list_path", str(image_list),
        "--ImageReader.camera_model", "PINHOLE",
        "--ImageReader.single_camera", "0",
        "--SiftExtraction.use_gpu", gpu,
        "--SiftExtraction.max_num_features", "8192",
    ], "Feature extraction (SIFT)")

    # Step 2: Feature matching
    if args.matcher == "exhaustive":
        run_colmap([
            colmap, "exhaustive_matcher",
            "--database_path", str(db),
            "--SiftMatching.use_gpu", gpu,
        ], "Exhaustive feature matching")
    else:
        run_colmap([
            colmap, "sequential_matcher",
            "--database_path", str(db),
            "--SiftMatching.use_gpu", gpu,
            "--SequentialMatching.overlap", "15",
        ], "Sequential feature matching")

    # Step 3: Mapper (full SfM reconstruction)
    run_colmap([
        colmap, "mapper",
        "--database_path", str(db),
        "--image_path", str(image_dir),
        "--image_list_path", str(image_list),
        "--output_path", str(output_dir / "sparse"),
    ], "Sparse reconstruction (mapper)")

    # Mapper may emit several sub-models (sparse/0, sparse/1, ...).  The first
    # one is not necessarily the largest — pick the one with the biggest
    # points3D.bin, which is the most complete reconstruction.
    model_dirs = sorted(
        [d for d in (output_dir / "sparse").iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    if not model_dirs:
        sys.exit("ERROR: Mapper produced no models.")

    def _model_size(d: Path) -> int:
        p = d / "points3D.bin"
        return p.stat().st_size if p.exists() else 0

    best_model = max(model_dirs, key=_model_size)
    if len(model_dirs) > 1:
        sizes = ", ".join(f"{d.name}={_model_size(d)/1e6:.1f}MB" for d in model_dirs)
        print(f"\n  Mapper produced {len(model_dirs)} models ({sizes}); "
              f"picking largest: {best_model.name}")
    else:
        print(f"\n  Using model: {best_model}")

    # Step 4: Export PLY
    ply = output_dir / "point_cloud.ply"
    run_colmap([
        colmap, "model_converter",
        "--input_path", str(best_model),
        "--output_path", str(ply),
        "--output_type", "PLY",
    ], "Export point cloud to PLY")

    # Summary
    print(f"\n{'═' * 60}")
    print(f"  DONE")
    print(f"  Sparse model : {best_model}")
    print(f"  Point cloud  : {ply}")
    print(f"  Images       : {image_dir}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
