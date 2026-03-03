"""
Combine COLMAP and Velodyne point clouds into a single PLY file.

Both inputs must be in the same coordinate frame (KITTI-360 world).
The script:
  1. Reads both binary-little-endian PLY files (float xyz + uchar rgb).
  2. Optionally filters COLMAP outliers using the Velodyne bounding box
     (expanded by a configurable margin).
  3. Optionally voxel-downsamples the merged cloud.
  4. Writes the merged result as a binary PLY.

Usage:
  python combine2PointCloud.py
  python combine2PointCloud.py --margin 20 --voxel_size 0.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
#  PLY I/O
# ═══════════════════════════════════════════════════════════════════════════

VERTEX_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
])


def read_ply(path: Path) -> np.ndarray:
    """Read a binary-little-endian PLY with (x, y, z, red, green, blue)."""
    with open(path, "rb") as f:
        n_verts = 0
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            if line == "end_header":
                break
        data = np.fromfile(f, dtype=VERTEX_DTYPE, count=n_verts)
    print(f"  Read {len(data):,} points from {path.name}")
    return data


def write_ply(path: Path, data: np.ndarray) -> None:
    """Write a binary-little-endian PLY with (x, y, z, red, green, blue)."""
    n = len(data)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        data.tofile(f)
    print(f"  Saved {n:,} points → {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Filtering & downsampling
# ═══════════════════════════════════════════════════════════════════════════

def bbox_filter(cloud: np.ndarray, ref: np.ndarray, margin: float) -> np.ndarray:
    """Keep only points within the bounding box of *ref* expanded by *margin* (metres)."""
    xyz_ref = np.column_stack([ref["x"], ref["y"], ref["z"]])
    lo = xyz_ref.min(axis=0) - margin
    hi = xyz_ref.max(axis=0) + margin

    xyz = np.column_stack([cloud["x"], cloud["y"], cloud["z"]])
    mask = np.all((xyz >= lo) & (xyz <= hi), axis=1)
    n_before = len(cloud)
    filtered = cloud[mask]
    print(f"  Bounding-box filter (margin={margin}m): "
          f"{n_before:,} → {len(filtered):,} points "
          f"(removed {n_before - len(filtered):,})")
    return filtered


def statistical_outlier_filter(
    cloud: np.ndarray, k: int = 20, std_ratio: float = 2.0
) -> np.ndarray:
    """Remove statistical outliers based on mean k-NN distance.

    This is a simplified version — for very large clouds it can be slow.
    Only applied when the cloud is ≤ 500K points.
    """
    n = len(cloud)
    if n > 500_000:
        print(f"  Statistical filter skipped (cloud too large: {n:,} points)")
        return cloud

    from scipy.spatial import cKDTree  # lazy import

    xyz = np.column_stack([cloud["x"], cloud["y"], cloud["z"]])
    tree = cKDTree(xyz)
    dists, _ = tree.query(xyz, k=k + 1)  # +1 because the point itself is included
    mean_dists = dists[:, 1:].mean(axis=1)

    threshold = mean_dists.mean() + std_ratio * mean_dists.std()
    mask = mean_dists < threshold
    filtered = cloud[mask]
    print(f"  Statistical outlier filter (k={k}, σ={std_ratio}): "
          f"{n:,} → {len(filtered):,} points")
    return filtered


def voxel_downsample(cloud: np.ndarray, voxel_size: float) -> np.ndarray:
    """Voxel-grid downsampling — keep one random point per cell."""
    if voxel_size <= 0:
        return cloud
    xyz = np.column_stack([cloud["x"], cloud["y"], cloud["z"]])
    voxel_idx = np.floor(xyz / voxel_size).astype(np.int64)
    _, unique = np.unique(voxel_idx, axis=0, return_index=True)
    result = cloud[unique]
    print(f"  Voxel downsample ({voxel_size}m): "
          f"{len(cloud):,} → {len(result):,} points")
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Combine COLMAP and Velodyne point clouds.",
    )
    _script_dir = Path(__file__).resolve().parent
    ap.add_argument(
        "--colmap_ply",
        default=str(_script_dir.parent
                    / "ColmapPointCloud" / "scripts" / "output"
                    / "colmap_ws" / "point_cloud.ply"),
        help="Path to the COLMAP point cloud PLY.",
    )
    ap.add_argument(
        "--velodyne_ply",
        default=str(_script_dir.parent
                    / "VelodynePointCloud" / "scripts" / "output"
                    / "point_cloud.ply"),
        help="Path to the Velodyne point cloud PLY.",
    )
    ap.add_argument(
        "--output",
        default=str(Path(__file__).parent / "output" / "combined_point_cloud.ply"),
        help="Output PLY path.",
    )
    ap.add_argument(
        "--margin", type=float, default=15.0,
        help="Bounding-box margin (m) for filtering COLMAP outliers "
             "using Velodyne extent. Set ≤0 to disable.",
    )
    ap.add_argument(
        "--stat_filter", action="store_true",
        help="Apply statistical outlier filter to COLMAP cloud "
             "(slow for >500K points).",
    )
    ap.add_argument(
        "--voxel_size", type=float, default=-1,
        help="Voxel size (m) for downsampling the merged cloud "
             "(-1 to disable).",
    )
    args = ap.parse_args()

    colmap_path = Path(args.colmap_ply).resolve()
    velo_path = Path(args.velodyne_ply).resolve()
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = (Path(__file__).parent / out_path).resolve()

    for p, tag in [(colmap_path, "COLMAP PLY"), (velo_path, "Velodyne PLY")]:
        if not p.exists():
            sys.exit(f"ERROR: {tag} not found at {p}")

    # ── Read ──────────────────────────────────────────────────────────
    print("Reading point clouds …")
    colmap = read_ply(colmap_path)
    velodyne = read_ply(velo_path)

    # ── Filter COLMAP outliers ────────────────────────────────────────
    if args.margin > 0:
        print("Filtering COLMAP outliers …")
        colmap = bbox_filter(colmap, velodyne, args.margin)

    if args.stat_filter:
        print("Statistical outlier filtering (COLMAP) …")
        colmap = statistical_outlier_filter(colmap)

    # ── Combine ───────────────────────────────────────────────────────
    print("Combining …")
    merged = np.concatenate([velodyne, colmap])
    print(f"  Velodyne:  {len(velodyne):,}")
    print(f"  COLMAP:    {len(colmap):,}")
    print(f"  Combined:  {len(merged):,}")

    # ── Optional voxel downsample ─────────────────────────────────────
    if args.voxel_size > 0:
        merged = voxel_downsample(merged, args.voxel_size)

    # ── Stats ─────────────────────────────────────────────────────────
    print("\nMerged cloud statistics:")
    for axis in ["x", "y", "z"]:
        vals = merged[axis]
        print(f"  {axis.upper()}: {vals.min():.2f} .. {vals.max():.2f}  "
              f"(range {vals.max() - vals.min():.2f}m)")

    # ── Write ─────────────────────────────────────────────────────────
    print()
    write_ply(out_path, merged)
    print("\nDone.")


if __name__ == "__main__":
    main()
