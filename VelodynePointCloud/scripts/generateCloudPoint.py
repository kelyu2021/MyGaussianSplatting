"""
Generate a merged world-frame point cloud from KITTI-360 Velodyne LiDAR data.

Requires:
  - Velodyne .bin scans  : data_3d_raw/<sequence>/velodyne_points/data/*.bin
  - Calibration files    : calibration/calib_cam_to_velo.txt
                           calibration/calib_cam_to_pose.txt
  - Pose file            : data_poses/<sequence>/poses.txt
                           (or cam0_to_world.txt)

Coordinate chain (Velodyne → World):
  p_world = T_world_pose  @  T_pose_cam0  @  inv(T_velo_cam0)  @  p_velo

Output: a binary-little-endian PLY with (x, y, z, red, green, blue).
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
#  I/O helpers
# ---------------------------------------------------------------------------

def load_velodyne_points(filepath: str) -> np.ndarray:
    """Load a Velodyne scan from a KITTI .bin file.

    Returns an (N, 4) float32 array: [x, y, z, reflectance].
    """
    points = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
    return points


def parse_3x4_matrix(values: list[float]) -> np.ndarray:
    """Convert 12 floats into a 4×4 homogeneous matrix."""
    assert len(values) == 12, f"Expected 12 values, got {len(values)}"
    T = np.eye(4)
    T[:3, :] = np.array(values).reshape(3, 4)
    return T


def load_calib_cam_to_velo(filepath: Path) -> np.ndarray:
    """Load T_velo_cam0 from calib_cam_to_velo.txt (single-line, 12 floats)."""
    with open(filepath, "r") as f:
        line = f.readline().strip()
    values = [float(v) for v in line.split()]
    return parse_3x4_matrix(values)


def load_calib_cam_to_pose(filepath: Path, camera: str = "image_00") -> np.ndarray:
    """Load T_pose_camN from calib_cam_to_pose.txt."""
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith(camera + ":"):
                parts = line.split(":")[1].strip().split()
                values = [float(v) for v in parts]
                return parse_3x4_matrix(values)
    raise ValueError(f"Camera '{camera}' not found in {filepath}")


def load_poses(filepath: Path) -> dict[int, np.ndarray]:
    """Load per-frame poses from poses.txt.

    Expected format per line:
        frame_id  T00 T01 T02 T03  T10 T11 T12 T13  T20 T21 T22 T23

    Returns a dict  { frame_id : 4×4 numpy array }.
    """
    poses: dict[int, np.ndarray] = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            frame_id = int(parts[0])
            values = [float(v) for v in parts[1:13]]
            poses[frame_id] = parse_3x4_matrix(values)
    print(f"Loaded {len(poses)} poses from {filepath.name}")
    return poses


def load_cam0_to_world(filepath: Path) -> dict[int, np.ndarray]:
    """Alternative loader if the user has cam0_to_world.txt instead of poses.txt.

    Same line format as poses.txt.
    """
    return load_poses(filepath)


# ---------------------------------------------------------------------------
#  Point-cloud output
# ---------------------------------------------------------------------------

def write_ply(filepath: str, points: np.ndarray, colors: np.ndarray) -> None:
    """Write an (N, 3) point cloud with (N, 3) uint8 colors to a binary PLY."""
    n = points.shape[0]
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
    with open(filepath, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(n):
            f.write(struct.pack("<fff", *points[i, :3].tolist()))
            f.write(struct.pack("<BBB", *colors[i, :3].tolist()))
    print(f"Saved {n:,} points → {filepath}")


def write_ply_fast(filepath: str, points: np.ndarray, colors: np.ndarray) -> None:
    """Faster PLY writer using a single packed array (requires contiguous data)."""
    n = points.shape[0]
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
    # Build a structured array for efficient binary write
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    vertex_data = np.empty(n, dtype=dtype)
    vertex_data["x"] = points[:, 0]
    vertex_data["y"] = points[:, 1]
    vertex_data["z"] = points[:, 2]
    vertex_data["red"] = colors[:, 0]
    vertex_data["green"] = colors[:, 1]
    vertex_data["blue"] = colors[:, 2]

    with open(filepath, "wb") as f:
        f.write(header.encode("ascii"))
        vertex_data.tofile(f)
    print(f"Saved {n:,} points → {filepath}")


# ---------------------------------------------------------------------------
#  Colour helpers
# ---------------------------------------------------------------------------

def reflectance_to_color(reflectance: np.ndarray) -> np.ndarray:
    """Map [0, 1] reflectance to a grayscale RGB colour (N, 3) uint8."""
    gray = np.clip(reflectance * 255.0, 0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def height_to_color(z: np.ndarray) -> np.ndarray:
    """Map z-height to a colour-mapped RGB (N, 3) uint8 for visualization."""
    z_min, z_max = np.percentile(z, [2, 98])
    z_norm = np.clip((z - z_min) / (z_max - z_min + 1e-8), 0, 1)
    # Simple blue → green → red colour ramp
    r = np.clip(2.0 * z_norm - 0.5, 0, 1)
    g = np.clip(1.0 - 2.0 * np.abs(z_norm - 0.5), 0, 1)
    b = np.clip(1.0 - 2.0 * z_norm, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
#  Voxel down-sampling
# ---------------------------------------------------------------------------

def voxel_downsample(points: np.ndarray, colors: np.ndarray,
                     voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Simple voxel-grid down-sampling that keeps one random point per cell."""
    if voxel_size <= 0:
        return points, colors
    voxel_indices = np.floor(points / voxel_size).astype(np.int64)
    # Use a hash for fast unique detection
    _, unique_idx = np.unique(
        voxel_indices, axis=0, return_index=True
    )
    print(f"Voxel downsample ({voxel_size}m): {points.shape[0]:,} → {len(unique_idx):,} points")
    return points[unique_idx], colors[unique_idx]


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a merged world-frame point cloud from KITTI-360 Velodyne data."
    )
    parser.add_argument(
        "--data_dir", type=str, default="../data/kitti360",
        help="Root directory of the KITTI-360 data (contains calibration/, data_3d_raw/, data_poses/).",
    )
    parser.add_argument(
        "--sequence", type=str, default="2013_05_28_drive_0000_sync",
        help="Sequence directory name.",
    )
    parser.add_argument(
        "--start_frame", type=int, default=0,
        help="First frame index to include.",
    )
    parser.add_argument(
        "--end_frame", type=int, default=-1,
        help="Last frame index to include (-1 = all available).",
    )
    parser.add_argument(
        "--output", type=str, default="output/point_cloud.ply",
        help="Output PLY file path.",
    )
    parser.add_argument(
        "--color_mode", type=str, default="reflectance",
        choices=["reflectance", "height"],
        help="How to colour the points: 'reflectance' (grayscale) or 'height' (z-based colour ramp).",
    )
    parser.add_argument(
        "--voxel_size", type=float, default=-1,
        help="Voxel size in metres for down-sampling (-1 to disable).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve & validate paths
    # ------------------------------------------------------------------
    data_dir = Path(args.data_dir).resolve()
    calib_dir = data_dir / "calibration"
    velodyne_dir = data_dir / "data_3d_raw" / args.sequence / "velodyne_points" / "data"

    # Try poses.txt first, fall back to cam0_to_world.txt
    poses_dir = data_dir / "data_poses" / args.sequence
    poses_file = poses_dir / "poses.txt"
    cam0_to_world_file = poses_dir / "cam0_to_world.txt"

    for p, label in [
        (calib_dir / "calib_cam_to_velo.txt", "cam-to-velo calibration"),
        (calib_dir / "calib_cam_to_pose.txt", "cam-to-pose calibration"),
        (velodyne_dir, "Velodyne data directory"),
    ]:
        if not p.exists():
            print(f"ERROR: {label} not found at {p}")
            sys.exit(1)

    if poses_file.exists():
        pose_loader = load_poses
        pose_path = poses_file
    elif cam0_to_world_file.exists():
        pose_loader = load_cam0_to_world
        pose_path = cam0_to_world_file
    else:
        print(f"ERROR: Neither poses.txt nor cam0_to_world.txt found in {poses_dir}")
        print("Please place your poses file there.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load calibration
    # ------------------------------------------------------------------
    T_velo_cam0 = load_calib_cam_to_velo(calib_dir / "calib_cam_to_velo.txt")
    T_pose_cam0 = load_calib_cam_to_pose(calib_dir / "calib_cam_to_pose.txt", camera="image_00")
    T_cam0_velo = np.linalg.inv(T_velo_cam0)

    # Combined transform: Velodyne → pose frame
    T_pose_velo = T_pose_cam0 @ T_cam0_velo

    print("Calibration loaded:")
    print(f"  T_velo_cam0 (cam→velo):\n{T_velo_cam0[:3, :]}")
    print(f"  T_pose_cam0 (cam0→pose):\n{T_pose_cam0[:3, :]}")

    # ------------------------------------------------------------------
    # Load poses
    # ------------------------------------------------------------------
    poses = pose_loader(pose_path)

    # ------------------------------------------------------------------
    # Enumerate Velodyne scans
    # ------------------------------------------------------------------
    velo_files = sorted(velodyne_dir.glob("*.bin"))
    if not velo_files:
        print(f"ERROR: No .bin files found in {velodyne_dir}")
        sys.exit(1)
    print(f"Found {len(velo_files)} Velodyne scans")

    start = args.start_frame
    end = args.end_frame if args.end_frame >= 0 else int(velo_files[-1].stem)

    # ------------------------------------------------------------------
    # Process each frame
    # ------------------------------------------------------------------
    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    processed = 0

    for velo_file in velo_files:
        frame_id = int(velo_file.stem)

        # Frame range filter
        if frame_id < start or frame_id > end:
            continue

        # Skip frames without a known pose
        if frame_id not in poses:
            continue

        # Load scan
        raw = load_velodyne_points(str(velo_file))
        xyz = raw[:, :3]
        reflectance = raw[:, 3]

        # Transform: Velodyne → World
        T_world_pose = poses[frame_id]
        T_world_velo = T_world_pose @ T_pose_velo
        xyz_h = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
        xyz_world = (T_world_velo @ xyz_h.T).T[:, :3]

        # Colour
        if args.color_mode == "reflectance":
            colors = reflectance_to_color(reflectance)
        else:
            colors = height_to_color(xyz_world[:, 2])

        all_points.append(xyz_world)
        all_colors.append(colors)
        processed += 1

        if processed % 50 == 0:
            total_pts = sum(p.shape[0] for p in all_points)
            print(f"  ... processed {processed} frames ({total_pts:,} points so far)")

    if processed == 0:
        print("ERROR: No frames were processed. Check that poses match the Velodyne frame indices.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------
    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors)
    print(f"Merged {processed} frames → {merged_points.shape[0]:,} points")

    # ------------------------------------------------------------------
    # Optional voxel down-sampling
    # ------------------------------------------------------------------
    if args.voxel_size > 0:
        merged_points, merged_colors = voxel_downsample(
            merged_points, merged_colors, args.voxel_size
        )

    # ------------------------------------------------------------------
    # Save PLY
    # ------------------------------------------------------------------
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_ply_fast(str(output_path), merged_points, merged_colors)


if __name__ == "__main__":
    main()
