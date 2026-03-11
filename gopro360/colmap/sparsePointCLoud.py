#!/usr/bin/env python3
"""
Generate a COLMAP sparse point cloud from a GoPro 360° equirectangular video.

Pipeline
--------
  1. Extract frames from the video at a configurable FPS
  2. Convert each equirectangular frame to perspective cubemap faces
     (front, right, back, left) so COLMAP can use the PINHOLE camera model
  3. COLMAP feature_extractor  (SIFT, CPU)
  4. COLMAP sequential_matcher
  5. COLMAP mapper             (automatic sparse reconstruction)
  6. COLMAP model_converter    (export PLY)

Usage
-----
  python pointCLoud.py \\
      --video  ../data/GS010001_10s_2048_4096.mp4 \\
      --output ../colmap_output \\
      --extract_fps 2 \\
      --face_size 1024
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════
#  Equirectangular → Perspective conversion
# ═══════════════════════════════════════════════════════════════════════════

def _rotation_matrix(yaw: float, pitch: float) -> np.ndarray:
    """Return 3×3 rotation matrix for a given yaw (left/right) and pitch (up/down) in radians."""
    Ry = np.array([
        [np.cos(yaw),  0, np.sin(yaw)],
        [0,            1, 0           ],
        [-np.sin(yaw), 0, np.cos(yaw)],
    ])
    Rx = np.array([
        [1, 0,             0            ],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch),  np.cos(pitch)],
    ])
    return Ry @ Rx


# Four horizontal cubemap faces — skip top/bottom (little useful texture for SfM)
CUBEMAP_FACES = {
    "front": (0,            0),
    "right": (np.pi / 2,    0),
    "back":  (np.pi,        0),
    "left":  (-np.pi / 2,   0),
}


def equirect_to_perspective(
    equirect: np.ndarray,
    face_size: int,
    fov_deg: float = 90.0,
    yaw: float = 0.0,
    pitch: float = 0.0,
) -> np.ndarray:
    """Sample a perspective view from an equirectangular image.

    Parameters
    ----------
    equirect : H×W×3 equirectangular image
    face_size : output width = height in pixels
    fov_deg : field of view in degrees (90° for standard cubemap)
    yaw, pitch : view direction in radians

    Returns
    -------
    face_size × face_size × 3 perspective image
    """
    h_eq, w_eq = equirect.shape[:2]
    f = face_size / (2.0 * np.tan(np.radians(fov_deg) / 2.0))

    # Build a grid of ray directions in the perspective camera frame
    u = np.arange(face_size, dtype=np.float64) - face_size / 2.0 + 0.5
    v = np.arange(face_size, dtype=np.float64) - face_size / 2.0 + 0.5
    uu, vv = np.meshgrid(u, v)
    rays = np.stack([uu, -vv, np.full_like(uu, f)], axis=-1)  # (H, W, 3)

    # Rotate rays to the world frame
    R = _rotation_matrix(yaw, pitch)
    rays_w = rays @ R.T  # broadcast (H, W, 3) × (3, 3)

    # Convert to spherical coordinates
    x, y, z = rays_w[..., 0], rays_w[..., 1], rays_w[..., 2]
    lon = np.arctan2(x, z)         # [-π, π]
    lat = np.arctan2(y, np.sqrt(x**2 + z**2))  # [-π/2, π/2]

    # Map to equirectangular pixel coords
    px = (lon / np.pi + 1.0) * 0.5 * w_eq  # [0, w_eq]
    py = (0.5 - lat / np.pi) * h_eq         # [0, h_eq]

    # Bilinear sampling via cv2.remap
    map_x = px.astype(np.float32)
    map_y = py.astype(np.float32)
    persp = cv2.remap(
        equirect, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )
    return persp


# ═══════════════════════════════════════════════════════════════════════════
#  Frame extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_frames(video_path: Path, output_dir: Path, fps: float) -> list[Path]:
    """Extract frames from a video at a given FPS using OpenCV.

    Returns list of saved frame paths, sorted.
    If frames already exist in output_dir, skip extraction.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(output_dir.glob("frame_*.png"))
    if existing:
        print(f"[extract] Found {len(existing)} existing frames in {output_dir}, skipping extraction")
        return existing

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, round(src_fps / fps))

    print(f"[extract] Video FPS={src_fps:.2f}, total frames={total}, "
          f"extracting every {interval} frames (≈{fps} FPS)")

    expected = total // interval
    saved: list[Path] = []
    idx = 0
    with tqdm(total=expected, desc="Extracting frames", unit="frame") as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % interval == 0:
                p = output_dir / f"frame_{idx:06d}.png"
                cv2.imwrite(str(p), frame)
                saved.append(p)
                pbar.update(1)
            idx += 1
    cap.release()
    print(f"[extract] Saved {len(saved)} frames to {output_dir}")
    return sorted(saved)


# ═══════════════════════════════════════════════════════════════════════════
#  Cubemap face generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_cubemap_faces(
    frame_paths: list[Path],
    images_dir: Path,
    face_size: int,
    fov_deg: float = 90.0,
) -> tuple[list[str], float]:
    """Convert equirectangular frames to perspective cubemap faces.

    Returns
    -------
    image_names : list of relative image paths (e.g. "frame_000000_front.png")
    focal_length : focal length in pixels for the PINHOLE model
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    focal = face_size / (2.0 * np.tan(np.radians(fov_deg) / 2.0))

    existing = sorted(images_dir.glob("frame_*_*.png"))
    if existing:
        image_names = [p.name for p in existing]
        print(f"[cubemap] Found {len(image_names)} existing cubemap images, skipping generation")
        return image_names, focal

    image_names: list[str] = []
    for fp in tqdm(frame_paths, desc="Generating cubemap faces", unit="frame"):
        equirect = cv2.imread(str(fp))
        if equirect is None:
            tqdm.write(f"[cubemap] WARNING: cannot read {fp}, skipping")
            continue
        for face_name, (yaw, pitch) in CUBEMAP_FACES.items():
            persp = equirect_to_perspective(equirect, face_size, fov_deg, yaw, pitch)
            name = f"{fp.stem}_{face_name}.png"
            cv2.imwrite(str(images_dir / name), persp)
            image_names.append(name)

    print(f"[cubemap] Generated {len(image_names)} perspective images "
          f"(face_size={face_size}, focal={focal:.1f})")
    return image_names, focal


# ═══════════════════════════════════════════════════════════════════════════
#  COLMAP pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_cmd(cmd: list[str], desc: str = "") -> None:
    """Run a shell command, printing it and raising on failure."""
    print(f"\n{'=' * 60}")
    if desc:
        print(f"[COLMAP] {desc}")
    print(f"  $ {' '.join(cmd)}")
    print('=' * 60)
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {' '.join(cmd)}")


def run_colmap_pipeline(
    workspace: Path,
    images_dir: Path,
    face_size: int,
    focal: float,
    gpu_index: int = 4,
) -> Path:
    """Run the COLMAP sparse reconstruction pipeline.

    Steps: feature_extractor → sequential_matcher → mapper → model_converter

    Returns path to the exported PLY file.
    """
    db_path = workspace / "database.db"
    sparse_dir = workspace / "sparse"

    # Clean up previous COLMAP state to avoid corrupted re-runs
    if db_path.exists():
        print(f"[COLMAP] Removing old database: {db_path}")
        db_path.unlink()
    if sparse_dir.exists():
        print(f"[COLMAP] Removing old sparse dir: {sparse_dir}")
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Feature extraction ─────────────────────────────────────────
    run_cmd([
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--ImageReader.camera_model", "PINHOLE",
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_params", f"{focal},{focal},{face_size/2},{face_size/2}",
        "--SiftExtraction.use_gpu", "1",
        "--SiftExtraction.gpu_index", str(gpu_index),
        "--SiftExtraction.max_num_features", "4096",
    ], desc="Feature extraction (SIFT)")

    # ── 2. Exhaustive matching ─────────────────────────────────────────
    run_cmd([
        "colmap", "exhaustive_matcher",
        "--database_path", str(db_path),
        "--SiftMatching.use_gpu", "1",
        "--SiftMatching.gpu_index", str(gpu_index),
    ], desc="Exhaustive matching")

    # ── 3. Sparse reconstruction (mapper) ─────────────────────────────
    run_cmd([
        "colmap", "mapper",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--output_path", str(sparse_dir),
        "--Mapper.ba_refine_focal_length", "1",
        "--Mapper.ba_refine_extra_params", "1",
        "--Mapper.min_num_matches", "10",
        "--Mapper.init_min_num_inliers", "50",
        "--Mapper.init_min_tri_angle", "2",
    ], desc="Sparse reconstruction (mapper)")

    # Find the reconstruction directory (usually sparse/0)
    recon_dirs = sorted(sparse_dir.iterdir())
    if not recon_dirs:
        raise RuntimeError("COLMAP mapper produced no reconstruction!")
    recon_dir = recon_dirs[0]
    print(f"[COLMAP] Using reconstruction: {recon_dir}")

    # ── 4. Export PLY ─────────────────────────────────────────────────
    ply_path = workspace / "point_cloud.ply"
    run_cmd([
        "colmap", "model_converter",
        "--input_path", str(recon_dir),
        "--output_path", str(ply_path),
        "--output_type", "PLY",
    ], desc="Export point cloud → PLY")

    return ply_path


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate COLMAP point cloud from GoPro 360° video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video", type=str, required=True,
        help="Path to equirectangular 360° video (MP4)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output workspace directory (default: gopro360/colmap/output)",
    )
    parser.add_argument(
        "--extract_fps", type=float, default=2.0,
        help="Frame extraction rate (frames per second)",
    )
    parser.add_argument(
        "--face_size", type=int, default=1024,
        help="Perspective face size in pixels (width = height)",
    )
    parser.add_argument(
        "--fov", type=float, default=90.0,
        help="Field of view for perspective faces in degrees",
    )
    parser.add_argument(
        "--gpu", type=int, default=4,
        help="GPU index for COLMAP SIFT extraction/matching",
    )
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        output_dir = Path(__file__).resolve().parent / "output"
    else:
        output_dir = Path(args.output).resolve()

    raw_frames_dir = output_dir / "raw_frames"
    images_dir = output_dir / "images"
    colmap_ws = output_dir / "colmap_ws"
    colmap_ws.mkdir(parents=True, exist_ok=True)

    print(f"Video:       {video_path}")
    print(f"Output:      {output_dir}")
    print(f"Extract FPS: {args.extract_fps}")
    print(f"Face size:   {args.face_size}")
    print(f"FOV:         {args.fov}°")
    print()

    # Step 1: Extract frames
    print("=" * 60)
    print("STEP 1: Extract frames from video")
    print("=" * 60)
    frame_paths = extract_frames(video_path, raw_frames_dir, args.extract_fps)
    if len(frame_paths) < 3:
        print("ERROR: Need at least 3 frames for reconstruction", file=sys.stderr)
        sys.exit(1)

    # Step 2: Convert to perspective cubemap faces
    print("\n" + "=" * 60)
    print("STEP 2: Convert equirectangular → perspective faces")
    print("=" * 60)
    image_names, focal = generate_cubemap_faces(
        frame_paths, images_dir, args.face_size, args.fov
    )

    # Step 3-6: Run COLMAP pipeline
    print("\n" + "=" * 60)
    print("STEP 3-6: COLMAP sparse reconstruction pipeline")
    print("=" * 60)
    ply_path = run_colmap_pipeline(colmap_ws, images_dir, args.face_size, focal, args.gpu)

    print("\n" + "=" * 60)
    print("DONE!")
    print(f"  Point cloud: {ply_path}")
    print(f"  Sparse model: {colmap_ws / 'sparse'}")
    print(f"  Images:       {images_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
