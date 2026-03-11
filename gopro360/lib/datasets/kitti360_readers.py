"""
KITTI-360 scene reader for the Gaussian Splatting pipeline.

Reads calibration files, camera poses, images, and semantic sky masks from
the KITTI-360 dataset layout and returns a ``SceneInfo`` object compatible
with the ``Dataset`` class.
"""

from __future__ import annotations

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image

from lib.config import cfg
from lib.datasets.base_readers import (
    SceneInfo, CameraInfo, BasicPointCloud, getNerfppNorm,
)


# ── Calibration & Pose Helpers ────────────────────────────────────────────

def _parse_perspective(filepath: str) -> dict:
    """Parse ``perspective.txt`` → per-camera projection info.

    Returns ``{cam_id: {'P', 'fx', 'fy', 'cx', 'cy', 'width', 'height'}}``.
    """
    raw: dict[str, list[float]] = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, vals = line.partition(":")
            vals = vals.strip()
            if vals:
                try:
                    raw[key.strip()] = [float(v) for v in vals.split()]
                except ValueError:
                    continue  # skip non-numeric lines (e.g. calib_time)

    out = {}
    for cid in (0, 1):
        tag = f"{cid:02d}"
        P = np.array(raw[f"P_rect_{tag}"]).reshape(3, 4)
        S = raw[f"S_rect_{tag}"]
        out[cid] = dict(
            P=P,
            fx=P[0, 0], fy=P[1, 1],
            cx=P[0, 2], cy=P[1, 2],
            width=int(S[0]), height=int(S[1]),
        )
    return out


def _parse_cam_to_pose(filepath: str) -> dict[str, np.ndarray]:
    """Parse ``calib_cam_to_pose.txt`` → ``{'image_XX': 4×4}``."""
    transforms = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, _, vals = line.partition(":")
            T = np.eye(4)
            T[:3, :] = np.array([float(v) for v in vals.split()]).reshape(3, 4)
            transforms[key.strip()] = T
    return transforms


def _parse_poses_file(filepath: str, *, is_4x4: bool = False) -> dict[int, np.ndarray]:
    """Parse ``cam0_to_world.txt`` (4×4) or ``poses.txt`` (3×4).

    Returns ``{frame_id: 4×4 ndarray}``.
    """
    poses: dict[int, np.ndarray] = {}
    with open(filepath) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            fid = int(parts[0])
            vals = [float(v) for v in parts[1:]]
            if is_4x4:
                T = np.array(vals).reshape(4, 4)
            else:
                T = np.eye(4)
                T[:3, :] = np.array(vals).reshape(3, 4)
            poses[fid] = T
    return poses


# ── Point Cloud I/O ───────────────────────────────────────────────────────

_PLY_VERTEX_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
])


def _read_combined_ply(path: str) -> BasicPointCloud:
    """Read the combined COLMAP + Velodyne PLY (from combine2PointCloud.py)."""
    with open(path, "rb") as f:
        n = 0
        while True:
            hdr = f.readline().decode("ascii", errors="replace").strip()
            if hdr.startswith("element vertex"):
                n = int(hdr.split()[-1])
            if hdr == "end_header":
                break
        buf = np.fromfile(f, dtype=_PLY_VERTEX_DTYPE, count=n)

    pts = np.column_stack([buf["x"], buf["y"], buf["z"]])
    rgb = np.column_stack([buf["red"], buf["green"], buf["blue"]]) / 255.0
    nrm = np.zeros_like(pts)
    print(f"[KITTI360] Loaded {len(pts):,} initial points from {path}")
    return BasicPointCloud(points=pts, colors=rgb, normals=nrm)


# ── KITTI-360 Scene Reader ────────────────────────────────────────────────

# Cityscapes / KITTI-360 semantic label for *sky*
_SKY_LABEL_ID = 23


def readKITTI360SceneInfo(source_path: str, **kwargs) -> SceneInfo:
    """Read KITTI-360 data and return a ``SceneInfo``.

    Expected ``cfg.data`` keys (forwarded via **kwargs):

    =========== ======= ============================================
    Key         Default Description
    =========== ======= ============================================
    drive        …0000  Sequence folder name
    selected_frames [250,490] Start / end frame IDs (inclusive)
    cameras      [0,1]  Camera indices (0=left, 1=right perspective)
    split_test   8      Every Nth frame is held out for testing
    point_cloud_path …  Path to the initial combined PLY
    =========== ======= ============================================
    """
    drive        = kwargs.get("drive", "2013_05_28_drive_0000_sync")
    sel          = kwargs.get("selected_frames", [250, 490])
    cameras      = kwargs.get("cameras", [0, 1])
    split_test   = int(kwargs.get("split_test", 8))
    pcd_cfg_path = kwargs.get("point_cloud_path", None)

    start_f, end_f = int(sel[0]), int(sel[1])
    src = Path(source_path)

    # ── Calibration ───────────────────────────────────────────────────
    calib_dir = src / "calibration"
    persp       = _parse_perspective(str(calib_dir / "perspective.txt"))
    cam_to_pose = _parse_cam_to_pose(str(calib_dir / "calib_cam_to_pose.txt"))

    # ── Poses ─────────────────────────────────────────────────────────
    pose_dir   = src / "data_poses" / drive
    cam0_world = _parse_poses_file(str(pose_dir / "cam0_to_world.txt"), is_4x4=True)
    imu_world  = _parse_poses_file(str(pose_dir / "poses.txt"),         is_4x4=False)

    CAM_DIRS = {0: "image_00", 1: "image_01"}

    # ── Available frames ──────────────────────────────────────────────
    img_root = src / "data_2d_raw" / drive / "image_00" / "data_rect"
    available = sorted(
        int(p.stem)
        for p in img_root.glob("*.png")
        if start_f <= int(p.stem) <= end_f
    )
    available = [f for f in available if f in cam0_world]
    print(f"[KITTI360] {len(available)} frames with known poses "
          f"in [{start_f}, {end_f}]")

    # ── Build CameraInfo lists ────────────────────────────────────────
    train_cams: list[CameraInfo] = []
    test_cams:  list[CameraInfo] = []
    uid = 0
    sem_base = src / "data_2d_semantics" / "train" / drive

    for frame_idx, fid in enumerate(available):
        is_test = (split_test > 0) and (frame_idx % split_test == 0)

        for cam_id in cameras:
            cam_name = CAM_DIRS[cam_id]
            img_path = str(
                src / "data_2d_raw" / drive / cam_name
                / "data_rect" / f"{fid:010d}.png"
            )
            if not os.path.isfile(img_path):
                continue

            # camera-to-world
            if cam_id == 0:
                c2w = cam0_world[fid]
            else:
                if fid not in imu_world:
                    continue
                c2w = imu_world[fid] @ cam_to_pose[cam_name]

            # world-to-camera  (3DGS convention: R transposed)
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3].T
            T = w2c[:3, 3]

            # intrinsics
            cal = persp[cam_id]
            K = np.array([
                [cal["fx"], 0,          cal["cx"]],
                [0,         cal["fy"],  cal["cy"]],
                [0,         0,          1         ],
            ])
            FovX = float(2.0 * np.arctan(cal["width"]  / (2.0 * cal["fx"])))
            FovY = float(2.0 * np.arctan(cal["height"] / (2.0 * cal["fy"])))

            # image
            image = Image.open(img_path)

            # guidance: sky mask from semantic segmentation
            guidance: dict = {}
            sem_path = sem_base / cam_name / "semantic" / f"{fid:010d}.png"
            if sem_path.exists():
                sem = np.array(Image.open(str(sem_path)))
                sky = (sem == _SKY_LABEL_ID).astype(np.uint8) * 255
                guidance["sky_mask"] = Image.fromarray(sky)

            cam_info = CameraInfo(
                uid=uid,
                R=R, T=T,
                FovY=FovY, FovX=FovX,
                K=K,
                image=image,
                image_path=img_path,
                image_name=f"cam{cam_id}/{fid:010d}",
                width=cal["width"],
                height=cal["height"],
                metadata={
                    "frame": fid,
                    "cam": cam_id,
                    "frame_idx": frame_idx,
                    "ego_pose": c2w,
                },
                guidance=guidance,
            )
            uid += 1
            (test_cams if is_test else train_cams).append(cam_info)

    print(f"[KITTI360] Train cameras: {len(train_cams)},  "
          f"Test cameras: {len(test_cams)}")

    # ── Initial point cloud (only needed for training) ─────────────────
    if cfg.mode == 'train':
        if pcd_cfg_path:
            pcd_abs = Path(pcd_cfg_path)
            if not pcd_abs.is_absolute():
                pcd_abs = (Path(cfg.workspace) / pcd_cfg_path).resolve()
        else:
            # Default: combined COLMAP + Velodyne output
            pcd_abs = (
                Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                / ".." / ".." / "combineColmapVelodyne" / "output" / "combined_point_cloud.ply"
            ).resolve()
        if not pcd_abs.exists():
            sys.exit(f"ERROR: Initial point cloud not found at {pcd_abs}\n"
                     "Run combine2PointCloud.py first.")
        pcd = _read_combined_ply(str(pcd_abs))
        ply_path = str(pcd_abs)
    else:
        print("[KITTI360] Skipping initial point cloud load (non-train mode)")
        pcd = BasicPointCloud(
            points=np.zeros((0, 3)),
            colors=np.zeros((0, 3)),
            normals=np.zeros((0, 3)),
        )
        ply_path = ""

    # ── Normalisation & metadata ──────────────────────────────────────
    nerf_norm = getNerfppNorm(train_cams)
    if pcd.points.shape[0] > 0:
        centre = pcd.points.mean(axis=0)
        radius = float(np.linalg.norm(pcd.points - centre, axis=1).max())
    else:
        centre = nerf_norm['center']
        radius = float(nerf_norm['radius'])

    metadata = {
        "num_images":    len(train_cams) + len(test_cams),
        "num_cams":      len(cameras),
        "num_frames":    len(available),
        "scene_center":  centre,
        "scene_radius":  radius,
        "sphere_center": centre,
        "sphere_radius": radius,
    }

    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cams,
        test_cameras=test_cams,
        nerf_normalization=nerf_norm,
        ply_path=ply_path,
        metadata=metadata,
    )
