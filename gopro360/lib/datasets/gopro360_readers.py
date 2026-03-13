"""
GoPro 360 scene reader for Gaussian Splatting.

Reads camera poses and intrinsics from a COLMAP binary model produced by
the gopro360 sparse/dense reconstruction pipeline, then returns a
``SceneInfo`` compatible with the Gaussians ``lib/`` stack.

Key difference from KITTI-360
-----------------------------
Each panoramic frame is decomposed into **4 perspective cubemap faces**
(front, right, back, left).  These act as 4 virtual cameras, analogous to
the 2 stereo cameras in KITTI-360:

  * ``num_cams = 4``  (front=0, right=1, back=2, left=3)
  * ``cam`` metadata on each CameraInfo indicates which face it is
  * Train/test split is done **per-frame** — all 4 faces of a frame
    go together into either train or test
  * ``frame_idx`` is per *original panoramic frame* (0..N-1), shared by
    all 4 faces of the same frame — used by CameraPose (frame mode)

Expected directory layout (produced by sparsePointCLoud.py + densePointCloud.py)::

    gopro360/output/
        colmap_ws/sparse/0/   <- original sparse model
        dense/
            sparse/           <- undistorted sparse model (preferred)
            images/           <- undistorted cubemap faces
            fused.ply         <- dense point cloud
        images/               <- original cubemap faces

Image naming convention: ``frame_NNNNNN_<face>.png``
  where <face> is one of: front, right, back, left.
"""

from __future__ import annotations

import os
import sys
import numpy as np
from pathlib import Path
from collections import OrderedDict
from PIL import Image

# Gaussians base types
from lib.datasets.base_readers import (
    SceneInfo, CameraInfo, BasicPointCloud, getNerfppNorm,
)

# Gopro360 utility modules
from lib.utils.colmap_utils import (
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
    qvec2rotmat,
    get_intrinsics,
)
from lib.utils.ply_utils import read_fused_ply, read_sparse_points


# ── Face name → virtual camera ID mapping ─────────────────────────────────
FACE_TO_CAM_ID = {
    "front": 0,
    "right": 1,
    "back":  2,
    "left":  3,
}
NUM_VIRTUAL_CAMS = len(FACE_TO_CAM_ID)


# ═══════════════════════════════════════════════════════════════════════════
#  GoPro 360 Scene Reader
# ═══════════════════════════════════════════════════════════════════════════

def readGoPro360SceneInfo(
    source_path: str,
    point_cloud_path: str = "",
    split_test: int = 8,
    mode: str = "train",
    workspace: str = "",
    mask_dir: str = "",
    **_kwargs,
) -> SceneInfo:
    """Read GoPro 360 COLMAP data and return a ``SceneInfo``.

    Parameters
    ----------
    source_path : str
        Absolute path to ``gopro360/output``.
    point_cloud_path : str
        Explicit path to a PLY file.  If empty, auto-detects fused.ply.
    split_test : int
        Every Nth *frame* goes to the test set (all 4 faces together).
    mode : str
        ``'train'`` or ``'eval'``.
    """
    src = Path(source_path)
    split_test = int(split_test)

    # ── Locate sparse model (cameras, images, points3D) ──────────────
    dense_sparse = src / "dense" / "sparse"
    orig_sparse  = src / "colmap_ws" / "sparse" / "0"

    if dense_sparse.exists() and (dense_sparse / "cameras.bin").exists():
        sparse_dir = dense_sparse
    elif orig_sparse.exists():
        sparse_dir = orig_sparse
    else:
        sys.exit(f"ERROR: No COLMAP sparse model found under {src}")
    print(f"[GoPro360] Sparse model: {sparse_dir}")

    # ── Locate images ────────────────────────────────────────────────
    dense_images = src / "dense" / "images"
    orig_images  = src / "images"

    if dense_images.exists() and any(dense_images.iterdir()):
        images_dir = dense_images
    elif orig_images.exists():
        images_dir = orig_images
    else:
        sys.exit(f"ERROR: No images found under {src}")
    print(f"[GoPro360] Images dir:   {images_dir}")

    # ── Read COLMAP binary model ─────────────────────────────────────
    cameras_bin = read_cameras_binary(str(sparse_dir / "cameras.bin"))
    images_bin  = read_images_binary(str(sparse_dir / "images.bin"))
    print(f"[GoPro360] COLMAP cameras: {len(cameras_bin)},  "
          f"images: {len(images_bin)}")

    # ── Group images by panoramic frame ──────────────────────────────
    # Each panoramic frame has up to 4 face images.
    # Key: frame_name (e.g. "frame_000005"), Value: list of COLMAP images
    frame_groups: OrderedDict[str, list] = OrderedDict()
    sorted_images = sorted(images_bin.values(), key=lambda x: x.name)

    for colmap_img in sorted_images:
        stem = Path(colmap_img.name).stem   # e.g. "frame_000005_front"
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in FACE_TO_CAM_ID:
            frame_name = parts[0]           # e.g. "frame_000005"
            face_name  = parts[1]           # e.g. "front"
        else:
            frame_name = stem
            face_name  = "unknown"

        frame_groups.setdefault(frame_name, []).append(
            (colmap_img, face_name)
        )

    unique_frames = list(frame_groups.keys())
    print(f"[GoPro360] Unique panoramic frames: {len(unique_frames)},  "
          f"faces per frame: {[len(v) for v in frame_groups.values()][:3]}...")

    # ── Build CameraInfo lists ───────────────────────────────────────
    # Split is per-frame: all faces of a frame go to same split
    train_cams: list[CameraInfo] = []
    test_cams:  list[CameraInfo] = []
    uid = 0

    for frame_idx, frame_name in enumerate(unique_frames):
        is_test = (split_test > 0) and (frame_idx % split_test == 0)

        for colmap_img, face_name in frame_groups[frame_name]:
            img_path = str(images_dir / colmap_img.name)
            if not os.path.isfile(img_path):
                print(f"[GoPro360] WARNING: image not found: {img_path}, skipping")
                continue

            cam = cameras_bin[colmap_img.camera_id]
            intr = get_intrinsics(cam)
            width, height = int(cam.width), int(cam.height)

            K = np.array([
                [intr["fx"], 0,          intr["cx"]],
                [0,          intr["fy"], intr["cy"]],
                [0,          0,          1          ],
            ])
            FovX = float(2.0 * np.arctan(width  / (2.0 * intr["fx"])))
            FovY = float(2.0 * np.arctan(height / (2.0 * intr["fy"])))

            # COLMAP stores world-to-camera: R_w2c, t_w2c
            R_w2c = qvec2rotmat(colmap_img.qvec)
            t_w2c = colmap_img.tvec

            # camera-to-world (for metadata / ego_pose)
            c2w = np.eye(4)
            c2w[:3, :3] = R_w2c.T
            c2w[:3, 3]  = -R_w2c.T @ t_w2c

            # 3DGS convention: R stored transposed, T is translation
            R = R_w2c.T   # = c2w[:3,:3]
            T = t_w2c      # w2c translation

            # Virtual camera ID from face name
            cam_id = FACE_TO_CAM_ID.get(face_name, 0)

            image = Image.open(img_path)
            guidance: dict = {}

            # Load mask (sky + roof) if available
            if mask_dir:
                mask_path = Path(workspace) / mask_dir / colmap_img.name if workspace else src / mask_dir / colmap_img.name
                if mask_path.exists():
                    guidance["mask"] = Image.open(str(mask_path)).convert("L")

            cam_info = CameraInfo(
                uid=uid,
                R=R, T=T,
                FovY=FovY, FovX=FovX,
                K=K,
                image=image,
                image_path=img_path,
                image_name=f"{frame_name}_{face_name}",
                width=width,
                height=height,
                metadata={
                    "frame_name": frame_name,
                    "face": face_name,
                    "frame_idx": frame_idx,   # per panoramic frame (shared by all faces)
                    "ego_pose": c2w,
                    "cam": cam_id,            # 0=front, 1=right, 2=back, 3=left
                },
                guidance=guidance,
            )
            uid += 1
            (test_cams if is_test else train_cams).append(cam_info)

    print(f"[GoPro360] Train cameras: {len(train_cams)},  "
          f"Test cameras: {len(test_cams)}")

    if len(train_cams) == 0:
        sys.exit("ERROR: No training cameras found! Check source_path and images.")

    # ── Initial point cloud ──────────────────────────────────────────
    ply_path = ""
    pcd = BasicPointCloud(
        points=np.zeros((0, 3)),
        colors=np.zeros((0, 3)),
        normals=np.zeros((0, 3)),
    )

    if mode == "train":
        # Resolve point cloud path
        if point_cloud_path:
            pcd_abs = Path(point_cloud_path)
            if not pcd_abs.is_absolute() and workspace:
                pcd_abs = (Path(workspace) / point_cloud_path).resolve()
        else:
            pcd_abs = src / "dense" / "fused.ply"

        if pcd_abs.exists():
            pcd = read_fused_ply(str(pcd_abs))
            ply_path = str(pcd_abs)
        else:
            # Fallback: try sparse point cloud PLY
            sparse_ply = src / "colmap_ws" / "point_cloud.ply"
            if sparse_ply.exists():
                pcd = read_fused_ply(str(sparse_ply))
                ply_path = str(sparse_ply)
            else:
                # Last resort: read sparse points from binary model
                pts3d_path = sparse_dir / "points3D.bin"
                if pts3d_path.exists():
                    points3D = read_points3D_binary(str(pts3d_path))
                    pcd = read_sparse_points(points3D)
                    ply_path = str(pts3d_path)
                else:
                    sys.exit(
                        f"ERROR: No point cloud found. Tried:\n"
                        f"  {pcd_abs}\n  {sparse_ply}\n  {pts3d_path}"
                    )

    # ── Normalisation & metadata ─────────────────────────────────────
    nerf_norm = getNerfppNorm(train_cams)

    if pcd.points.shape[0] > 0:
        centre = pcd.points.mean(axis=0)
        dists = np.linalg.norm(pcd.points - centre, axis=1)
        # Use robust radius (99th percentile) to avoid outlier inflation.
        # The raw max can be 100x the useful scene extent due to SfM outliers.
        radius_pct = float(np.percentile(dists, 99))
        radius_max = float(dists.max())
        radius = radius_pct
        print(f"[GoPro360] Scene radius: robust={radius_pct:.2f}  "
              f"(max={radius_max:.2f}, median={np.median(dists):.2f})")

        # Also filter extreme outlier points (>5× robust radius) from the pcd
        keep = dists < 5.0 * radius_pct
        n_removed = int((~keep).sum())
        if n_removed > 0:
            print(f"[GoPro360] Removed {n_removed} outlier points "
                  f"(>{5.0 * radius_pct:.1f} from centre)")
            pcd = BasicPointCloud(
                points=pcd.points[keep],
                colors=pcd.colors[keep],
                normals=pcd.normals[keep],
            )
    else:
        centre = nerf_norm["center"]
        radius = float(nerf_norm["radius"])

    metadata = {
        "num_images":    len(train_cams) + len(test_cams),
        "num_cams":      NUM_VIRTUAL_CAMS,               # 4 faces = 4 virtual cameras
        "num_frames":    len(unique_frames),              # actual panoramic frames
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
