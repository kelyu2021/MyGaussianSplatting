"""
Generate COLMAP sparse reconstruction from KITTI-360 stereo perspective images.

Uses known camera poses and intrinsics to produce a sparse point cloud
suitable for Gaussian Splatting training.  Sky (and other unwanted semantic
classes) is excluded via per-image masks derived from semantic labels.

Required data layout
--------------------
  data_dir/
  ├── calibration/
  │   ├── perspective.txt
  │   └── calib_cam_to_pose.txt          (fallback for poses.txt)
  ├── data_2d_raw/<sequence>/
  │   ├── image_00/data_rect/*.png        (left rectified)
  │   └── image_01/data_rect/*.png        (right rectified)
  ├── data_2d_semantics/train/<sequence>/
  │   ├── image_00/semantic/*.png         (class-ID label maps)
  │   └── image_01/semantic/*.png
  └── data_poses/<sequence>/
      └── cam0_to_world.txt               (preferred)
          or poses.txt + calib_cam_to_pose.txt

Pipeline
--------
  1. Parse calibration → rectified PINHOLE intrinsics & stereo baseline
  2. Load per-frame cam0→world poses
  3. Derive cam1→world from cam0 + baseline
  4. Copy images → COLMAP workspace, generate sky masks
  5. COLMAP feature_extractor  (SIFT, with masks)
  6. COLMAP exhaustive_matcher (or sequential_matcher)
  7. COLMAP point_triangulator (known poses → triangulated 3-D points)
  8. Export sparse model & PLY

Output
------
  <output>/
  ├── images/cam{0,1}/*.png       (images for 3-D Gaussian Splatting)
  ├── sparse/0/                   (COLMAP sparse model)
  └── point_cloud.ply             (exported point cloud)
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
#  KITTI-360 / Cityscapes semantic class IDs to mask out
#    0  = unlabeled
#    1  = ego vehicle
#    2  = rectification border
#    3  = out of roi
#    23 = sky
# ---------------------------------------------------------------------------
DEFAULT_EXCLUDE_LABELS = [0, 1, 2, 3, 23]


# ═══════════════════════════════════════════════════════════════════════════
#  Calibration & pose loading
# ═══════════════════════════════════════════════════════════════════════════

def parse_perspective(filepath: Path) -> dict:
    """Parse perspective.txt → rectified PINHOLE intrinsics & stereo baseline.

    Returns dict with keys: fx, fy, cx, cy, w, h, baseline.
    """
    data: dict[str, list[float]] = {}
    with open(filepath) as f:
        for line in f:
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = key.strip()
            try:
                data[key] = [float(v) for v in val.strip().split()]
            except ValueError:
                pass

    P00 = np.array(data["P_rect_00"]).reshape(3, 4)
    P01 = np.array(data["P_rect_01"]).reshape(3, 4)
    fx, fy = P00[0, 0], P00[1, 1]
    cx, cy = P00[0, 2], P00[1, 2]
    w, h = int(data["S_rect_00"][0]), int(data["S_rect_00"][1])
    # baseline = -P01[0,3] / fx  (positive, ≈ 0.594 m)
    baseline = -P01[0, 3] / fx

    return dict(fx=fx, fy=fy, cx=cx, cy=cy, w=w, h=h, baseline=baseline)


def load_cam0_to_world(filepath: Path) -> dict[int, np.ndarray]:
    """cam0_to_world.txt → { frame_id : 4×4 T_world_cam0 }."""
    poses: dict[int, np.ndarray] = {}
    with open(filepath) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 13:
                continue
            fid = int(parts[0])
            vals = [float(v) for v in parts[1:]]
            T = np.eye(4)
            if len(vals) >= 16:
                T = np.array(vals[:16]).reshape(4, 4)
            else:
                T[:3, :] = np.array(vals[:12]).reshape(3, 4)
            poses[fid] = T
    return poses


def load_poses_txt(filepath: Path, calib_c2p: Path) -> dict[int, np.ndarray]:
    """poses.txt (T_world_pose) + calib_cam_to_pose → { frame : T_world_cam0 }.

    Fallback when cam0_to_world.txt is absent.
    """
    T_pose_cam0: np.ndarray | None = None
    with open(calib_c2p) as f:
        for line in f:
            if line.startswith("image_00:"):
                vals = [float(v) for v in line.split(":")[1].strip().split()]
                T_pose_cam0 = np.eye(4)
                T_pose_cam0[:3, :] = np.array(vals).reshape(3, 4)
                break
    if T_pose_cam0 is None:
        raise ValueError("image_00 entry not found in calib_cam_to_pose.txt")

    poses: dict[int, np.ndarray] = {}
    with open(filepath) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 13:
                continue
            fid = int(parts[0])
            vals = [float(v) for v in parts[1:13]]
            T_wp = np.eye(4)
            T_wp[:3, :] = np.array(vals).reshape(3, 4)
            poses[fid] = T_wp @ T_pose_cam0
    return poses


# ═══════════════════════════════════════════════════════════════════════════
#  Quaternion helpers
# ═══════════════════════════════════════════════════════════════════════════

def _norm_q(q: np.ndarray) -> np.ndarray:
    """Normalise quaternion; enforce qw ≥ 0 for canonical form."""
    n = np.linalg.norm(q)
    q = q / n if n > 0 else q
    return q if q[0] >= 0 else -q


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → (qw, qx, qy, qz) unit quaternion."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        return _norm_q(np.array([
            0.25 * s,
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
        ]))
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return _norm_q(np.array([
            (R[2, 1] - R[1, 2]) / s, 0.25 * s,
            (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s,
        ]))
    if R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return _norm_q(np.array([
            (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
            0.25 * s, (R[1, 2] + R[2, 1]) / s,
        ]))
    s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    return _norm_q(np.array([
        (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
        (R[1, 2] + R[2, 1]) / s, 0.25 * s,
    ]))


# ═══════════════════════════════════════════════════════════════════════════
#  COLMAP text-model writers
# ═══════════════════════════════════════════════════════════════════════════

def write_cameras_txt(path: Path, cam: dict, cam_ids: list[int]) -> None:
    """Write cameras.txt with PINHOLE model."""
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cam_ids)}\n")
        for cid in cam_ids:
            f.write(
                f"{cid} PINHOLE {cam['w']} {cam['h']} "
                f"{cam['fx']:.10f} {cam['fy']:.10f} "
                f"{cam['cx']:.10f} {cam['cy']:.10f}\n"
            )


def write_images_txt(path: Path, entries: list[dict]) -> None:
    """Write images.txt with known poses (COLMAP convention: T_cam_world)."""
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(entries)}\n")
        for e in entries:
            f.write(
                f"{e['image_id']} "
                f"{e['qw']:.12f} {e['qx']:.12f} "
                f"{e['qy']:.12f} {e['qz']:.12f} "
                f"{e['tx']:.12f} {e['ty']:.12f} {e['tz']:.12f} "
                f"{e['camera_id']} {e['name']}\n"
            )
            f.write("\n")  # empty POINTS2D line


def write_points3d_txt(path: Path) -> None:
    """Write an empty points3D.txt (triangulator will populate it)."""
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, "
                "TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0\n")


# ═══════════════════════════════════════════════════════════════════════════
#  Database helpers
# ═══════════════════════════════════════════════════════════════════════════

def update_db_cameras(db_path: Path, cam: dict) -> None:
    """Override auto-estimated intrinsics with the known KITTI-360 values."""
    blob = struct.pack("dddd", cam["fx"], cam["fy"], cam["cx"], cam["cy"])
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "UPDATE cameras SET model=1, width=?, height=?, params=?, "
        "prior_focal_length=1",
        (cam["w"], cam["h"], blob),
    )
    conn.commit()
    n = conn.execute("SELECT changes()").fetchone()[0]
    conn.close()
    print(f"  Updated {n} camera(s) with known intrinsics")


def get_db_images(db_path: Path) -> dict[str, tuple[int, int]]:
    """Return { image_name : (image_id, camera_id) } from the COLMAP DB."""
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("SELECT image_id, name, camera_id FROM images").fetchall()
    conn.close()
    return {name: (iid, cid) for iid, name, cid in rows}


# ═══════════════════════════════════════════════════════════════════════════
#  Mask generation (semantic label → binary mask)
# ═══════════════════════════════════════════════════════════════════════════

def generate_mask(
    sem_path: Path,
    out_path: Path,
    exclude: list[int],
    size_wh: tuple[int, int],
) -> None:
    """Create a COLMAP-compatible mask from a KITTI-360 semantic label image.

    Output PNG: 0 = masked (feature extraction skipped), 255 = valid.
    """
    from PIL import Image as PILImage

    if sem_path.exists():
        sem = np.array(PILImage.open(sem_path))
        mask = np.full(sem.shape[:2], 255, dtype=np.uint8)
        for lab in exclude:
            mask[sem == lab] = 0
    else:
        # No semantic map → keep all pixels
        mask = np.full((size_wh[1], size_wh[0]), 255, dtype=np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(mask).save(str(out_path))


# ═══════════════════════════════════════════════════════════════════════════
#  COLMAP command runner
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
#  Known-model builder (fallback when COLMAP DB is not yet created)
# ═══════════════════════════════════════════════════════════════════════════

def _write_known_model_pre(
    model_dir: Path,
    cam: dict,
    frames: list[int],
    poses: dict[int, np.ndarray],
    T_cam0_cam1: np.ndarray,
) -> None:
    """Write a known-pose model with sequential image IDs (pre-DB fallback)."""
    entries: list[dict] = []
    iid = 1
    for fid in frames:
        fname = f"{fid:010d}.png"
        T_w_c0 = poses[fid]
        for tag, T_off, cid in [("cam0", np.eye(4), 1), ("cam1", T_cam0_cam1, 2)]:
            T_c_w = np.linalg.inv(T_w_c0 @ T_off)
            q = rotmat_to_quat(T_c_w[:3, :3])
            t = T_c_w[:3, 3]
            entries.append(dict(
                image_id=iid, qw=q[0], qx=q[1], qy=q[2], qz=q[3],
                tx=t[0], ty=t[1], tz=t[2],
                camera_id=cid, name=f"{tag}/{fname}",
            ))
            iid += 1

    write_cameras_txt(model_dir / "cameras.txt", cam, [1, 2])
    write_images_txt(model_dir / "images.txt", entries)
    write_points3d_txt(model_dir / "points3D.txt")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="COLMAP sparse reconstruction from KITTI-360 stereo images "
                    "(known poses, semantic sky masking).",
    )
    ap.add_argument("--data_dir", default="../data/kitti360",
                    help="Root of the KITTI-360 data tree.")
    ap.add_argument("--sequence", default="2013_05_28_drive_0000_sync",
                    help="Sequence directory name.")
    ap.add_argument("--output", default="output/colmap_ws",
                    help="Output workspace directory.")
    ap.add_argument("--start_frame", type=int, default=-1,
                    help="First frame index (-1 = auto).")
    ap.add_argument("--end_frame", type=int, default=-1,
                    help="Last frame index (-1 = auto).")
    ap.add_argument("--exclude_labels", type=int, nargs="+",
                    default=DEFAULT_EXCLUDE_LABELS,
                    help="Semantic label IDs to mask out "
                         "(default: 0 1 2 3 23 = unlabeled/ego/border/oor/sky).")
    ap.add_argument("--use_gpu", type=int, default=1, choices=[0, 1],
                    help="Use GPU for SIFT & matching (1=yes, 0=no).")
    ap.add_argument("--matcher", default="exhaustive",
                    choices=["exhaustive", "sequential"],
                    help="Feature matching strategy.")
    ap.add_argument("--colmap_exe", default="colmap",
                    help="Path to the COLMAP executable.")
    ap.add_argument("--skip_colmap", action="store_true",
                    help="Prepare workspace only; do not run COLMAP.")
    args = ap.parse_args()

    # ── resolve paths ─────────────────────────────────────────────────
    data = Path(args.data_dir).resolve()
    script_dir = Path(__file__).parent.resolve()
    out = (
        Path(args.output).resolve()
        if Path(args.output).is_absolute()
        else (script_dir / args.output).resolve()
    )
    seq = args.sequence

    img00 = data / "data_2d_raw" / seq / "image_00" / "data_rect"
    img01 = data / "data_2d_raw" / seq / "image_01" / "data_rect"
    sem00 = data / "data_2d_semantics" / "train" / seq / "image_00" / "semantic"
    sem01 = data / "data_2d_semantics" / "train" / seq / "image_01" / "semantic"
    persp = data / "calibration" / "perspective.txt"
    calib_c2p = data / "calibration" / "calib_cam_to_pose.txt"
    c2w_file = data / "data_poses" / seq / "cam0_to_world.txt"
    poses_file = data / "data_poses" / seq / "poses.txt"

    # ── validate required paths ───────────────────────────────────────
    for p, tag in [
        (persp, "calibration/perspective.txt"),
        (img00, "data_2d_raw image_00"),
        (img01, "data_2d_raw image_01"),
    ]:
        if not p.exists():
            sys.exit(f"ERROR: {tag} not found at {p}")

    # ── calibration ───────────────────────────────────────────────────
    print("Loading calibration …")
    cam = parse_perspective(persp)
    print(f"  Rectified {cam['w']}×{cam['h']}  "
          f"fx={cam['fx']:.2f}  fy={cam['fy']:.2f}  "
          f"cx={cam['cx']:.2f}  cy={cam['cy']:.2f}")
    print(f"  Stereo baseline: {cam['baseline']:.4f} m")

    # ── poses ─────────────────────────────────────────────────────────
    print("Loading poses …")
    if c2w_file.exists():
        poses = load_cam0_to_world(c2w_file)
        print(f"  {len(poses)} cam0→world poses from cam0_to_world.txt")
    elif poses_file.exists() and calib_c2p.exists():
        poses = load_poses_txt(poses_file, calib_c2p)
        print(f"  {len(poses)} poses via poses.txt + calib_cam_to_pose")
    else:
        sys.exit("ERROR: No pose data found. "
                 "Need cam0_to_world.txt or poses.txt + calib_cam_to_pose.txt")

    # ── determine usable frames ───────────────────────────────────────
    f00 = {int(p.stem) for p in img00.glob("*.png")}
    f01 = {int(p.stem) for p in img01.glob("*.png")}
    common = sorted(f00 & f01 & set(poses.keys()))
    if not common:
        sys.exit("ERROR: No frames with stereo images AND poses.")

    lo = args.start_frame if args.start_frame >= 0 else common[0]
    hi = args.end_frame if args.end_frame >= 0 else common[-1]
    frames = [f for f in common if lo <= f <= hi]
    print(f"  {len(frames)} usable frames [{frames[0]} … {frames[-1]}]")

    # ── stereo baseline transform ─────────────────────────────────────
    # cam1 is shifted +x by baseline in the rectified cam0 frame
    T_cam0_cam1 = np.eye(4)
    T_cam0_cam1[0, 3] = cam["baseline"]

    # ── create workspace directories ──────────────────────────────────
    ws_img = out / "images"
    ws_mask = out / "masks"
    ws_known = out / "sparse" / "known_model"
    ws_out = out / "sparse" / "0"
    db = out / "database.db"

    for d in [
        ws_img / "cam0", ws_img / "cam1",
        ws_mask / "cam0", ws_mask / "cam1",
        ws_known, ws_out,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    if db.exists():
        db.unlink()

    # ── copy images & generate masks ──────────────────────────────────
    print("Preparing images & sky masks …")
    has_sem0, has_sem1 = sem00.exists(), sem01.exists()
    if not has_sem0:
        print(f"  WARNING: semantic labels for image_00 not found at {sem00}")
    if not has_sem1:
        print(f"  WARNING: semantic labels for image_01 not found at {sem01}")

    n_masks = 0
    for fid in frames:
        fname = f"{fid:010d}.png"
        for tag, src_dir, sem_dir, has_sem in [
            ("cam0", img00, sem00, has_sem0),
            ("cam1", img01, sem01, has_sem1),
        ]:
            # Image
            dst = ws_img / tag / fname
            if not dst.exists():
                shutil.copy2(str(src_dir / fname), str(dst))

            # Mask  (COLMAP expects <image_name>.png appended)
            msk = ws_mask / tag / (fname + ".png")
            if not msk.exists():
                generate_mask(
                    sem_dir / fname, msk,
                    args.exclude_labels, (cam["w"], cam["h"]),
                )
                n_masks += 1

    total_imgs = len(frames) * 2
    print(f"  {total_imgs} images ready, {n_masks} new masks generated")
    print(f"  Excluded semantic labels: {args.exclude_labels}")

    # ── early exit if --skip_colmap ───────────────────────────────────
    if args.skip_colmap:
        _write_known_model_pre(ws_known, cam, frames, poses, T_cam0_cam1)
        print(f"\n--skip_colmap: workspace prepared at {out}")
        print("Run COLMAP manually:")
        print(f"  colmap feature_extractor --database_path {db} "
              f"--image_path {ws_img} --ImageReader.mask_path {ws_mask} "
              "--ImageReader.camera_model PINHOLE "
              "--ImageReader.single_camera_per_folder 1")
        print(f"  colmap exhaustive_matcher --database_path {db}")
        print(f"  colmap point_triangulator --database_path {db} "
              f"--image_path {ws_img} --input_path {ws_known} "
              f"--output_path {ws_out}")
        return

    # ── verify COLMAP is installed ────────────────────────────────────
    colmap = args.colmap_exe
    try:
        subprocess.run([colmap, "help"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        _write_known_model_pre(ws_known, cam, frames, poses, T_cam0_cam1)
        print(f"\nERROR: COLMAP not found ('{colmap}').")
        print("  Install → https://colmap.github.io/install.html")
        print(f"  Workspace prepared at {out} — run COLMAP manually.")
        sys.exit(1)

    gpu = str(args.use_gpu)

    # ── Step 1: Feature extraction ────────────────────────────────────
    run_colmap([
        colmap, "feature_extractor",
        "--database_path", str(db),
        "--image_path", str(ws_img),
        "--ImageReader.mask_path", str(ws_mask),
        "--ImageReader.camera_model", "PINHOLE",
        "--ImageReader.single_camera_per_folder", "1",
        "--ImageReader.camera_params",
        f"{cam['fx']},{cam['fy']},{cam['cx']},{cam['cy']}",
        "--FeatureExtraction.use_gpu", gpu,
        "--SiftExtraction.max_num_features", "8192",
    ], "Feature extraction (sky-masked SIFT)")

    # Lock known intrinsics (override any COLMAP auto-estimate)
    print("Locking known intrinsics in database …")
    update_db_cameras(db, cam)

    # ── Step 2: Feature matching ──────────────────────────────────────
    if args.matcher == "exhaustive":
        run_colmap([
            colmap, "exhaustive_matcher",
            "--database_path", str(db),
            "--FeatureMatching.use_gpu", gpu,
        ], "Exhaustive feature matching")
    else:
        run_colmap([
            colmap, "sequential_matcher",
            "--database_path", str(db),
            "--FeatureMatching.use_gpu", gpu,
            "--SequentialMatching.overlap", "15",
        ], "Sequential feature matching (overlap=15)")

    # ── Step 3: Build known-pose model using DB image IDs ─────────────
    print("Building known-pose model for triangulation …")
    db_imgs = get_db_images(db)

    entries: list[dict] = []
    for fid in frames:
        fname = f"{fid:010d}.png"
        T_w_c0 = poses[fid]
        for tag, T_offset in [("cam0", np.eye(4)), ("cam1", T_cam0_cam1)]:
            name = f"{tag}/{fname}"
            if name not in db_imgs:
                continue
            iid, cid = db_imgs[name]
            T_w_c = T_w_c0 @ T_offset
            T_c_w = np.linalg.inv(T_w_c)
            q = rotmat_to_quat(T_c_w[:3, :3])
            t = T_c_w[:3, 3]
            entries.append(dict(
                image_id=iid, qw=q[0], qx=q[1], qy=q[2], qz=q[3],
                tx=t[0], ty=t[1], tz=t[2],
                camera_id=cid, name=name,
            ))

    cam_ids_used = sorted({e["camera_id"] for e in entries})
    write_cameras_txt(ws_known / "cameras.txt", cam, cam_ids_used)
    write_images_txt(ws_known / "images.txt", entries)
    write_points3d_txt(ws_known / "points3D.txt")
    print(f"  {len(entries)} registered images, "
          f"{len(cam_ids_used)} camera model(s)")

    # ── Step 4: Triangulate 3-D points ────────────────────────────────
    run_colmap([
        colmap, "point_triangulator",
        "--database_path", str(db),
        "--image_path", str(ws_img),
        "--input_path", str(ws_known),
        "--output_path", str(ws_out),
        "--Mapper.ba_refine_focal_length", "0",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0",
    ], "Point triangulation (known poses, fixed intrinsics)")

    # ── Step 5: Export PLY ────────────────────────────────────────────
    ply = out / "point_cloud.ply"
    run_colmap([
        colmap, "model_converter",
        "--input_path", str(ws_out),
        "--output_path", str(ply),
        "--output_type", "PLY",
    ], "Export point cloud to PLY")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  DONE")
    print(f"  Sparse model : {ws_out}")
    print(f"  Point cloud  : {ply}")
    print(f"  Images       : {ws_img}")
    print(f"{'═' * 60}")
    print(f"\nFor Gaussian Splatting, point to: {out}")


if __name__ == "__main__":
    main()
