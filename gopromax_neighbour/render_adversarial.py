#!/usr/bin/env python3
"""
Render adversarial camera path – other side of the road, opposite direction.
=============================================================================

Takes a trained 3D Gaussian Splatting model and the original COLMAP camera
poses, then renders from a **mirrored + reversed** camera trajectory:

  1.  Shift all camera positions laterally to the other side of the road.
  2.  Reverse the frame order (walk in the opposite direction).
  3.  Rotate each camera 180° around the world "up" axis so it faces the
      new walking direction.

Usage
-----
    cd gopromax_neighbour
    python render_adversarial.py \
        --config configs/gopromax_neighbour_1200.yaml \
        --model_root output_version_2 \
        --road_width 6.0 \
        --epoch 1200 \
        --fps 10
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys
from collections import OrderedDict, namedtuple
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import torchvision
from tqdm import tqdm

# ── imports from the self-contained train.py ──────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from train import (  # noqa: E402
    Camera,
    GaussianModel,
    render,
    read_cameras_binary,
    read_images_binary,
    qvec2rotmat,
    get_intrinsics,
    getWorld2View2,
    getProjectionMatrixK,
    load_config,
    FACE_TO_CAM_ID,
)


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _depth_colorize(depth_hw1: np.ndarray) -> np.ndarray:
    d = depth_hw1.squeeze()
    d_min, d_max = d.min(), d.max()
    if d_max - d_min > 1e-6:
        d_norm = ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        d_norm = np.zeros_like(d, dtype=np.uint8)
    colored = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)
    return colored[..., [2, 1, 0]]  # BGR → RGB


def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    return (t.detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════
#  Load trained Gaussians
# ═══════════════════════════════════════════════════════════════════════

def _load_checkpoint(ckpt_dir: str, epoch: int | None, sh_degree: int):
    ckpt_dir = Path(ckpt_dir)
    if epoch is not None:
        ckpt_path = ckpt_dir / f"epoch_{epoch}.pth"
    else:
        ckpt_files = sorted(ckpt_dir.glob("epoch_*.pth"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
        ckpt_path = ckpt_files[-1]

    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(str(ckpt_path), weights_only=False)
    loaded_epoch = state.get("epoch", 0)

    gaussians = GaussianModel(sh_degree=sh_degree)
    gaussians.load_state_dict(state)
    gaussians.active_sh_degree = gaussians.max_sh_degree
    print(f"  {gaussians.num_points:,} Gaussians, SH degree {gaussians.active_sh_degree}")
    return gaussians, loaded_epoch


# ═══════════════════════════════════════════════════════════════════════
#  Build adversarial camera path
# ═══════════════════════════════════════════════════════════════════════

def _parse_face(filename: str):
    stem = Path(filename).stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in FACE_TO_CAM_ID:
        return parts[0], parts[1]
    return stem, "unknown"


def build_adversarial_cameras(
    sparse_dir: str,
    cam_width: int,
    cam_height: int,
    K: np.ndarray,
    FovX: float,
    FovY: float,
    road_width: float = 6.0,
    lateral_sign: float = 1.0,
    max_frames: int = 0,
) -> list[Camera]:
    """Read COLMAP poses, shift laterally, reverse order, rotate 180° yaw.

    Parameters
    ----------
    sparse_dir   : path to COLMAP sparse/ directory with cameras.bin, images.bin
    cam_width, cam_height : image dimensions (used for projection)
    K            : 3×3 intrinsic matrix
    FovX, FovY   : horizontal / vertical field-of-view (radians)
    road_width   : lateral shift in metres  (positive = other side of road)
    lateral_sign : +1 shift to the left of walking direction,
                   -1 shift to the right
    max_frames   : if >0, only use the first N frames
    """
    sparse = Path(sparse_dir)
    cameras_bin = read_cameras_binary(str(sparse / "cameras.bin"))
    images_bin = read_images_binary(str(sparse / "images.bin"))
    print(f"[Adversarial] COLMAP images: {len(images_bin)}")

    # ── Group by frame ────────────────────────────────────────────────
    frame_groups: OrderedDict[str, list] = OrderedDict()
    for img in sorted(images_bin.values(), key=lambda x: x.name):
        frame_name, face_name = _parse_face(img.name)
        R_w2c = qvec2rotmat(img.qvec)
        R_c2w = R_w2c.T
        T_w2c = img.tvec
        frame_groups.setdefault(frame_name, []).append(
            (face_name, R_c2w, T_w2c, img))

    frames = list(frame_groups.items())
    if max_frames > 0:
        frames = frames[:max_frames]
        print(f"[Adversarial] Limiting to first {max_frames} frames")
    n_frames = len(frames)
    print(f"[Adversarial] {n_frames} frames, "
          f"faces/frame: {len(frames[0][1])}")

    # ── Compute frame centres ─────────────────────────────────────────
    centres = []
    for _, faces in frames:
        # All faces of a frame share the same position; use the first one.
        R_c2w, T_w2c = faces[0][1], faces[0][2]
        C = -R_c2w @ T_w2c  # camera centre in world
        centres.append(C)
    centres = np.array(centres)  # (N, 3)

    # ── Walking direction (first → last centre) ──────────────────────
    forward = centres[-1] - centres[0]
    forward /= np.linalg.norm(forward) + 1e-12

    # ── Up direction (average of camera up vectors) ──────────────────
    up_accum = np.zeros(3)
    n = 0
    for _, faces in frames:
        for _, R_c2w, _, _ in faces:
            up_accum += R_c2w @ np.array([0.0, -1.0, 0.0])  # cam Y-down → world up
            n += 1
    up = up_accum / (np.linalg.norm(up_accum) + 1e-12)

    # ── Lateral direction ─────────────────────────────────────────────
    lateral = np.cross(forward, up)
    lateral /= np.linalg.norm(lateral) + 1e-12
    lateral *= lateral_sign

    print(f"  forward : {forward}")
    print(f"  up      : {up}")
    print(f"  lateral : {lateral}  (shift = {road_width:.1f} m)")

    # ── 180° rotation around the up axis (Rodrigues, θ=π) ────────────
    #  R_yaw = 2·(u⊗u) − I
    R_yaw_180 = 2.0 * np.outer(up, up) - np.eye(3)

    # ── Build new cameras: reversed frame order + shift + yaw ────────
    adversarial_cams: list[Camera] = []
    uid = 0
    dummy_image = torch.zeros(3, cam_height, cam_width, dtype=torch.float32)

    for new_idx, (frame_name, faces) in enumerate(reversed(frames)):
        for face_name, R_c2w_old, T_w2c_old, colmap_img in faces:
            # Old camera centre
            C_old = -R_c2w_old @ T_w2c_old

            # Shift to other side of road
            C_new = C_old + road_width * lateral

            # Rotate 180° in yaw
            R_c2w_new = R_yaw_180 @ R_c2w_old

            # New world-to-camera translation
            T_w2c_new = -R_c2w_new.T @ C_new

            cam_id = FACE_TO_CAM_ID.get(face_name, 0)
            image_name = f"adv_{new_idx:04d}_{face_name}"

            cam = Camera(
                uid=uid,
                R=R_c2w_new,
                T=T_w2c_new,
                FoVx=FovX,
                FoVy=FovY,
                K=K.copy(),
                image=dummy_image.clone(),
                image_name=image_name,
                metadata={"cam": cam_id, "face": face_name,
                          "frame_idx": new_idx},
                guidance={},
            )
            adversarial_cams.append(cam)
            uid += 1

    print(f"[Adversarial] Built {len(adversarial_cams)} cameras")
    return adversarial_cams


# ═══════════════════════════════════════════════════════════════════════
#  Panoramic stitching helper (left | front | right | back)
# ═══════════════════════════════════════════════════════════════════════

def _save_panoramic_video(rgb_frames, cam_ids, save_dir, fps):
    face_order = [3, 0, 1, 2]  # left, front, right, back
    per_face = {c: [] for c in face_order}
    for i, cid in enumerate(cam_ids):
        if cid in per_face:
            per_face[cid].append(i)

    counts = [len(per_face[c]) for c in face_order]
    if min(counts) == 0 or len(set(counts)) != 1:
        print(f"  Skipping panoramic (uneven counts: {counts})")
        return

    n = counts[0]
    strips = []
    for t in range(n):
        panels = [rgb_frames[per_face[c][t]] for c in face_order]
        strips.append(np.concatenate(panels, axis=1))

    path = os.path.join(save_dir, "panoramic_adversarial.mp4")
    imageio.mimwrite(path, strips, fps=fps)
    print(f"  Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main rendering loop
# ═══════════════════════════════════════════════════════════════════════

def render_adversarial(cfg, model_root, road_width, lateral_sign,
                       epoch, fps):
    workspace = os.getcwd()
    data_cfg = cfg["data"]

    white_bg = data_cfg.get("white_background", False)
    bg_color = torch.tensor(
        [1, 1, 1] if white_bg else [0, 0, 0],
        dtype=torch.float32, device="cuda")

    model_path = os.path.join(
        workspace, model_root, cfg["task"], cfg["exp_name"])
    trained_model_dir = os.path.join(model_path, "trained_model")

    # ── Resolve source path ───────────────────────────────────────────
    src_path = cfg["source_path"]
    if not os.path.isabs(src_path):
        src_path = os.path.join(workspace, src_path)
    sparse_dir = os.path.join(src_path, "sparse")

    print(f"Model   : {model_path}")
    print(f"Sparse  : {sparse_dir}")
    print(f"Road w. : {road_width} m")

    # ── Read one COLMAP camera to get intrinsics ──────────────────────
    cameras_bin = read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
    first_cam = next(iter(cameras_bin.values()))
    intr = get_intrinsics(first_cam)
    W, H = first_cam.width, first_cam.height
    K = np.array([
        [intr["fx"], 0, intr["cx"]],
        [0, intr["fy"], intr["cy"]],
        [0, 0, 1],
    ], dtype=np.float64)
    FovX = float(2.0 * np.arctan(W / (2.0 * intr["fx"])))
    FovY = float(2.0 * np.arctan(H / (2.0 * intr["fy"])))

    # Apply resolution scale (same as load_camera)
    scale = min(1.0, 1600 / W)
    K_scaled = K.copy()
    K_scaled[:2] *= scale
    W_scaled = int(W * scale)
    H_scaled = int(H * scale)

    # ── Load trained model ────────────────────────────────────────────
    sh_degree = cfg.get("model", {}).get("sh_degree",
                cfg.get("model", {}).get("gaussian", {}).get("sh_degree", 3))
    gaussians, loaded_epoch = _load_checkpoint(
        trained_model_dir, epoch, sh_degree)

    # ── Build adversarial cameras ─────────────────────────────────────
    max_frames = data_cfg.get("max_frames", 0)
    adv_cameras = build_adversarial_cameras(
        sparse_dir=sparse_dir,
        cam_width=W_scaled,
        cam_height=H_scaled,
        K=K_scaled,
        FovX=FovX,
        FovY=FovY,
        road_width=road_width,
        lateral_sign=lateral_sign,
        max_frames=max_frames,
    )

    # ── Output directory ──────────────────────────────────────────────
    save_dir = os.path.join(
        model_path, "adversarial",
        f"ours_epoch_{loaded_epoch}_road{road_width:.0f}m")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Output  : {save_dir}")

    # ── Render ────────────────────────────────────────────────────────
    rgb_frames = []
    depth_frames = []
    cam_ids = []

    with torch.no_grad():
        for cam in tqdm(adv_cameras, desc="Rendering adversarial path"):
            result = render(cam, gaussians, bg_color)

            rgb_np = _tensor_to_uint8(result["rgb"])
            rgb_frames.append(rgb_np)

            depth_np = result["depth"].detach().permute(1, 2, 0).cpu().numpy()
            depth_frames.append(depth_np)

            cam_id = cam.meta.get("cam", 0) if hasattr(cam, "meta") else 0
            cam_ids.append(cam_id)

            # Save per-frame images
            imageio.imwrite(
                os.path.join(save_dir, f"{cam.image_name}_rgb.png"),
                rgb_np)
            imageio.imwrite(
                os.path.join(save_dir, f"{cam.image_name}_depth.png"),
                _depth_colorize(depth_np))

    # ── Per-face videos ───────────────────────────────────────────────
    print("\nSaving videos …")
    unique_cams = sorted(set(cam_ids))
    for cid in unique_cams:
        face_name = {v: k for k, v in FACE_TO_CAM_ID.items()}.get(cid, str(cid))
        face_rgbs = [f for f, c in zip(rgb_frames, cam_ids) if c == cid]
        if face_rgbs:
            path = os.path.join(save_dir, f"adversarial_{face_name}.mp4")
            imageio.mimwrite(path, face_rgbs, fps=fps)
            print(f"  Saved → {path}")

        face_depths = [_depth_colorize(d) for d, c in zip(depth_frames, cam_ids) if c == cid]
        if face_depths:
            path = os.path.join(save_dir, f"adversarial_{face_name}_depth.mp4")
            imageio.mimwrite(path, face_depths, fps=fps)
            print(f"  Saved → {path}")

    # ── Panoramic video ───────────────────────────────────────────────
    _save_panoramic_video(rgb_frames, cam_ids, save_dir, fps)

    print("\nDone.")


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render adversarial camera path (other side of road, "
                    "opposite direction).")
    parser.add_argument(
        "--config", default="configs/gopromax_neighbour_1200.yaml",
        help="Path to YAML config.")
    parser.add_argument(
        "--model_root", default="output_version_2",
        help="Root directory containing the trained model "
             "(default: output_version_2).")
    parser.add_argument(
        "--road_width", type=float, default=6.0,
        help="Lateral shift in metres to the other side of the road "
             "(default: 6.0).")
    parser.add_argument(
        "--lateral_sign", type=float, default=1.0,
        help="+1 = shift left of walking direction, "
             "-1 = shift right (default: +1).")
    parser.add_argument(
        "--epoch", type=int, default=None,
        help="Checkpoint epoch (default: latest).")
    parser.add_argument(
        "--fps", type=int, default=10,
        help="Output video FPS (default: 10).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    render_adversarial(
        cfg,
        model_root=args.model_root,
        road_width=args.road_width,
        lateral_sign=args.lateral_sign,
        epoch=args.epoch,
        fps=args.fps,
    )
