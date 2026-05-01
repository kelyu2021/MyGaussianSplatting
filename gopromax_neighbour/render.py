"""
GoPro Max Neighbour – Rendering / Evaluation Script
====================================================

Renders trained Gaussian models and saves per-image outputs plus
optional trajectory videos.

Usage
-----
    # Evaluate (save per-image renders for train & test sets)
    python render.py --config configs/gopromax_neighbour.yaml --mode evaluate

    # Trajectory video (all frames sorted by ID)
    python render.py --config configs/gopromax_neighbour.yaml --mode trajectory

    # Optionally specify a checkpoint epoch
    python render.py --config configs/gopromax_neighbour.yaml --mode evaluate --epoch 180

Outputs
-------
    evaluate mode:
        {model_path}/train/ours_epoch_{num}/{name}_rgb.png
        {model_path}/train/ours_epoch_{num}/{name}_gt.png
        {model_path}/train/ours_epoch_{num}/{name}_depth.png
        {model_path}/train/ours_epoch_{num}/{name}_diff.png
        (same under test/)

    trajectory mode:
        {model_path}/trajectory/ours_epoch_{num}/color.mp4
        {model_path}/trajectory/ours_epoch_{num}/color_gt.mp4
        {model_path}/trajectory/ours_epoch_{num}/depth.mp4
        {model_path}/trajectory/ours_epoch_{num}/diff.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import torchvision
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════
#  Path setup – import everything from the self-contained train.py
# ═══════════════════════════════════════════════════════════════════════════
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from train import (                                                      # noqa: E402
    GaussianModel,
    Camera,
    load_camera,
    load_config,
    read_scene,
    render,
    psnr,
    l1_loss,
    ssim,
    FACE_TO_CAM_ID,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Visualisation helpers
# ═══════════════════════════════════════════════════════════════════════════

def _depth_colorize(depth_hw1: np.ndarray) -> np.ndarray:
    """Colorize a [H, W, 1] depth array → [H, W, 3] uint8 (RGB)."""
    d = depth_hw1.squeeze()
    d_min, d_max = d.min(), d.max()
    if d_max - d_min > 1e-6:
        d_norm = ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        d_norm = np.zeros_like(d, dtype=np.uint8)
    # Stack to 3 channels for grayscale RGB
    gray = np.stack([d_norm]*3, axis=-1)
    return gray


def _diff_colorize(diff_hw1: np.ndarray) -> np.ndarray:
    """Colorize a [H, W, 1] error map → [H, W, 3] uint8 (RGB)."""
    d = diff_hw1.squeeze()
    d_min, d_max = d.min(), d.max()
    if d_max - d_min > 1e-6:
        d_norm = ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        d_norm = np.zeros_like(d, dtype=np.uint8)
    colored = cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)
    return colored[..., [2, 1, 0]]  # BGR → RGB


def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """[C, H, W] float tensor → [H, W, C] uint8 numpy."""
    return (t.detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)


def _compute_diff(rgb: torch.Tensor, rgb_gt: torch.Tensor,
                  mask: torch.Tensor | None = None) -> np.ndarray:
    """Compute per-pixel squared-error diff map → [H, W, 1] float."""
    rgb_np = rgb.detach().cpu().permute(1, 2, 0).numpy()
    gt_np = rgb_gt.detach().cpu().permute(1, 2, 0).numpy()
    if mask is not None:
        m = mask.detach().cpu().numpy()
        if m.ndim == 2:
            m = m[..., None]
        elif m.ndim == 3 and m.shape[0] in (1, 3):
            m = m.transpose(1, 2, 0)
            if m.shape[-1] == 3:
                m = m[..., :1]
        rgb_np = rgb_np * m
        gt_np = gt_np * m
    diff = ((rgb_np - gt_np) ** 2).sum(axis=-1, keepdims=True)
    return diff


def _save_video(frames: list, path: str, fps: int,
                cams: list | None = None,
                visualize_func=None):
    """Save a list of frames as MP4(s), optionally split by camera."""
    if not frames:
        return

    if cams is None or len(set(cams)) <= 1:
        if visualize_func is not None:
            frames = [visualize_func(f) for f in frames]
        imageio.mimwrite(path, frames, fps=fps)
        return

    unique_cams = sorted(set(cams))
    base, ext = os.path.splitext(path)
    for cam in unique_cams:
        cam_frames = [f for f, c in zip(frames, cams) if c == cam]
        if visualize_func is not None:
            cam_frames = [visualize_func(f) for f in cam_frames]
        imageio.mimwrite(f"{base}_{cam}{ext}", cam_frames, fps=fps)


# ═══════════════════════════════════════════════════════════════════════════
#  Load trained model
# ═══════════════════════════════════════════════════════════════════════════

def _load_checkpoint(trained_model_dir: str, epoch: int | None,
                     sh_degree: int) -> tuple[GaussianModel, int]:
    """Load a GaussianModel from a checkpoint.

    Returns (gaussians, loaded_epoch).
    """
    ckpt_dir = Path(trained_model_dir)
    if epoch is not None:
        ckpt_path = ckpt_dir / f"epoch_{epoch}.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        # Find latest checkpoint
        ckpt_files = sorted(ckpt_dir.glob("epoch_*.pth"))
        if not ckpt_files:
            raise FileNotFoundError(
                f"No checkpoints found in {trained_model_dir}")
        ckpt_path = ckpt_files[-1]

    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(str(ckpt_path), weights_only=False)
    loaded_epoch = state.get("epoch", 0)

    gaussians = GaussianModel(sh_degree=sh_degree)
    gaussians.load_state_dict(state)
    gaussians.active_sh_degree = gaussians.max_sh_degree
    print(f"  Loaded epoch {loaded_epoch}, "
          f"{gaussians.num_points:,} Gaussians, "
          f"SH degree {gaussians.active_sh_degree}")
    return gaussians, loaded_epoch


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluate mode – per-image rendering for train / test sets
# ═══════════════════════════════════════════════════════════════════════════

def render_sets(cfg: dict, epoch: int | None = None,
                skip_train: bool = False, skip_test: bool = False):
    """Render and save per-image outputs for train and test camera sets."""
    workspace = os.getcwd()
    data_cfg = cfg["data"]

    white_bg = data_cfg.get("white_background", False)
    bg_color = torch.tensor(
        [1, 1, 1] if white_bg else [0, 0, 0],
        dtype=torch.float32, device="cuda")

    model_path = os.path.join(
        workspace, "output", cfg["task"], cfg["exp_name"])
    trained_model_dir = os.path.join(model_path, "trained_model")

    with torch.no_grad():
        # Load scene data
        scene_info = read_scene(
            source_path=cfg["source_path"],
            point_cloud_path=data_cfg["point_cloud_path"],
            images_dir=data_cfg["images"],
            mask_dir=data_cfg.get("mask_dir", ""),
            split_test=data_cfg.get("split_test", 8),
            workspace=workspace,
        )

        # Load trained model
        sh_degree = cfg.get("model", {}).get("sh_degree",
                    cfg.get("model", {}).get("gaussian", {}).get("sh_degree", 3))
        gaussians, loaded_epoch = _load_checkpoint(
            trained_model_dir, epoch, sh_degree)

        # Build camera sets
        splits = []
        if not skip_train:
            print("Loading training cameras …")
            train_cameras = [
                load_camera(ci) for ci in tqdm(scene_info.train_cameras)]
            splits.append(("train", train_cameras))
        if not skip_test:
            print("Loading test cameras …")
            test_cameras = [
                load_camera(ci) for ci in tqdm(scene_info.test_cameras)]
            splits.append(("test", test_cameras))

        times: list[float] = []

        for split_name, cameras in splits:
            save_dir = os.path.join(
                model_path, split_name,
                f"ours_epoch_{loaded_epoch}",
            )
            os.makedirs(save_dir, exist_ok=True)

            for idx, camera in enumerate(tqdm(cameras,
                                              desc=f"Rendering {split_name}")):
                torch.cuda.synchronize()
                t0 = time.time()

                result = render(camera, gaussians, bg_color)

                torch.cuda.synchronize()
                t1 = time.time()
                times.append((t1 - t0) * 1000)

                name = camera.image_name

                # ── RGB ───────────────────────────────────────────
                torchvision.utils.save_image(
                    result['rgb'],
                    os.path.join(save_dir, f'{name}_rgb.png'),
                )

                # ── Ground truth ──────────────────────────────────
                torchvision.utils.save_image(
                    camera.original_image[:3],
                    os.path.join(save_dir, f'{name}_gt.png'),
                )

                # ── Depth ─────────────────────────────────────────
                depth = result['depth'].detach().permute(1, 2, 0).cpu().numpy()
                imageio.imwrite(
                    os.path.join(save_dir, f'{name}_depth.png'),
                    _depth_colorize(depth),
                )

                # ── Diff (squared error) ──────────────────────────
                mask = camera.guidance.get("mask")
                diff = _compute_diff(result['rgb'], camera.original_image[:3], mask)
                imageio.imwrite(
                    os.path.join(save_dir, f'{name}_diff.png'),
                    _diff_colorize(diff),
                )

        if times:
            print(f"\nRendering times (ms): "
                  f"mean={sum(times[1:]) / max(len(times) - 1, 1):.2f}")


# ═══════════════════════════════════════════════════════════════════════════
#  Panoramic video helper
# ═══════════════════════════════════════════════════════════════════════════

def _save_panoramic(rgbs, rgbs_gt, depths, diffs, cams_list,
                    save_dir, fps):
    """Build panoramic videos with faces in equirectangular order.

    Layout (horizontal strip matching the original panorama):
        ┌──────┬───────┬───────┬──────┐
        │ left │ front │ right │ back │
        └──────┴───────┴───────┴──────┘
    """
    # cam_id: 0=front, 1=right, 2=back, 3=left
    # Equirectangular order: left(3), front(0), right(1), back(2)
    face_order = [3, 0, 1, 2]
    per_face = {c: [] for c in face_order}
    for i, cam_id in enumerate(cams_list):
        if cam_id in per_face:
            per_face[cam_id].append(i)

    counts = [len(per_face[c]) for c in face_order]
    if min(counts) == 0 or len(set(counts)) != 1:
        print(f"  Skipping panoramic video (uneven face counts: {counts})")
        return

    n_frames = counts[0]

    def _build_strip(frames_list):
        strip_frames = []
        for t in range(n_frames):
            panels = [frames_list[per_face[c][t]] for c in face_order]
            strip_frames.append(np.concatenate(panels, axis=1))
        return strip_frames

    print("Saving panoramic videos …")
    pano_rgb = _build_strip(rgbs)
    imageio.mimwrite(os.path.join(save_dir, 'panoramic_color.mp4'), pano_rgb, fps=fps)
    print(f"  Saved → {os.path.join(save_dir, 'panoramic_color.mp4')}")

    pano_gt = _build_strip(rgbs_gt)
    imageio.mimwrite(os.path.join(save_dir, 'panoramic_color_gt.mp4'), pano_gt, fps=fps)
    print(f"  Saved → {os.path.join(save_dir, 'panoramic_color_gt.mp4')}")

    pano_depth = _build_strip([_depth_colorize(d) for d in depths])
    imageio.mimwrite(os.path.join(save_dir, 'panoramic_depth.mp4'), pano_depth, fps=fps)
    print(f"  Saved → {os.path.join(save_dir, 'panoramic_depth.mp4')}")

    pano_diff = _build_strip([_diff_colorize(d) for d in diffs])
    imageio.mimwrite(os.path.join(save_dir, 'panoramic_diff.mp4'), pano_diff, fps=fps)
    print(f"  Saved → {os.path.join(save_dir, 'panoramic_diff.mp4')}")


# ═══════════════════════════════════════════════════════════════════════════
#  Trajectory mode – full video fly-through
# ═══════════════════════════════════════════════════════════════════════════

def render_trajectory(cfg: dict, epoch: int | None = None, fps: int = 10):
    """Render all frames in order and produce trajectory videos."""
    workspace = os.getcwd()
    data_cfg = cfg["data"]

    white_bg = data_cfg.get("white_background", False)
    bg_color = torch.tensor(
        [1, 1, 1] if white_bg else [0, 0, 0],
        dtype=torch.float32, device="cuda")

    model_path = os.path.join(
        workspace, "output", cfg["task"], cfg["exp_name"])
    trained_model_dir = os.path.join(model_path, "trained_model")

    with torch.no_grad():
        # Load scene data
        scene_info = read_scene(
            source_path=cfg["source_path"],
            point_cloud_path=data_cfg["point_cloud_path"],
            images_dir=data_cfg["images"],
            mask_dir=data_cfg.get("mask_dir", ""),
            split_test=data_cfg.get("split_test", 8),
            workspace=workspace,
        )

        # Load trained model
        sh_degree = cfg.get("model", {}).get("sh_degree",
                    cfg.get("model", {}).get("gaussian", {}).get("sh_degree", 3))
        gaussians, loaded_epoch = _load_checkpoint(
            trained_model_dir, epoch, sh_degree)

        save_dir = os.path.join(
            model_path, 'trajectory',
            f"ours_epoch_{loaded_epoch}",
        )
        os.makedirs(save_dir, exist_ok=True)

        # Build all cameras (train + test), sorted by uid
        print("Loading all cameras …")
        all_cam_infos = scene_info.train_cameras + scene_info.test_cameras
        all_cameras = [load_camera(ci) for ci in tqdm(all_cam_infos)]
        all_cameras = sorted(all_cameras, key=lambda c: c.id)

        rgbs_gt, rgbs = [], []
        depths, diffs = [], []
        cams_list = []

        for idx, camera in enumerate(tqdm(all_cameras,
                                          desc="Rendering Trajectory")):
            result = render(camera, gaussians, bg_color)

            cam_id = camera.meta.get('cam', 0) if hasattr(camera, 'meta') else 0
            cams_list.append(cam_id)
            name = camera.image_name

            # Accumulate frames
            rgbs_gt.append(_tensor_to_uint8(camera.original_image[:3]))
            rgbs.append(_tensor_to_uint8(result['rgb']))

            depth = result['depth'].detach().permute(1, 2, 0).cpu().numpy()
            depths.append(depth)

            mask = camera.guidance.get("mask")
            diff = _compute_diff(result['rgb'], camera.original_image[:3], mask)
            diffs.append(diff)

            # Save per-frame images
            torchvision.utils.save_image(
                result['rgb'],
                os.path.join(save_dir, f'{name}_rgb.png'),
            )
            torchvision.utils.save_image(
                camera.original_image[:3],
                os.path.join(save_dir, f'{name}_gt.png'),
            )
            imageio.imwrite(
                os.path.join(save_dir, f'{name}_depth.png'),
                _depth_colorize(depth),
            )
            imageio.imwrite(
                os.path.join(save_dir, f'{name}_diff.png'),
                _diff_colorize(diff),
            )

        # ── Save videos ───────────────────────────────────────────
        print("\nSaving trajectory videos …")
        _save_video(rgbs_gt,  os.path.join(save_dir, 'color_gt.mp4'),  fps, cams_list)
        _save_video(rgbs,     os.path.join(save_dir, 'color.mp4'),     fps, cams_list)
        _save_video(depths,   os.path.join(save_dir, 'depth.mp4'),     fps, cams_list,
                    visualize_func=_depth_colorize)
        _save_video(diffs,    os.path.join(save_dir, 'diff.mp4'),      fps, cams_list,
                    visualize_func=_diff_colorize)

        # ── Panoramic videos (left|front|right|back) ─────────────
        _save_panoramic(rgbs, rgbs_gt, depths, diffs, cams_list,
                        save_dir, fps)

        print("Done.")


# ═══════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GoPro Max Neighbour – Rendering / Evaluation")
    parser.add_argument(
        "--config", default="configs/gopromax_neighbour.yaml",
        help="Path to YAML config file.")
    parser.add_argument(
        "--mode", choices=["evaluate", "trajectory"], default="evaluate",
        help="Rendering mode.")
    parser.add_argument(
        "--epoch", type=int, default=None,
        help="Checkpoint epoch to load (default: latest).")
    parser.add_argument(
        "--skip_train", action="store_true",
        help="Skip rendering train set (evaluate mode only).")
    parser.add_argument(
        "--skip_test", action="store_true",
        help="Skip rendering test set (evaluate mode only).")
    parser.add_argument(
        "--fps", type=int, default=10,
        help="Video FPS (trajectory mode only).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_path = os.path.join("output", cfg["task"], cfg["exp_name"])
    print(f"Rendering  {model_path}")

    if args.mode == "evaluate":
        render_sets(cfg, epoch=args.epoch,
                    skip_train=args.skip_train,
                    skip_test=args.skip_test)
    elif args.mode == "trajectory":
        render_trajectory(cfg, epoch=args.epoch, fps=args.fps)
    else:
        raise ValueError(
            f"Unknown mode '{args.mode}'. Use --mode evaluate or --mode trajectory")
