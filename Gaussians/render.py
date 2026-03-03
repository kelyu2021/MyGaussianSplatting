"""
KITTI-360 Gaussian Splatting – Rendering / Evaluation Script
=============================================================

Renders trained Gaussian models and saves per-image outputs plus
optional trajectory videos.  Matches the output structure of the
original ``street_gaussians/render.py``.

Usage
-----
    # Evaluate (save per-image renders for train & test sets)
    python render.py --cfg_file configs/kitti360_drive_0000.yaml --mode evaluate

    # Trajectory video (all frames sorted by ID)
    python render.py --cfg_file configs/kitti360_drive_0000.yaml --mode trajectory

Outputs
-------
    evaluate mode:
        {model_path}/train/ours_{iter}/{name}_rgb.png
        {model_path}/train/ours_{iter}/{name}_gt.png
        {model_path}/train/ours_{iter}/{name}_depth.png
        {model_path}/train/ours_{iter}/{name}_diff.png
        (same under test/)

    trajectory mode:
        {model_path}/trajectory/ours_{iter}/color.mp4
        {model_path}/trajectory/ours_{iter}/color_gt.mp4
        {model_path}/trajectory/ours_{iter}/depth.mp4
        {model_path}/trajectory/ours_{iter}/diff.mp4
        {model_path}/trajectory/ours_{iter}/color_bkgd.mp4
"""

from __future__ import annotations

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
#  Path setup
# ═══════════════════════════════════════════════════════════════════════════
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from lib.config import cfg                                              # noqa: E402
from lib.datasets.dataset import Dataset                                # noqa: E402
from lib.models.scene import Scene                                      # noqa: E402
from lib.models.street_gaussian_model import StreetGaussianModel        # noqa: E402
from lib.models.street_gaussian_renderer import StreetGaussianRenderer  # noqa: E402
from lib.utils.camera_utils import Camera                               # noqa: E402
from lib.utils.general_utils import safe_state                          # noqa: E402
from lib.utils.img_utils import visualize_depth_numpy                   # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
#  Visualisation helpers
# ═══════════════════════════════════════════════════════════════════════════

def _depth_colorize(depth_hw1: np.ndarray) -> np.ndarray:
    """Colorize a [H, W, 1] depth array → [H, W, 3] uint8 (RGB)."""
    return visualize_depth_numpy(depth_hw1, cmap=cv2.COLORMAP_JET)[0][..., [2, 1, 0]]


def _diff_colorize(diff_hw1: np.ndarray) -> np.ndarray:
    """Colorize a [H, W, 1] error map → [H, W, 3] uint8 (RGB)."""
    return visualize_depth_numpy(diff_hw1, cmap=cv2.COLORMAP_TURBO)[0][..., [2, 1, 0]]


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
    concat_cameras = cfg.render.get('concat_cameras', [])

    if len(concat_cameras) == len(unique_cams):
        # Concatenate cameras side-by-side
        frames_per_cam = {
            cam: [f for f, c in zip(frames, cams) if c == cam]
            for cam in concat_cameras
        }
        n = len(next(iter(frames_per_cam.values())))
        concat_frames = []
        for i in range(n):
            row = [frames_per_cam[cam][i] for cam in concat_cameras]
            concat_frames.append(np.concatenate(row, axis=1))
        if visualize_func is not None:
            concat_frames = [visualize_func(f) for f in concat_frames]
        imageio.mimwrite(path, concat_frames, fps=fps)
    else:
        # Separate video per camera
        base, ext = os.path.splitext(path)
        for cam in unique_cams:
            cam_frames = [f for f, c in zip(frames, cams) if c == cam]
            if visualize_func is not None:
                cam_frames = [visualize_func(f) for f in cam_frames]
            imageio.mimwrite(f"{base}_{cam}{ext}", cam_frames, fps=fps)


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluate mode – per-image rendering for train / test sets
# ═══════════════════════════════════════════════════════════════════════════

def render_sets():
    """Render and save per-image outputs for train and test camera sets."""
    cfg.render.save_image = True
    cfg.render.save_video = False

    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()

        times: list[float] = []

        splits = []
        if not cfg.eval.skip_train:
            splits.append(("train", scene.getTrainCameras()))
        if not cfg.eval.skip_test:
            splits.append(("test", scene.getTestCameras()))

        for split_name, cameras in splits:
            save_dir = os.path.join(
                cfg.model_path, split_name,
                f"ours_{scene.loaded_iter}",
            )
            os.makedirs(save_dir, exist_ok=True)

            for idx, camera in enumerate(tqdm(cameras,
                                              desc=f"Rendering {split_name}")):
                torch.cuda.synchronize()
                t0 = time.time()

                result = renderer.render(camera, gaussians)

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
                mask = (camera.original_mask.bool()
                        if hasattr(camera, 'original_mask')
                        else None)
                diff = _compute_diff(result['rgb'], camera.original_image[:3], mask)
                imageio.imwrite(
                    os.path.join(save_dir, f'{name}_diff.png'),
                    _diff_colorize(diff),
                )

        if times:
            print(f"\nRendering times (ms): {times}")
            print(f"Average rendering time: {sum(times[1:]) / max(len(times) - 1, 1):.2f} ms")


# ═══════════════════════════════════════════════════════════════════════════
#  Trajectory mode – full video fly-through
# ═══════════════════════════════════════════════════════════════════════════

def render_trajectory():
    """Render all frames in order and produce trajectory videos."""
    cfg.render.save_image = False
    cfg.render.save_video = True

    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()

        save_dir = os.path.join(
            cfg.model_path, 'trajectory',
            f"ours_{scene.loaded_iter}",
        )
        os.makedirs(save_dir, exist_ok=True)

        # Merge train + test, sort by camera id
        cameras = scene.getTrainCameras() + scene.getTestCameras()
        cameras = sorted(cameras, key=lambda c: c.id)

        rgbs_gt, rgbs, rgbs_bkgd = [], [], []
        rgbs_obj, accs_obj = [], []
        depths, diffs = [], []
        cams_list = []
        fps = cfg.render.fps

        for idx, camera in enumerate(tqdm(cameras, desc="Rendering Trajectory")):
            result = renderer.render_all(camera, gaussians)

            cam_id = camera.meta.get('cam', 0)
            cams_list.append(cam_id)
            name = camera.image_name

            # Accumulate frames
            rgbs_gt.append(_tensor_to_uint8(camera.original_image[:3]))
            rgbs.append(_tensor_to_uint8(result['rgb']))
            rgbs_bkgd.append(_tensor_to_uint8(result['rgb_background']))
            rgbs_obj.append(_tensor_to_uint8(result['rgb_object']))
            accs_obj.append(_tensor_to_uint8(result['acc_object'].float()))

            depth = result['depth'].detach().permute(1, 2, 0).cpu().numpy()
            depths.append(depth)

            mask = (camera.original_mask.bool()
                    if hasattr(camera, 'original_mask')
                    else None)
            diff = _compute_diff(result['rgb'], camera.original_image[:3], mask)
            diffs.append(diff)

            # Optionally save per-frame images
            if cfg.render.save_image:
                torchvision.utils.save_image(
                    result['rgb'],
                    os.path.join(save_dir, f'{name}_rgb.png'),
                )
                torchvision.utils.save_image(
                    result['rgb_background'],
                    os.path.join(save_dir, f'{name}_rgb_bkgd.png'),
                )
                torchvision.utils.save_image(
                    result['rgb_object'],
                    os.path.join(save_dir, f'{name}_rgb_obj.png'),
                )
                torchvision.utils.save_image(
                    result['acc_object'].float(),
                    os.path.join(save_dir, f'{name}_acc_obj.png'),
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
        _save_video(rgbs_bkgd, os.path.join(save_dir, 'color_bkgd.mp4'), fps, cams_list)
        _save_video(rgbs_obj, os.path.join(save_dir, 'color_obj.mp4'), fps, cams_list)
        _save_video(accs_obj, os.path.join(save_dir, 'acc_obj.mp4'),   fps, cams_list)
        _save_video(depths,   os.path.join(save_dir, 'depth.mp4'),     fps, cams_list,
                    visualize_func=_depth_colorize)
        _save_video(diffs,    os.path.join(save_dir, 'diff.mp4'),      fps, cams_list,
                    visualize_func=_diff_colorize)
        print("Done.")


# ═══════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Rendering  {cfg.model_path}")
    safe_state(cfg.eval.quiet)

    if cfg.mode == 'evaluate':
        render_sets()
    elif cfg.mode == 'trajectory':
        render_trajectory()
    else:
        raise ValueError(
            f"Unknown mode '{cfg.mode}'. Use --mode evaluate or --mode trajectory"
        )
