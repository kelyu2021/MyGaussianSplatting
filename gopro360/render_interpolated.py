"""
Render an image from an interpolated viewpoint between two existing frames.

Usage (run from gopro360/ directory):
    # Render from exactly halfway between frame 5 and frame 6 (front face)
    python render_interpolated.py \
        --cfg_file configs/gopro360_mask.yaml \
        --frame_a 5 --frame_b 6 --alpha 0.5 --face front \
        --output interpolated.png

    # Render from 30% toward frame 10 from frame 9 (all 4 faces → panoramic)
    python render_interpolated.py \
        --cfg_file configs/gopro360_mask.yaml \
        --frame_a 9 --frame_b 10 --alpha 0.3 --face all \
        --output interpolated_pano.png

    # List available frames
    python render_interpolated.py \
        --cfg_file configs/gopro360_mask.yaml --list_frames

Arguments:
    --frame_a    First frame index  (e.g. 0, 1, 2, …)
    --frame_b    Second frame index
    --alpha      Interpolation factor: 0.0 = frame_a, 1.0 = frame_b (default 0.5)
    --face       Which cubemap face to render: front, right, back, left, or all (default: front)
    --output     Output image path (default: interpolated.png)
    --depth      Also save a depth map (adds _depth suffix)
    --list_frames  Print all available frame indices and exit
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import torchvision
from scipy.spatial.transform import Rotation, Slerp

# ═══════════════════════════════════════════════════════════════════════════
#  We need to parse our custom args BEFORE importing lib.config (which
#  calls argparse.parse_args() at import time).  So we split sys.argv.
# ═══════════════════════════════════════════════════════════════════════════

CUSTOM_ARGS = [
    '--frame_a', '--frame_b', '--alpha',
    '--face', '--output', '--depth', '--list_frames', '--comparison',
]

_custom_argv = []
_lib_argv = [sys.argv[0]]
skip_next = False
for i, arg in enumerate(sys.argv[1:], 1):
    if skip_next:
        skip_next = False
        continue
    if arg in CUSTOM_ARGS:
        _custom_argv.append(arg)
        if arg not in ('--depth', '--list_frames', '--comparison') and i < len(sys.argv) - 1:
            _custom_argv.append(sys.argv[i + 1])
            skip_next = True
    else:
        _lib_argv.append(arg)

# Temporarily replace sys.argv so lib.config sees only its own args
_original_argv = sys.argv
sys.argv = _lib_argv

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from lib.config import cfg  # noqa: E402
from lib.datasets.gopro360_dataset import GoPro360Dataset  # noqa: E402
from lib.models.scene import Scene  # noqa: E402
from lib.models.street_gaussian_model import StreetGaussianModel  # noqa: E402
from lib.models.street_gaussian_renderer import StreetGaussianRenderer  # noqa: E402
from lib.utils.camera_utils import Camera  # noqa: E402
from lib.utils.img_utils import visualize_depth_numpy  # noqa: E402

# Restore sys.argv and parse our custom arguments
sys.argv = _original_argv

custom_parser = argparse.ArgumentParser(description="Render interpolated viewpoint")
custom_parser.add_argument('--frame_a', type=int, default=0)
custom_parser.add_argument('--frame_b', type=int, default=1)
custom_parser.add_argument('--alpha', type=float, default=0.5)
custom_parser.add_argument('--face', type=str, default='front',
                           choices=['front', 'right', 'back', 'left', 'all'])
custom_parser.add_argument('--output', type=str, default='interpolated.png')
custom_parser.add_argument('--depth', action='store_true')
custom_parser.add_argument('--list_frames', action='store_true')
custom_parser.add_argument('--comparison', action='store_true',
                           help='Generate 5-row comparison grid: GT_a, GS_a, GT_b, GS_b, interpolated')
custom_args = custom_parser.parse_known_args(_custom_argv)[0]

FACE_MAP = {'front': 0, 'right': 1, 'back': 2, 'left': 3}


def _depth_colorize(depth_hw1: np.ndarray) -> np.ndarray:
    return visualize_depth_numpy(depth_hw1, cmap=cv2.COLORMAP_JET)[0][..., [2, 1, 0]]


def _get_frame_cameras(all_cameras: list, frame_idx: int) -> dict[int, "Camera"]:
    """Return {face_id: Camera} for a given frame index."""
    result = {}
    for cam in all_cameras:
        meta = cam.meta if hasattr(cam, 'meta') else {}
        fidx = meta.get('frame_idx', None)
        face = meta.get('cam', None)
        if fidx == frame_idx:
            result[face] = cam
    return result


def _interpolate_c2w(c2w_a: np.ndarray, c2w_b: np.ndarray,
                     alpha: float) -> np.ndarray:
    """Smoothly interpolate between two 4×4 camera-to-world matrices.

    Translation: linear interpolation.
    Rotation: SLERP (spherical linear interpolation).
    """
    # Translation
    t_a = c2w_a[:3, 3]
    t_b = c2w_b[:3, 3]
    t_interp = (1 - alpha) * t_a + alpha * t_b

    # Rotation via SLERP
    R_a = Rotation.from_matrix(c2w_a[:3, :3])
    R_b = Rotation.from_matrix(c2w_b[:3, :3])
    slerp = Slerp([0, 1], Rotation.concatenate([R_a, R_b]))
    R_interp = slerp(alpha).as_matrix()

    c2w_interp = np.eye(4)
    c2w_interp[:3, :3] = R_interp
    c2w_interp[:3, 3] = t_interp
    return c2w_interp


def _make_interpolated_camera(cam_a: Camera, cam_b: Camera,
                              alpha: float) -> Camera:
    """Create a new Camera with pose interpolated between cam_a and cam_b.

    Intrinsics are taken from cam_a (assumed identical across frames).
    """
    c2w_a = cam_a.get_extrinsic()
    c2w_b = cam_b.get_extrinsic()
    c2w_interp = _interpolate_c2w(c2w_a, c2w_b, alpha)

    # Build a camera by copying cam_a and overriding the pose
    import copy
    new_cam = copy.deepcopy(cam_a)
    new_cam.set_extrinsic(c2w_interp)
    return new_cam


def list_frames(all_cameras):
    """Print all unique frame indices."""
    frames = set()
    for cam in all_cameras:
        meta = cam.meta if hasattr(cam, 'meta') else {}
        fidx = meta.get('frame_idx', None)
        if fidx is not None:
            frames.add(fidx)
    frames = sorted(frames)
    print(f"Available frames ({len(frames)} total):")
    print(f"  {frames}")
    return frames


def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """[C, H, W] float tensor → [H, W, C] uint8 numpy."""
    return (t.detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)


def _add_label(img: np.ndarray, text: str) -> np.ndarray:
    """Draw a label in the top-left corner of an image."""
    img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.7, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (0, 0), (tw + 10, th + 14), (0, 0, 0), -1)
    cv2.putText(img, text, (5, th + 8), font, scale, (255, 255, 255), thickness)
    return img


def render_interpolated():
    cfg.mode = 'evaluate'

    with torch.no_grad():
        dataset = GoPro360Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()

        all_cameras = scene.getTrainCameras() + scene.getTestCameras()
        all_cameras = sorted(all_cameras, key=lambda c: c.id)

        if custom_args.list_frames:
            list_frames(all_cameras)
            return

        frame_a = custom_args.frame_a
        frame_b = custom_args.frame_b
        alpha = custom_args.alpha

        cams_a = _get_frame_cameras(all_cameras, frame_a)
        cams_b = _get_frame_cameras(all_cameras, frame_b)

        if not cams_a:
            avail = list_frames(all_cameras)
            print(f"\nERROR: frame_a={frame_a} not found.")
            return
        if not cams_b:
            avail = list_frames(all_cameras)
            print(f"\nERROR: frame_b={frame_b} not found.")
            return

        faces_to_render = (
            list(FACE_MAP.values()) if custom_args.face == 'all'
            else [FACE_MAP[custom_args.face]]
        )

        rendered_images = {}
        rendered_depths = {}
        # For comparison mode: also collect GT and GS renders for frames a & b
        gt_a, gs_a, gt_b, gs_b = {}, {}, {}, {}

        for face_id in faces_to_render:
            face_name = [k for k, v in FACE_MAP.items() if v == face_id][0]
            if face_id not in cams_a:
                print(f"WARNING: face '{face_name}' not found in frame {frame_a}, skipping.")
                continue
            if face_id not in cams_b:
                print(f"WARNING: face '{face_name}' not found in frame {frame_b}, skipping.")
                continue

            # Render from original frame a
            if custom_args.comparison:
                gt_a[face_id] = _tensor_to_uint8(cams_a[face_id].original_image[:3])
                res_a = renderer.render(cams_a[face_id], gaussians)
                gs_a[face_id] = _tensor_to_uint8(res_a['rgb'])
                print(f"  Rendered frame {frame_a} face '{face_name}' (original + GS)")

                gt_b[face_id] = _tensor_to_uint8(cams_b[face_id].original_image[:3])
                res_b = renderer.render(cams_b[face_id], gaussians)
                gs_b[face_id] = _tensor_to_uint8(res_b['rgb'])
                print(f"  Rendered frame {frame_b} face '{face_name}' (original + GS)")

            # Render interpolated
            cam = _make_interpolated_camera(cams_a[face_id], cams_b[face_id], alpha)
            result = renderer.render(cam, gaussians)

            rgb = result['rgb'].detach().cpu().clamp(0, 1)
            rendered_images[face_id] = rgb

            if custom_args.depth:
                depth = result['depth'].detach().permute(1, 2, 0).cpu().numpy()
                rendered_depths[face_id] = depth

            print(f"  Rendered face '{face_name}' "
                  f"(interpolated: frame {frame_a} →({alpha:.2f})→ frame {frame_b})")

        # Save output
        out_path = custom_args.output
        base, ext = os.path.splitext(out_path)
        if not ext:
            ext = '.png'
            out_path = base + ext

        # ── Comparison grid mode ──────────────────────────────────
        if custom_args.comparison and len(rendered_images) >= 1:
            face_order = [fid for fid in [0, 1, 2, 3] if fid in rendered_images] \
                         if custom_args.face == 'all' else faces_to_render
            face_names = {0: 'front', 1: 'right', 2: 'back', 3: 'left'}

            rows = []
            row_labels = [
                f'GT frame {frame_a}',
                f'GS frame {frame_a}',
                f'Interpolated (α={alpha})',
                f'GS frame {frame_b}',
                f'GT frame {frame_b}',
            ]
            row_dicts = [gt_a, gs_a, None, gs_b, gt_b]  # None = interpolated

            for label, src in zip(row_labels, row_dicts):
                panels = []
                for fid in face_order:
                    if src is not None:
                        img = src[fid]
                    else:
                        img = _tensor_to_uint8(rendered_images[fid])
                    img = _add_label(img, f"{label} / {face_names[fid]}")
                    panels.append(img)
                rows.append(np.concatenate(panels, axis=1))

            grid = np.concatenate(rows, axis=0)
            imageio.imwrite(out_path, grid)
            print(f"\nSaved comparison grid ({len(rows)}×{len(face_order)}) → {out_path}")
            return

        if custom_args.face == 'all' and len(rendered_images) == 4:
            # Stitch panoramic: left | front | right | back
            pano_order = [3, 0, 1, 2]  # left, front, right, back
            panels = [
                (rendered_images[fid].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                for fid in pano_order
            ]
            pano = np.concatenate(panels, axis=1)
            imageio.imwrite(out_path, pano)
            print(f"\nSaved panoramic image → {out_path}")

            if custom_args.depth and len(rendered_depths) == 4:
                depth_panels = [_depth_colorize(rendered_depths[fid]) for fid in pano_order]
                pano_depth = np.concatenate(depth_panels, axis=1)
                depth_path = f"{base}_depth{ext}"
                imageio.imwrite(depth_path, pano_depth)
                print(f"Saved panoramic depth → {depth_path}")
        else:
            face_id = faces_to_render[0]
            if face_id in rendered_images:
                torchvision.utils.save_image(rendered_images[face_id], out_path)
                print(f"\nSaved image → {out_path}")

                if custom_args.depth and face_id in rendered_depths:
                    depth_path = f"{base}_depth{ext}"
                    imageio.imwrite(depth_path, _depth_colorize(rendered_depths[face_id]))
                    print(f"Saved depth → {depth_path}")


if __name__ == '__main__':
    render_interpolated()
