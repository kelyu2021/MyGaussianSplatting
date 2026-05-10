"""
SDS Score vs. Jitter Distance Plot  (3DGS version)
===================================================

For a given camera and trained 3DGS model, sweeps a range of lateral jitter
distances, renders the corresponding viewpoint, computes the SDS realism
score for each, and saves an x/y plot + raw data.

Mirrors gopromax_neighbour/plot_sds_vs_jitter.py but uses the gaussian-splatting
rendering stack (GaussianModel + gaussian_renderer.render) and reads the
cameras.json / point_cloud.ply produced by train_neighbour.py.

Usage
-----
    python plot_sds_vs_jitter.py \\
        --img_name        0001_front \\
        --model_dir       output/run_01 \\
        --output_dir      output/run_01/sds_plot \\
        --min_dist        0.0 \\
        --max_dist        4.0 \\
        --num_dists       25 \\
        --side            right \\
        --prompt          "A street level image of an outdoor scene" \\
        --num_repeats     4 \\
        --num_samples     32
"""

import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import torch
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# ---- allow imports from the gaussian-splatting package ----------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from gaussian_renderer import render as gs_render
from scene import Scene
from scene.cameras import Camera
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except ImportError:
    SPARSE_ADAM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def load_camera_from_json(cameras_json: str, img_name: str):
    """Return the dict entry matching img_name from cameras.json.

    Matches against both the exact value and the stem (no extension), so
    passing '0017_front' will match an entry stored as '0017_front.jpg'.
    """
    with open(cameras_json, 'r') as f:
        cameras = json.load(f)
    img_stem = os.path.splitext(img_name)[0]
    for cam in cameras:
        stored = cam['img_name']
        if stored == img_name or os.path.splitext(stored)[0] == img_stem:
            return cam
    raise ValueError(f"Camera '{img_name}' not found in {cameras_json}")


def make_camera(cam_dict, R: np.ndarray, T: np.ndarray, uid: int = 0) -> Camera:
    """
    Build a gaussian-splatting Camera from a cameras.json entry and an
    explicit (R, T) pair.

    cameras.json convention (written by camera_to_JSON):
        rotation  : R  (3×3, camera-to-world rotation matrix, i.e. R_c2w)
        position  : camera centre in world coordinates
        fx, fy    : focal lengths in pixels
        width, height
    """
    width  = int(cam_dict['width'])
    height = int(cam_dict['height'])
    fx     = float(cam_dict['fx'])
    fy     = float(cam_dict['fy'])

    FoVx = 2.0 * math.atan(width  / (2.0 * fx))
    FoVy = 2.0 * math.atan(height / (2.0 * fy))

    # Dummy blank image (we only need geometry / projection, not GT colour)
    blank = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))

    cam = Camera(
        resolution=(width, height),
        colmap_id=uid,
        R=R,
        T=T,
        FoVx=FoVx,
        FoVy=FoVy,
        depth_params=None,
        image=blank,
        invdepthmap=None,
        image_name=cam_dict['img_name'],
        uid=uid,
    )
    return cam


def jitter_camera(cam_dict, offset_m: float) -> Camera:
    """
    Return a Camera laterally shifted by offset_m world units along the
    camera's right axis (+offset → right, -offset → left).

    cameras.json stores:
        rotation  : R_c2w  (3×3, camera-to-world rotation matrix,
                            written by camera_to_JSON as W2C[:3,:3]
                            where W2C = inv(extrinsic))
        position  : camera centre in world coordinates

    Relationship:  T (tvec) = R_w2c @ (-C)  = -R_c2w.T @ C
    The right axis in world coords is the first *column* of R_c2w,
    i.e.  right_world = R_c2w[:, 0].
    """
    R_c2w = np.array(cam_dict['rotation'], dtype=np.float64)   # (3,3)
    C     = np.array(cam_dict['position'], dtype=np.float64)   # (3,)

    right_world = R_c2w[:, 0]          # unit right vector in world coords (first column of R_c2w)
    C_new       = C + right_world * offset_m
    T_new       = -(R_c2w.T) @ C_new  # tvec for new position: R_w2c @ (-C_new)

    return make_camera(cam_dict, R=R_c2w.astype(np.float32),
                       T=T_new.astype(np.float32))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_frame(gaussians, pipeline, background, cam: Camera) -> np.ndarray:
    """Render a Camera and return an uint8 HxWx3 numpy array."""
    with torch.no_grad():
        pkg = gs_render(cam, gaussians, pipeline, background,
                        use_trained_exp=False,
                        separate_sh=SPARSE_ADAM_AVAILABLE)
    rgb = pkg['render'].clamp(0.0, 1.0)          # (3, H, W)
    rgb_np = (rgb.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return rgb_np


# ---------------------------------------------------------------------------
# SDS scoring (reused from gopromax_neighbour)
# ---------------------------------------------------------------------------

def _import_sds():
    """Import sds_score from gopromax_neighbour (sibling directory)."""
    sds_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'gopromax_neighbour')
    sys.path.insert(0, os.path.abspath(sds_dir))
    from sds_score import load_sd_components, compute_sds_score
    return load_sd_components, compute_sds_score


# ---------------------------------------------------------------------------
# Plotting helpers (same style as gopromax_neighbour)
# ---------------------------------------------------------------------------

def save_plot(output_dir, base_name, distances, sds_mean, sds_std,
              num_repeats, meters_per_unit, errorbar_style):
    x = distances * meters_per_unit
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, sds_mean, marker='o', color='steelblue', linewidth=2,
            label='SDS mean')
    if num_repeats > 1:
        if errorbar_style in ('band', 'both'):
            ax.fill_between(x, sds_mean - sds_std, sds_mean + sds_std,
                            alpha=0.25, color='steelblue', label='±1 std')
        if errorbar_style in ('errorbar', 'both'):
            ax.errorbar(x, sds_mean, yerr=sds_std, fmt='none',
                        color='steelblue', capsize=4)
    ax.set_xlabel('Lateral jitter (m)')
    ax.set_ylabel('SDS score  (lower = more realistic)')
    ax.set_title(f'SDS vs. lateral jitter — {base_name}')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    plot_path = os.path.join(output_dir, f'{base_name}_sds_vs_jitter.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='SDS score vs. lateral jitter — 3DGS version')
    parser.add_argument('--img_name', required=True,
                        help='img_name field in cameras.json, e.g. 0001_front')
    parser.add_argument('--model_dir', default='output/run_01',
                        help='Path to trained model directory (contains cfg_args)')
    parser.add_argument('--iteration', type=int, default=-1,
                        help='Which saved iteration to load (-1 = latest)')
    parser.add_argument('--output_dir', default='output/run_01/sds_plot')
    parser.add_argument('--min_dist', type=float, default=0.0)
    parser.add_argument('--max_dist', type=float, default=4.0)
    parser.add_argument('--num_dists', type=int, default=13)
    parser.add_argument('--side', choices=['right', 'left', 'both'],
                        default='right')
    parser.add_argument('--prompt',
                        default='A street level image of an outdoor scene')
    parser.add_argument('--model_id', default='runwayml/stable-diffusion-v1-5',
                        help='HuggingFace SD model ID for SDS scoring')
    parser.add_argument('--num_samples', type=int, default=32,
                        help='Timestep samples for SDS averaging')
    parser.add_argument('--num_repeats', type=int, default=1,
                        help='Independent SDS evaluations per distance')
    parser.add_argument('--fps', type=int, default=10,
                        help='FPS for output video(s)')
    parser.add_argument('--save_renders', action='store_true',
                        help='Save each frame as a PNG')
    parser.add_argument('--meters_per_unit', type=float, default=2,
                        help='World-unit → metres scale for x-axis (default 1)')
    parser.add_argument('--errorbar_style',
                        choices=['band', 'errorbar', 'both'], default='band')
    parser.add_argument('--skip_sds', action='store_true',
                        help='Skip SDS scoring; only render and save frames/video')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load trained Gaussians -------------------------------------------
    print(f"Loading model from {args.model_dir} ...")
    from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: F811
    from argparse import ArgumentParser as AP
    sub = AP()
    mp = ModelParams(sub, sentinel=True)
    pp = PipelineParams(sub)
    # Temporarily override sys.argv so get_combined_args reads cfg_args from
    # model_dir (which contains source_path and all other training settings).
    _orig_argv = sys.argv[:]
    sys.argv = [sys.argv[0], '--model_path', args.model_dir]
    sub_args = get_combined_args(sub)
    sys.argv = _orig_argv
    dataset  = mp.extract(sub_args)
    pipeline = pp.extract(sub_args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration,
                  shuffle=False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')

    # ---- Load base camera from cameras.json --------------------------------
    cameras_json = os.path.join(args.model_dir, 'cameras.json')
    cam_dict = load_camera_from_json(cameras_json, args.img_name)

    # ---- Optionally load SD components ------------------------------------
    if not args.skip_sds:
        load_sd_components, compute_sds_score = _import_sds()
        sd_components = load_sd_components(args.model_id, device='cuda',
                                           dtype=torch.float16)

    # ---- Sweep distances ---------------------------------------------------
    distances = np.linspace(args.min_dist, args.max_dist, args.num_dists)
    all_scores   = [[] for _ in range(len(distances))]
    frames_right = []
    frames_left  = []

    # Per-image subfolder for frames
    base_name = args.img_name
    frames_dir = os.path.join(args.output_dir, base_name)
    os.makedirs(frames_dir, exist_ok=True)

    for idx, dist in enumerate(distances):
        print(f"[{idx+1}/{len(distances)}] offset = {dist:.4f} world units")

        cam_r = jitter_camera(cam_dict,  dist)
        cam_l = jitter_camera(cam_dict, -dist)
        frame_r = render_frame(gaussians, pipeline, background, cam_r)
        frame_l = render_frame(gaussians, pipeline, background, cam_l)
        frames_right.append(frame_r)
        frames_left.append(frame_l)

        # Always save every rendered frame into the per-image subfolder
        Image.fromarray(frame_r).save(
            os.path.join(frames_dir,
                         f'render_{idx+1:04d}_right_{dist:.4f}.png'))
        Image.fromarray(frame_l).save(
            os.path.join(frames_dir,
                         f'render_{idx+1:04d}_left_{dist:.4f}.png'))

        if not args.skip_sds:
            for rep in range(args.num_repeats):
                if args.side == 'right':
                    img = Image.fromarray(frame_r)
                elif args.side == 'left':
                    img = Image.fromarray(frame_l)
                else:
                    sr = compute_sds_score(Image.fromarray(frame_r),
                                           args.prompt,
                                           num_samples=args.num_samples,
                                           sd_components=sd_components)
                    sl = compute_sds_score(Image.fromarray(frame_l),
                                           args.prompt,
                                           num_samples=args.num_samples,
                                           sd_components=sd_components)
                    score = (sr + sl) / 2.0
                    all_scores[idx].append(score)
                    print(f"    rep {rep+1}/{args.num_repeats} SDS = {score:.6f}")
                    continue

                score = compute_sds_score(img, args.prompt,
                                          num_samples=args.num_samples,
                                          sd_components=sd_components)
                all_scores[idx].append(score)
                print(f"    rep {rep+1}/{args.num_repeats} SDS = {score:.6f}")

    # ---- Save videos -------------------------------------------------------
    for tag, frames in [('right', frames_right), ('left', frames_left)]:
        vid_path = os.path.join(args.output_dir, f'{base_name}_{tag}.mp4')
        imageio.mimwrite(vid_path, frames, fps=args.fps, quality=8)
        print(f"Video saved: {vid_path}")

    # Combined: center → right → center → left → center (no duplicate frames at junctions)
    frames_combined = (
        frames_right +                        # center → right
        list(reversed(frames_right))[1:] +   # right  → center
        frames_left[1:] +                     # center → left
        list(reversed(frames_left))[1:]       # left   → center
    )
    combined_path = os.path.join(args.output_dir, f'{base_name}_combined.mp4')
    imageio.mimwrite(combined_path, frames_combined, fps=args.fps, quality=8)
    print(f"Combined video saved: {combined_path}")

    if args.skip_sds:
        print("SDS scoring skipped (--skip_sds). Done.")
        return

    # ---- Aggregate & save results ------------------------------------------
    scores_arr = np.array(all_scores, dtype=np.float64)   # (num_dists, num_repeats)
    sds_mean = scores_arr.mean(axis=1)
    sds_std  = (scores_arr.std(axis=1, ddof=1)
                if args.num_repeats > 1 else np.zeros_like(sds_mean))

    npz_path = os.path.join(args.output_dir, f'{base_name}_sds_vs_jitter.npz')
    np.savez(npz_path, distances=distances, sds_scores_all=scores_arr,
             sds_mean=sds_mean, sds_std=sds_std)
    print(f"Raw data saved: {npz_path}")

    csv_path = os.path.join(args.output_dir, f'{base_name}_sds_vs_jitter.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = (['jitter_distance', 'sds_mean', 'sds_std'] +
                  [f'sds_rep_{i+1}' for i in range(args.num_repeats)])
        writer.writerow(header)
        for i, dist in enumerate(distances):
            row = ([f'{dist:.6f}', f'{sds_mean[i]:.8f}', f'{sds_std[i]:.8f}'] +
                   [f'{s:.8f}' for s in scores_arr[i]])
            writer.writerow(row)
    print(f"CSV saved: {csv_path}")

    save_plot(args.output_dir, base_name, distances, sds_mean, sds_std,
              args.num_repeats, args.meters_per_unit, args.errorbar_style)

    print("Done.")


if __name__ == '__main__':
    main()
