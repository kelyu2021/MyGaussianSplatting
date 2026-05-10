"""
SDS Score vs. Jitter Distance Plot
===================================

For a given camera and trained model, sweeps a range of lateral jitter
distances (meters), renders the corresponding viewpoint, computes the SDS
realism score for each, and saves an x/y plot.

Usage
-----
    python plot_sds_vs_jitter.py \\
        --img_name       0001_front \\
        --model_path     output/.../epoch_300.pth \\
        --cameras_json   output/.../cameras.json \\
        --output_dir     output/.../sds_plot \\
        --min_dist       0.0 \\
        --max_dist       3.0 \\
        --num_dists      13 \\
        --prompt         "A street level image of an outdoor scene"
"""

import argparse
import os
import json

import csv

import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
from PIL import Image

from train_da2loss import (
    GaussianModel,
    load_camera,
    render,
    CameraInfo,
    SphericalHarmonicSky,
    _compute_sky_bg,
)
from sds_score import load_sd_components, compute_sds_score
from render_wobble import (
    load_camerainfo_from_json,
    make_caminfo_jittered,
)


def render_at_offset(gaussians, bg_color, caminfo_base, offset_m, sky_model=None):
    """Render a single frame laterally shifted by offset_m (metres, + = right)."""
    pos_base = -caminfo_base.R @ caminfo_base.T
    right = caminfo_base.R[:, 0]
    pos_jit = pos_base + right * offset_m
    caminfo_jit = make_caminfo_jittered(caminfo_base, pos_jit, caminfo_base.R)
    cam = load_camera(caminfo_jit)
    with torch.no_grad():
        result = render(cam, gaussians, bg_color)
        rgb = result['rgb']  # (3, H, W)
        if sky_model is not None:
            acc = result['acc']
            sky_bg = _compute_sky_bg(cam, sky_model)
            rgb = rgb + (1.0 - acc) * sky_bg
    rgb_np = rgb.detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
    return (rgb_np * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Plot SDS score vs. jitter distance")
    parser.add_argument('--img_name', required=True)
    parser.add_argument('--model_path', required=True, help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--cameras_json', required=True)
    parser.add_argument('--output_dir', default='sds_plot')
    parser.add_argument('--min_dist', type=float, default=0.0,
                        help='Minimum lateral offset in world units (default: 0)')
    parser.add_argument('--max_dist', type=float, default=3.0,
                        help='Maximum lateral offset in world units (default: 3)')
    parser.add_argument('--num_dists', type=int, default=13,
                        help='Number of distances to sweep (default: 13)')
    parser.add_argument('--side', choices=['right', 'left', 'both'], default='right',
                        help='Which direction to jitter: right (+), left (-), or both (average). Default: right')
    parser.add_argument('--prompt', default='A street level image of an outdoor scene',
                        help='Text prompt for SDS scoring')
    parser.add_argument('--model_id', '--sd_model_id', dest='sd_model_id',
                        default='runwayml/stable-diffusion-v1-5',
                        help='HuggingFace Stable Diffusion model ID')
    parser.add_argument('--num_samples', type=int, default=32,
                        help='Timestep samples for SDS averaging (default: 32)')
    parser.add_argument('--sky_sh_degree', type=int, default=3)
    parser.add_argument('--fps', type=int, default=10,
                        help='FPS for output video(s) (default: 10)')
    parser.add_argument('--save_renders', action='store_true',
                        help='Also save each rendered frame as a PNG')
    parser.add_argument('--meters_per_unit', type=float, default=7.0,
                        help='World units → meters scale for x-axis display. '
                             'Estimate 7 m.')
    parser.add_argument('--num_repeats', type=int, default=1,
                        help='Number of independent SDS evaluations per distance '
                             '(used to estimate mean/std). Default: 1.')
    parser.add_argument('--errorbar_style', choices=['band', 'errorbar', 'both'],
                        default='band',
                        help='How to display variance: shaded band (±1 std), '
                             'errorbars, or both. Default: band.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load Gaussian model ------------------------------------------------
    state = torch.load(args.model_path, map_location='cuda', weights_only=False)
    gaussians = GaussianModel(sh_degree=state.get('active_sh_degree', 3))
    gaussians.load_state_dict(state)
    gaussians.active_sh_degree = gaussians.max_sh_degree

    sky_model = None
    if 'sky_model_state' in state:
        print('Sky model found in checkpoint – applying SH sky compositing.')
        sky_model = SphericalHarmonicSky(sh_degree=args.sky_sh_degree).cuda()
        sky_model.load_state_dict(state['sky_model_state'])
        sky_model.eval()
    else:
        print('No sky model in checkpoint – rendering without sky compositing.')

    bg_color = torch.zeros(3, dtype=torch.float32, device='cuda')

    # ---- Load camera --------------------------------------------------------
    caminfo_base = load_camerainfo_from_json(args.cameras_json, args.img_name)

    # ---- Load SD components once (reuse across all SDS calls) ---------------
    sd_components = load_sd_components(args.sd_model_id, device='cuda', dtype=torch.float16)

    # ---- Sweep distances ----------------------------------------------------
    distances = np.linspace(args.min_dist, args.max_dist, args.num_dists)
    # all_scores[i] holds a list of independent SDS evaluations at distances[i]
    all_scores = [[] for _ in range(len(distances))]
    video_frames_right = []
    video_frames_left  = []

    def _score_for_side(frame_r, frame_l):
        if args.side == 'right':
            return compute_sds_score(Image.fromarray(frame_r), args.prompt,
                                     num_samples=args.num_samples,
                                     sd_components=sd_components)
        elif args.side == 'left':
            return compute_sds_score(Image.fromarray(frame_l), args.prompt,
                                     num_samples=args.num_samples,
                                     sd_components=sd_components)
        else:  # both – average right and left
            sr = compute_sds_score(Image.fromarray(frame_r), args.prompt,
                                   num_samples=args.num_samples,
                                   sd_components=sd_components)
            sl = compute_sds_score(Image.fromarray(frame_l), args.prompt,
                                   num_samples=args.num_samples,
                                   sd_components=sd_components)
            return (sr + sl) / 2.0

    for idx, dist in enumerate(distances):
        print(f"[{idx+1}/{len(distances)}] distance = {dist:.4f}")

        # Render once per position (geometry is deterministic). SDS is the
        # stochastic part – we re-evaluate it `num_repeats` times for stats.
        frame_r = render_at_offset(gaussians, bg_color, caminfo_base,  dist, sky_model)
        frame_l = render_at_offset(gaussians, bg_color, caminfo_base, -dist, sky_model)
        video_frames_right.append(frame_r)
        video_frames_left.append(frame_l)

        if args.save_renders:
            Image.fromarray(frame_r).save(
                os.path.join(args.output_dir, f'render_{idx+1:04d}_right_{dist:.4f}.png'))
            Image.fromarray(frame_l).save(
                os.path.join(args.output_dir, f'render_{idx+1:04d}_left_{dist:.4f}.png'))

        for rep in range(args.num_repeats):
            score = _score_for_side(frame_r, frame_l)
            all_scores[idx].append(score)
            print(f"    rep {rep+1}/{args.num_repeats} SDS = {score:.6f}")

    scores_arr = np.array(all_scores, dtype=np.float64)  # (num_dists, num_repeats)
    sds_mean = scores_arr.mean(axis=1)
    sds_std  = scores_arr.std(axis=1, ddof=1) if args.num_repeats > 1 else np.zeros_like(sds_mean)

    # ---- Save raw data (npz + csv) ------------------------------------------
    base_name = os.path.splitext(os.path.basename(args.img_name))[0]
    data_path = os.path.join(args.output_dir, f'{base_name}_sds_vs_jitter.npz')
    np.savez(data_path,
             distances=distances,
             sds_scores_all=scores_arr,
             sds_mean=sds_mean,
             sds_std=sds_std)
    print(f"Raw data saved to {data_path}")

    csv_path = os.path.join(args.output_dir, f'{base_name}_sds_vs_jitter.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['jitter_distance', 'sds_mean', 'sds_std'] + \
                 [f'sds_rep_{i+1}' for i in range(args.num_repeats)]
        writer.writerow(header)
        for i, dist in enumerate(distances):
            row = [f'{dist:.6f}', f'{sds_mean[i]:.8f}', f'{sds_std[i]:.8f}'] + \
                  [f'{s:.8f}' for s in scores_arr[i]]
            writer.writerow(row)
    print(f"CSV saved to {csv_path}")

    # ---- Save video(s) -----------------------------------------------------
    # Separate directional videos
    if video_frames_right:
        vid_path = os.path.join(args.output_dir, f'{base_name}_jitter_right.mp4')
        imageio.mimwrite(vid_path, video_frames_right, fps=args.fps)
        print(f"Right video saved to {vid_path}")
    if video_frames_left:
        vid_path = os.path.join(args.output_dir, f'{base_name}_jitter_left.mp4')
        imageio.mimwrite(vid_path, video_frames_left, fps=args.fps)
        print(f"Left video saved to {vid_path}")

    # Combined video: center→right→center→left→center
    if video_frames_right and video_frames_left:
        combined = (
            video_frames_right +           # center → right
            video_frames_right[::-1] +     # right  → center
            video_frames_left +            # center → left
            video_frames_left[::-1]        # left   → center
        )
        combined_path = os.path.join(args.output_dir, f'{base_name}_jitter_combined.mp4')
        imageio.mimwrite(combined_path, combined, fps=args.fps)
        print(f"Combined video saved to {combined_path}")

    # ---- Plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    distances_m = distances * args.meters_per_unit

    # Mean line
    ax.plot(distances_m, sds_mean, marker='o', linewidth=1.5, markersize=5,
            color='C0', label=f'mean (n={args.num_repeats})')

    # Variance display
    if args.num_repeats > 1:
        if args.errorbar_style in ('band', 'both'):
            ax.fill_between(distances_m,
                            sds_mean - sds_std,
                            sds_mean + sds_std,
                            alpha=0.25, color='C0', label='±1 std')
        if args.errorbar_style in ('errorbar', 'both'):
            ax.errorbar(distances_m, sds_mean, yerr=sds_std,
                        fmt='none', ecolor='C0', capsize=3, alpha=0.8)
        # Scatter raw samples for transparency
        for i, d in enumerate(distances_m):
            ax.scatter(np.full(args.num_repeats, d), scores_arr[i],
                       s=8, color='C0', alpha=0.3, zorder=1)
        ax.legend(loc='best', fontsize=8)

    ax.set_xlabel('Jitter distance (m)')
    ax.set_ylabel('SDS score (lower = more realistic)')
    ax.set_title(f'SDS score vs. lateral jitter — {base_name}')
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()

    plot_path = os.path.join(args.output_dir, f'{base_name}_sds_vs_jitter.png')
    fig.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.close(fig)


if __name__ == '__main__':
    main()
