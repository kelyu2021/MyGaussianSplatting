"""
Render and Difix-fix novel views from each FRONT and BACK camera
(both train and eval splits), translated perpendicular to the
driving trajectory by a fixed number of world units.

Perpendicular direction is computed per-viewpoint as the world-space
direction of the FRONT camera's local +X axis. That axis is sideways
to the car's heading by construction. The same world translation is
applied to the front and back cameras of that viewpoint so both shift
to the same side of the road.

Usage (from Difix3D/):
    PATH=/home/lyuk4/miniconda3/envs/difix3d/bin:/usr/bin:/bin \
    CUDA_HOME=/home/lyuk4/miniconda3/envs/difix3d \
    CC=/usr/bin/gcc CXX=/usr/bin/g++ \
    PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=1 \
    /home/lyuk4/miniconda3/envs/difix3d/bin/python examples/gsplat/render_perturbed.py \
        --data_dir data/scene_01 \
        --ckpt outputs/difix3d/gsplat/scene_01/ckpts/ckpt_59999_rank0.pt \
        --output_dir outputs/difix3d/gsplat/scene_01_perp \
        --distance 1.0
"""
import argparse
import os

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from gsplat.rendering import rasterization

from datasets.colmap import Dataset, Parser
from src.pipeline_difix import DifixPipeline


def viewpoint_id(name: str) -> str:
    """Extract viewpoint number from a COLMAP-recorded name like '0001_front.jpg'."""
    return name.split("_")[0]


def camera_direction(name: str) -> str:
    """Return 'front'/'back'/'left'/'right' from name."""
    base = name.split(".")[0]
    return base.split("_")[-1]


def build_perpendicular_table(parser):
    """For each viewpoint id, the world-space perpendicular-to-trajectory
    direction = front-camera's local +X expressed in world coords."""
    name_to_idx = {n: i for i, n in enumerate(parser.image_names)}
    perp = {}
    for name in parser.image_names:
        if camera_direction(name) != "front":
            continue
        idx = name_to_idx[name]
        c2w = parser.camtoworlds[idx]
        dir_world = c2w[:3, :3] @ np.array([1.0, 0.0, 0.0])
        dir_world = dir_world / np.linalg.norm(dir_world)
        perp[viewpoint_id(name)] = dir_world
    return perp


def find_nearest_train_path(parser, perturbed_pos, exclude_idx):
    """Return image path of the training camera with the closest position
    to `perturbed_pos`, excluding the index being perturbed."""
    best_idx, best_dist = None, np.inf
    for i, path in enumerate(parser.image_paths):
        # use only training photos (skip eval) for reference, and skip self
        if i == exclude_idx:
            continue
        if "_eval_" in os.path.basename(path):
            continue
        d = np.linalg.norm(parser.camtoworlds[i][:3, 3] - perturbed_pos)
        if d < best_dist:
            best_dist = d
            best_idx = i
    return parser.image_paths[best_idx]


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--distance", type=float, default=1.0)
    p.add_argument("--side", choices=["right", "left"], default=None,
                   help="Which side to perturb to. If omitted, render both sides.")
    p.add_argument("--data_factor", type=int, default=1)
    p.add_argument("--sh_degree", type=int, default=3)
    args = p.parse_args()

    sides = [args.side] if args.side is not None else ["right", "left"]
    side_signs = {"right": +1.0, "left": -1.0}

    side_dirs = {}
    for side in sides:
        side_root = os.path.join(args.output_dir, side)
        pred_dir = os.path.join(side_root, "Pred")
        fixed_dir = os.path.join(side_root, "Fixed")
        ref_dir = os.path.join(side_root, "Ref")
        for d in (pred_dir, fixed_dir, ref_dir):
            os.makedirs(d, exist_ok=True)
        side_dirs[side] = (pred_dir, fixed_dir, ref_dir)

    device = "cuda"

    # 1) Load COLMAP and pick all front/back cameras (train + eval)
    parser = Parser(data_dir=args.data_dir, factor=args.data_factor,
                    normalize=False, test_every=1)
    selected = [i for i, n in enumerate(parser.image_names)
                if camera_direction(n) in ("front", "back")]
    print(f"[render] {len(selected)} front/back poses (train+eval)")

    perp_table = build_perpendicular_table(parser)
    print(f"[render] perpendicular directions computed for {len(perp_table)} viewpoints")

    # 2) Load splats
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    splats = {k: v.to(device) for k, v in ckpt["splats"].items()}
    means = splats["means"]
    quats = splats["quats"]
    scales = torch.exp(splats["scales"])
    opacities = torch.sigmoid(splats["opacities"])
    colors = torch.cat([splats["sh0"], splats["shN"]], 1)
    print(f"[render] loaded {means.shape[0]} Gaussians from step {ckpt['step']}")

    # 3) Load Difix pipeline (same as the trainer's fix step)
    difix = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
    difix.set_progress_bar_config(disable=True)
    difix.to(device)
    print("[render] Difix pipeline ready")

    for out_i, idx in enumerate(selected):
        name = parser.image_names[idx]
        vp = viewpoint_id(name)
        c2w_orig = parser.camtoworlds[idx].copy()
        perp_dir = perp_table[vp]

        camera_id = parser.camera_ids[idx]
        K = parser.Ks_dict[camera_id]
        W, H = parser.imsize_dict[camera_id]
        K_t = torch.from_numpy(K).to(device).float().unsqueeze(0)

        for side in sides:
            sign = side_signs[side]
            c2w_perturbed = c2w_orig.copy()
            c2w_perturbed[:3, 3] = c2w_orig[:3, 3] + perp_dir * args.distance * sign

            c2w_t = torch.from_numpy(c2w_perturbed).to(device).float().unsqueeze(0)

            render_colors, _, _ = rasterization(
                means=means, quats=quats, scales=scales, opacities=opacities,
                colors=colors, viewmats=torch.linalg.inv(c2w_t), Ks=K_t,
                width=W, height=H, sh_degree=args.sh_degree,
                near_plane=0.01, far_plane=1e10, camera_model="pinhole",
            )
            pred_np = (torch.clamp(render_colors[0], 0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
            pred_img = Image.fromarray(pred_np)

            # nearest training image as Difix reference (excludes self)
            ref_path = find_nearest_train_path(parser, c2w_perturbed[:3, 3], idx)
            ref_img = Image.open(ref_path).convert("RGB")

            fixed_img = difix(prompt="remove degradation", image=pred_img,
                              ref_image=ref_img, num_inference_steps=1,
                              timesteps=[199], guidance_scale=0.0).images[0]
            fixed_img = fixed_img.resize(pred_img.size, Image.LANCZOS)

            pred_dir, fixed_dir, ref_dir = side_dirs[side]
            stem = f"{out_i:04d}_{name.replace('/', '_').replace('.jpg', '').replace('.png', '')}"
            pred_img.save(os.path.join(pred_dir, f"{stem}.png"))
            fixed_img.save(os.path.join(fixed_dir, f"{stem}.png"))
            ref_img.save(os.path.join(ref_dir, f"{stem}.png"))
            print(f"[render] side={side}  {stem}  vp={vp} dir={camera_direction(name)}  "
                  f"ref={os.path.basename(ref_path)}")


if __name__ == "__main__":
    main()
