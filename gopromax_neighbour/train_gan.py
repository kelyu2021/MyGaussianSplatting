"""
GoPro Max Neighbour – GAN-based Adversarial Training for Gaussian Splatting
===========================================================================

Loads a pre-trained 3DGS checkpoint, then fine-tunes the Gaussians with a
WGAN-GP critic so that **off-road / jittered camera paths** produce the same
rendering quality as the original on-road cameras.

Architecture
------------
* **Generator** = the Gaussian Splatting model (renders images from any viewpoint).
* **Critic**    = a lightweight PatchGAN-style CNN that scores image patches
                  as "real" (on-road rendering) or "fake" (off-road rendering).

Training loop (per epoch)
-------------------------
1. For each training camera, build a corresponding *jittered* camera
   (lateral shift = ``road_width``, default 0.5 m).
2. Render from BOTH the original and jittered camera.
3. **Critic step**: maximise  ``critic(real) − critic(fake)``  (+ GP).
4. **Generator step**: minimise ``−critic(fake)``
   + reconstruction losses (L1 + D-SSIM) on the on-road view.
   => Gaussians learn to produce high-quality images even off-road.

Usage::

    cd MyGaussianSplatting/gopromax_neighbour
    python train_gan.py \\
        --config configs/gopromax_neighbour_1200.yaml \\
        --model_root output_version_2 \\
        --road_width 0.5 \\
        --epoch 1200
"""

from __future__ import annotations

import os
import sys
import math
import copy
import argparse
import csv
from pathlib import Path
from random import shuffle, seed as set_seed
from collections import OrderedDict

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ── Reuse everything from the base training script ──────────────────
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
    load_camera,
    read_scene,
    prepare_output,
    evaluate,
    l1_loss,
    ssim,
    psnr,
    save_log_images,
    FACE_TO_CAM_ID,
    TENSORBOARD_FOUND,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    pass

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ═══════════════════════════════════════════════════════════════════════════
#  §0  Two-Row Log Image Helper
# ═══════════════════════════════════════════════════════════════════════════

def save_log_images_gan(
    log_dir: str, epoch: int,
    gt_on, rendered_on, depth_on, acc_on,
    rendered_off, depth_off, acc_off,
    sky_mask=None, moving_mask=None,
):
    """Save an 8-panel visualisation grid (2 rows × 4 cols).

    Top row:    GT | Rendered | Depth | Acc                (original camera)
    Bottom row: Rendered | Depth | Acc | Sky+Moving mask   (perturbed camera)

    Mask panel colour coding:
      Red   = sky,  Green = moving objects,  Yellow = overlap,  Black = valid
    """
    import torchvision

    def _to3(t):
        c = t.detach().cpu().float()
        if c.dim() == 2:
            c = c.unsqueeze(0)
        if c.shape[0] == 1:
            c = c.expand(3, -1, -1)
        return c

    def _norm_depth(dep):
        d = dep.detach().cpu().float()
        d = (d - d.min()) / (d.max() - d.min() + 1e-6)
        return d.expand(3, -1, -1) if d.shape[0] == 1 else d

    gt_c = _to3(gt_on)
    rn_on_c = _to3(rendered_on)
    dep_on_c = _norm_depth(depth_on)
    acc_on_c = _to3(acc_on)

    rn_off_c = _to3(rendered_off)
    dep_off_c = _norm_depth(depth_off)
    acc_off_c = _to3(acc_off)

    # Combined mask panel: R=sky, G=moving, B=0
    H, W = gt_c.shape[1], gt_c.shape[2]
    mask_vis = torch.zeros(3, H, W)
    if sky_mask is not None:
        # sky_mask: 1=valid, 0=sky  ->  sky region = (1 - sky_mask)
        s = sky_mask.detach().cpu().float()
        if s.dim() == 3:
            s = s.squeeze(0)
        mask_vis[0] = 1.0 - s          # Red channel = sky
    if moving_mask is not None:
        # moving_mask: 1=moving, 0=static
        m = moving_mask.detach().cpu().float()
        if m.dim() == 3:
            m = m.squeeze(0)
        mask_vis[1] = m                 # Green channel = moving

    top = [gt_c, rn_on_c, dep_on_c, acc_on_c]
    bot = [mask_vis,rn_off_c, dep_off_c, acc_off_c]

    grid = torchvision.utils.make_grid(
        top + bot, nrow=4, padding=2, normalize=False)
    path = os.path.join(log_dir, f"epoch_{epoch:04d}.png")
    torchvision.utils.save_image(grid, path)


# ═══════════════════════════════════════════════════════════════════════════
#  §1  Critic Network (PatchGAN / WGAN-GP)
# ═══════════════════════════════════════════════════════════════════════════

class Critic(nn.Module):
    """Lightweight PatchGAN critic for WGAN-GP.

    Input: (B, 3, H, W) rendered RGB image.
    Output: (B, 1) scalar score (higher = more realistic).
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        # Progressive downsampling: 4 conv blocks
        # Each block: Conv -> InstanceNorm -> LeakyReLU
        # No BatchNorm (WGAN-GP recommends InstanceNorm or LayerNorm)
        def _block(ch_in, ch_out, stride=2):
            return nn.Sequential(
                nn.Conv2d(ch_in, ch_out, 4, stride=stride, padding=1),
                nn.InstanceNorm2d(ch_out, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            )

        bc = base_channels
        self.features = nn.Sequential(
            # First layer: no norm
            nn.Conv2d(in_channels, bc, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            _block(bc, bc * 2),       # -> bc*2
            _block(bc * 2, bc * 4),   # -> bc*4
            _block(bc * 4, bc * 8),   # -> bc*8
        )
        # Adaptive pooling -> (B, bc*8, 1, 1) -> scalar
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bc * 8, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) -> (B, 1) Wasserstein score."""
        feat = self.features(x)
        return self.head(feat)


def gradient_penalty(
    critic: Critic,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Compute WGAN-GP gradient penalty.

    Interpolates between real and fake images, runs them through the critic,
    and penalises gradients that deviate from unit norm.
    """
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    scores = critic(interpolated)
    grad_outputs = torch.ones_like(scores)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(B, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


# ═══════════════════════════════════════════════════════════════════════════
#  §2  Build Jittered (Off-Road) Cameras
# ═══════════════════════════════════════════════════════════════════════════

def _parse_face(filename: str):
    stem = Path(filename).stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in FACE_TO_CAM_ID:
        return parts[0], parts[1]
    return stem, "unknown"


def compute_trajectory_directions(sparse_dir: str):
    """Compute the forward, up, and lateral directions from COLMAP poses.

    Returns (forward, up, lateral) as unit numpy arrays.
    """
    sparse = Path(sparse_dir)
    images_bin = read_images_binary(str(sparse / "images.bin"))

    # Group by frame
    frame_groups: OrderedDict[str, list] = OrderedDict()
    for img in sorted(images_bin.values(), key=lambda x: x.name):
        frame_name, face_name = _parse_face(img.name)
        R_c2w = qvec2rotmat(img.qvec).T
        T_w2c = img.tvec
        frame_groups.setdefault(frame_name, []).append(
            (face_name, R_c2w, T_w2c))

    frames = list(frame_groups.items())

    # Compute frame centres
    centres = []
    for _, faces in frames:
        R_c2w, T_w2c = faces[0][1], faces[0][2]
        C = -R_c2w @ T_w2c
        centres.append(C)
    centres = np.array(centres)

    # Walking direction
    forward = centres[-1] - centres[0]
    forward /= np.linalg.norm(forward) + 1e-12

    # Up direction (average of camera up vectors)
    up_accum = np.zeros(3)
    for _, faces in frames:
        for _, R_c2w, _ in faces:
            up_accum += R_c2w @ np.array([0.0, -1.0, 0.0])
    up = up_accum / (np.linalg.norm(up_accum) + 1e-12)

    # Lateral direction
    lateral = np.cross(forward, up)
    lateral /= np.linalg.norm(lateral) + 1e-12

    return forward, up, lateral


def build_jittered_camera(
    cam: Camera,
    up: np.ndarray,
    road_width: float,
) -> Camera:
    """Build an off-road version of an on-road camera.

    Shifts the camera laterally by ``road_width`` metres, perpendicular
    to the camera's own forward (viewing) direction in the ground plane.
    """
    R_c2w = cam.R                # (3,3) numpy
    T_w2c = cam.T                # (3,)  numpy

    # Camera forward direction in world coords (camera -Z axis)
    cam_forward = R_c2w @ np.array([0.0, 0.0, -1.0])

    # Lateral = perpendicular to camera forward, in the ground plane
    lateral = np.cross(cam_forward, up)
    lateral /= np.linalg.norm(lateral) + 1e-12

    # Old camera centre in world coords
    C_old = -R_c2w @ T_w2c

    # Shift laterally
    C_new = C_old + road_width * lateral

    # Orientation unchanged
    R_c2w_new = R_c2w
    T_w2c_new = -R_c2w_new.T @ C_new

    # Build a new Camera
    jittered = Camera(
        uid=cam.id,
        R=R_c2w_new,
        T=T_w2c_new,
        FoVx=cam.FoVx,
        FoVy=cam.FoVy,
        K=cam.K.cpu().numpy(),
        image=cam.original_image.clone(),
        image_name=f"jitter_{cam.image_name}",
        metadata=cam.meta.copy() if hasattr(cam, "meta") else {},
        guidance={},
    )
    return jittered


# ═══════════════════════════════════════════════════════════════════════════
#  §3  Load Pre-Trained Checkpoint
# ═══════════════════════════════════════════════════════════════════════════

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
    print(f"  {gaussians.num_points:,} Gaussians, "
          f"SH degree {gaussians.active_sh_degree}")
    return gaussians, loaded_epoch, state


# ═══════════════════════════════════════════════════════════════════════════
#  §4  GAN Training Loop
# ═══════════════════════════════════════════════════════════════════════════

def training_gan(cfg: dict, model_root: str, road_width: float,
                 pretrained_epoch: int | None, gan_epochs: int,
                 critic_iters: int, lambda_gp: float,
                 lambda_recon: float, lambda_dssim: float,
                 lr_critic: float, lr_generator: float):
    """Main GAN-based fine-tuning entry point."""
    workspace = os.getcwd()
    data_cfg = cfg["data"]
    optim_cfg = cfg["optim"]

    # ── Output directories ────────────────────────────────────────────
    # Override exp_name so GAN outputs go to a separate folder
    gan_cfg = copy.deepcopy(cfg)
    gan_cfg["exp_name"] = cfg["exp_name"] + "_gan"
    dirs = prepare_output(gan_cfg, workspace)
    tb_writer = dirs["tb_writer"]

    # ── GPU ───────────────────────────────────────────────────────────
    gpus = cfg.get("gpus", [0])
    if gpus and gpus[0] >= 0:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpus[0]))

    white_bg = data_cfg.get("white_background", False)
    bg_color = torch.tensor(
        [1, 1, 1] if white_bg else [0, 0, 0],
        dtype=torch.float32, device="cuda")

    # ── Load dataset (for real on-road images & cameras) ──────────────
    scene_info = read_scene(
        source_path=cfg["source_path"],
        point_cloud_path=data_cfg["point_cloud_path"],
        images_dir=data_cfg["images"],
        mask_dir=data_cfg.get("mask_dir", ""),
        moving_mask_dir=data_cfg.get("moving_mask_dir", ""),
        split_test=data_cfg.get("split_test", 8),
        workspace=workspace,
    )

    # Save cameras.json
    import json as _json

    def _fov2focal(fov, pixels):
        return pixels / (2.0 * math.tan(fov / 2.0))

    def _camera_info_to_json(cid, cam_info):
        Rt = np.eye(4)
        Rt[:3, :3] = cam_info.R.T
        Rt[:3, 3] = cam_info.T
        W2C = np.linalg.inv(Rt)
        pos = W2C[:3, 3]
        rot = W2C[:3, :3]
        return {
            'id': cid,
            'img_name': cam_info.image_name,
            'width': cam_info.width,
            'height': cam_info.height,
            'position': pos.tolist(),
            'rotation': [x.tolist() for x in rot],
            'fy': _fov2focal(cam_info.FovY, cam_info.height),
            'fx': _fov2focal(cam_info.FovX, cam_info.width),
        }

    json_cams = []
    all_cam_infos = list(scene_info.test_cameras) + list(scene_info.train_cameras)
    for cid, ci in enumerate(all_cam_infos):
        json_cams.append(_camera_info_to_json(cid, ci))
    cam_json_path = os.path.join(dirs["model_path"], "cameras.json")
    with open(cam_json_path, "w") as fp:
        _json.dump(json_cams, fp)
    print(f"Saved cameras.json ({len(json_cams)} cameras) to {cam_json_path}")

    print("Loading training cameras ...")
    train_cameras = [load_camera(ci) for ci in tqdm(scene_info.train_cameras)]
    print("Loading test cameras ...")
    test_cameras = [load_camera(ci) for ci in tqdm(scene_info.test_cameras)]

    # ── Resolve source path & compute trajectory directions ───────────
    src_path = cfg["source_path"]
    if not os.path.isabs(src_path):
        src_path = os.path.join(workspace, src_path)
    sparse_dir = os.path.join(src_path, "sparse")

    forward_dir, up_dir, lateral_dir = compute_trajectory_directions(
        sparse_dir)
    print(f"Trajectory directions:")
    print(f"  forward : {forward_dir}")
    print(f"  up      : {up_dir}")
    print(f"  lateral : {lateral_dir}  (road_width = {road_width:.2f} m)")

    # ── Load pre-trained Gaussians ────────────────────────────────────
    model_path = os.path.join(
        workspace, model_root, cfg["task"], cfg["exp_name"])
    trained_model_dir = os.path.join(model_path, "trained_model")

    sh_degree = cfg.get("model", {}).get("sh_degree",
                cfg.get("model", {}).get("gaussian", {}).get("sh_degree", 3))
    gaussians, loaded_epoch, ckpt_state = _load_checkpoint(
        trained_model_dir, pretrained_epoch, sh_degree)

    print(f"Loaded pre-trained model from epoch {loaded_epoch}")

    # ── Setup Gaussian optimizer for fine-tuning ──────────────────────
    # Use lower learning rates for fine-tuning
    ft_optim_cfg = copy.deepcopy(optim_cfg)
    ft_optim_cfg["position_lr_init"] = lr_generator
    ft_optim_cfg["position_lr_final"] = lr_generator * 0.1
    ft_optim_cfg["position_lr_max_epochs"] = gan_epochs
    ft_optim_cfg["position_lr_max_steps"] = (
        gan_epochs * len(train_cameras))
    ft_optim_cfg["feature_lr"] = lr_generator * 5.0
    ft_optim_cfg["opacity_lr"] = lr_generator * 100.0
    ft_optim_cfg["scaling_lr"] = lr_generator * 10.0
    ft_optim_cfg["rotation_lr"] = lr_generator * 2.0

    gaussians.training_setup(ft_optim_cfg)
    # Restore active SH degree to max (already trained)
    gaussians.active_sh_degree = gaussians.max_sh_degree

    # ── Create Critic ─────────────────────────────────────────────────
    critic = Critic(in_channels=3, base_channels=64).cuda()
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=lr_critic, betas=(0.0, 0.9))

    # ── CSV logging ───────────────────────────────────────────────────
    train_csv_path = dirs["train_csv_path"]
    if not os.path.isfile(train_csv_path):
        with open(train_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "loss_critic", "loss_gen", "loss_recon",
                "wasserstein_dist", "psnr_onroad", "psnr_offroad",
                "n_points",
            ])

    # ── Training ──────────────────────────────────────────────────────
    print(f"\nGAN fine-tuning: {gan_epochs} epochs × "
          f"{len(train_cameras)} cameras/epoch")
    print(f"  critic_iters = {critic_iters}")
    print(f"  lambda_gp    = {lambda_gp}")
    print(f"  lambda_recon = {lambda_recon}")
    print(f"  lambda_dssim = {lambda_dssim}")
    print(f"  lr_critic    = {lr_critic}")
    print(f"  lr_generator = {lr_generator}")

    step = 0
    progress = tqdm(range(gan_epochs), desc="GAN Epochs", unit="ep")

    for epoch in range(1, gan_epochs + 1):
        viewpoint_stack = list(train_cameras)
        shuffle(viewpoint_stack)

        # Per-epoch accumulators
        ep_critic_loss = 0.0
        ep_gen_loss = 0.0
        ep_recon_loss = 0.0
        ep_w_dist = 0.0
        ep_psnr_on = 0.0
        ep_psnr_off = 0.0
        ep_count = 0

        # Stash last generator step images for logging
        log_gt_on = log_render_on = log_depth_on = log_acc_on = None
        log_render_off = log_depth_off = log_acc_off = None
        log_sky_mask = log_moving_mask = None

        for cam_idx, cam in enumerate(viewpoint_stack):
            step += 1

            # ── Build jittered camera ─────────────────────────────
            jit_cam = build_jittered_camera(cam, up_dir, road_width)

            # ── Load masks ────────────────────────────────────────
            sky_mask = None
            if "mask" in cam.guidance:
                sky_mask = cam.guidance["mask"]
                sky_mask = (sky_mask.cuda(non_blocking=True)
                            if not sky_mask.is_cuda else sky_mask)

            moving_mask = None
            if "moving_mask" in cam.guidance:
                moving_mask = cam.guidance["moving_mask"]
                moving_mask = (moving_mask.cuda(non_blocking=True)
                               if not moving_mask.is_cuda else moving_mask)

            mask = sky_mask
            if mask is not None and moving_mask is not None:
                mask = mask & (~moving_mask)
            elif moving_mask is not None:
                mask = ~moving_mask

            gt_image = cam.original_image
            gt_image = (gt_image.cuda(non_blocking=True)
                        if not gt_image.is_cuda else gt_image)

            # ==================================================
            #  CRITIC STEP  (train critic, freeze Gaussians)
            # ==================================================
            if cam_idx % (critic_iters + 1) < critic_iters:
                critic.train()

                # Render on-road (real) — detach from Gaussian graph
                with torch.no_grad():
                    real_pkg = render(cam, gaussians, bg_color)
                    real_img = real_pkg["rgb"].detach()  # (3, H, W)

                # Render off-road (fake) — detach from Gaussian graph
                with torch.no_grad():
                    fake_pkg = render(jit_cam, gaussians, bg_color)
                    fake_img = fake_pkg["rgb"].detach()  # (3, H, W)

                # Critic scores
                real_score = critic(real_img.unsqueeze(0))  # (1, 1)
                fake_score = critic(fake_img.unsqueeze(0))  # (1, 1)

                # WGAN loss: maximise E[C(real)] - E[C(fake)]
                # => minimise E[C(fake)] - E[C(real)]
                w_dist = real_score.mean() - fake_score.mean()
                loss_c = -w_dist

                # Gradient penalty
                gp = gradient_penalty(
                    critic, real_img.unsqueeze(0),
                    fake_img.unsqueeze(0), real_img.device)
                loss_c = loss_c + lambda_gp * gp

                critic_optimizer.zero_grad()
                loss_c.backward()
                critic_optimizer.step()

                ep_critic_loss += loss_c.item()
                ep_w_dist += w_dist.item()

            # ==================================================
            #  GENERATOR STEP  (train Gaussians, freeze critic)
            # ==================================================
            else:
                critic.eval()

                # Render on-road (for reconstruction loss)
                real_pkg = render(cam, gaussians, bg_color)
                real_img = real_pkg["rgb"]
                viewspace_pts = real_pkg["viewspace_points"]
                visibility = real_pkg["visibility_filter"]
                radii = real_pkg["radii"]

                # Render off-road (for adversarial loss)
                fake_pkg = render(jit_cam, gaussians, bg_color)
                fake_img = fake_pkg["rgb"]

                # --- Adversarial loss: fool the critic ---
                fake_score = critic(fake_img.unsqueeze(0))
                loss_adv = -fake_score.mean()

                # --- Reconstruction loss on on-road view ---
                Ll1 = l1_loss(real_img, gt_image, mask)
                loss_recon = (
                    (1.0 - lambda_dssim) * Ll1 +
                    lambda_dssim * (1.0 - ssim(real_img, gt_image, mask=mask)))

                # --- SH regularisation ---
                lambda_sh = optim_cfg.get("lambda_sh_reg", 1e-3)
                sh_reg = torch.tensor(0.0, device="cuda")
                if lambda_sh > 0:
                    sh_rest = gaussians._features_rest
                    sh_reg = lambda_sh * (sh_rest ** 2).mean()

                # --- Sky opacity penalty ---
                lambda_sky_acc = optim_cfg.get("lambda_sky_acc", 0.01)
                sky_loss = torch.tensor(0.0, device="cuda")
                if lambda_sky_acc > 0 and sky_mask is not None:
                    acc = real_pkg["acc"]
                    sky_region = 1.0 - sky_mask.float()
                    sky_loss = lambda_sky_acc * (acc * sky_region).mean()

                # --- Total generator loss ---
                loss_g = (loss_adv +
                          lambda_recon * loss_recon +
                          sh_reg +
                          sky_loss)

                loss_g.backward()

                # Density control stats
                with torch.no_grad():
                    gaussians.set_max_radii2D(radii, visibility)
                    gaussians.add_densification_stats(
                        viewspace_pts, visibility)

                gaussians.update_optimizer()

                ep_gen_loss += loss_g.item()
                ep_recon_loss += loss_recon.item()

                # Logging metrics (detached)
                with torch.no_grad():
                    ep_psnr_on += psnr(real_img, gt_image, mask).item()
                    ep_psnr_off += psnr(fake_img, gt_image, mask).item()

                # Stash for end-of-epoch log images
                log_gt_on = gt_image.detach()
                log_render_on = real_img.detach()
                log_depth_on = real_pkg["depth"].detach()
                log_acc_on = real_pkg["acc"].detach()
                log_render_off = fake_img.detach()
                log_depth_off = fake_pkg["depth"].detach()
                log_acc_off = fake_pkg["acc"].detach()
                log_sky_mask = sky_mask
                log_moving_mask = moving_mask

            ep_count += 1

        # ══════════════════════════════════════════════════════════
        #  END OF EPOCH
        # ══════════════════════════════════════════════════════════
        progress.update(1)
        n_gen_steps = max(1, ep_count - (ep_count * critic_iters
                          // (critic_iters + 1)))
        n_crit_steps = max(1, ep_count - n_gen_steps)

        avg_c = ep_critic_loss / n_crit_steps
        avg_g = ep_gen_loss / n_gen_steps
        avg_r = ep_recon_loss / n_gen_steps
        avg_w = ep_w_dist / n_crit_steps
        avg_psnr_on = ep_psnr_on / n_gen_steps
        avg_psnr_off = ep_psnr_off / n_gen_steps

        progress.set_postfix({
            "C": f"{avg_c:.4f}",
            "G": f"{avg_g:.4f}",
            "W": f"{avg_w:.4f}",
            "PSNR_on": f"{avg_psnr_on:.2f}",
            "#G": f"{gaussians.num_points:,}",
        })

        # ── CSV logging ───────────────────────────────────────────
        with open(train_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, avg_c, avg_g, avg_r, avg_w,
                avg_psnr_on, avg_psnr_off, gaussians.num_points,
            ])

        # ── TensorBoard ──────────────────────────────────────────
        if tb_writer:
            tb_writer.add_scalar("gan/loss_critic", avg_c, epoch)
            tb_writer.add_scalar("gan/loss_generator", avg_g, epoch)
            tb_writer.add_scalar("gan/loss_recon", avg_r, epoch)
            tb_writer.add_scalar("gan/wasserstein_dist", avg_w, epoch)
            tb_writer.add_scalar("gan/psnr_onroad", avg_psnr_on, epoch)
            tb_writer.add_scalar("gan/psnr_offroad", avg_psnr_off, epoch)

        # ── Save log images every 10 epochs ───────────────────────
        if epoch % 10 == 0 and log_render_on is not None:
            try:
                save_log_images_gan(
                    dirs["log_images_dir"], epoch,
                    log_gt_on, log_render_on, log_depth_on, log_acc_on,
                    log_render_off, log_depth_off, log_acc_off,
                    sky_mask=log_sky_mask,
                    moving_mask=log_moving_mask,
                )
            except Exception:
                pass

        # ── Evaluation every 50 epochs ────────────────────────────
        if epoch % 50 == 0 or epoch == gan_epochs:
            with torch.no_grad():
                evaluate(test_cameras, gaussians, bg_color,
                         epoch, tb_writer,
                         eval_csv_path=dirs["eval_csv_path"],
                         split="test",
                         n_points=gaussians.num_points,
                         save_dir=dirs["model_path"])

        # ── Save checkpoint every 50 epochs + final ───────────────
        if epoch % 50 == 0 or epoch == gan_epochs:
            with torch.no_grad():
                sd = gaussians.save_state_dict(
                    is_final=(epoch == gan_epochs))
                sd["epoch"] = epoch
                sd["step"] = step
                sd["pretrained_epoch"] = loaded_epoch
                ckpt_path = os.path.join(
                    dirs["trained_model_dir"],
                    f"epoch_{epoch}.pth")
                torch.save(sd, ckpt_path)
                print(f"\n[GAN EPOCH {epoch}] Checkpoint -> {ckpt_path}")

                # Also save critic
                critic_path = os.path.join(
                    dirs["trained_model_dir"],
                    f"critic_epoch_{epoch}.pth")
                torch.save({
                    "critic_state_dict": critic.state_dict(),
                    "critic_optimizer": critic_optimizer.state_dict(),
                    "epoch": epoch,
                }, critic_path)

        # ── Save PLY every 100 epochs + final ─────────────────────
        if epoch % 100 == 0 or epoch == gan_epochs:
            with torch.no_grad():
                ply_path = os.path.join(
                    dirs["point_cloud_dir"],
                    f"iteration_{epoch}", "point_cloud.ply")
                gaussians.save_ply(ply_path)

    print("\nGAN fine-tuning complete.")


# ═══════════════════════════════════════════════════════════════════════════
#  §5  Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GAN-based adversarial fine-tuning for Gaussian Splatting")
    parser.add_argument(
        "--config", default="configs/gopromax_neighbour_1200.yaml",
        help="Path to YAML config file.")
    parser.add_argument(
        "--model_root", default="output_version_2",
        help="Root directory containing the pre-trained model.")
    parser.add_argument(
        "--road_width", type=float, default=0.5,
        help="Lateral shift in metres for jittered cameras (default: 0.5).")
    parser.add_argument(
        "--epoch", type=int, default=None,
        help="Pre-trained checkpoint epoch to load (default: latest).")
    parser.add_argument(
        "--gan_epochs", type=int, default=200,
        help="Number of GAN fine-tuning epochs (default: 200).")
    parser.add_argument(
        "--critic_iters", type=int, default=5,
        help="Critic updates per generator update (default: 5).")
    parser.add_argument(
        "--lambda_gp", type=float, default=10.0,
        help="Gradient penalty coefficient (default: 10.0).")
    parser.add_argument(
        "--lambda_recon", type=float, default=10.0,
        help="Reconstruction loss weight (default: 10.0).")
    parser.add_argument(
        "--lambda_dssim", type=float, default=0.2,
        help="D-SSIM weight within reconstruction loss (default: 0.2).")
    parser.add_argument(
        "--lr_critic", type=float, default=1e-4,
        help="Critic learning rate (default: 1e-4).")
    parser.add_argument(
        "--lr_generator", type=float, default=1e-5,
        help="Generator (Gaussian params) learning rate (default: 1e-5).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Task: {cfg['task']}  Exp: {cfg['exp_name']}")

    # Reproducibility
    set_seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.autograd.set_detect_anomaly(False)

    training_gan(
        cfg,
        model_root=args.model_root,
        road_width=args.road_width,
        pretrained_epoch=args.epoch,
        gan_epochs=args.gan_epochs,
        critic_iters=args.critic_iters,
        lambda_gp=args.lambda_gp,
        lambda_recon=args.lambda_recon,
        lambda_dssim=args.lambda_dssim,
        lr_critic=args.lr_critic,
        lr_generator=args.lr_generator,
    )


if __name__ == "__main__":
    main()
