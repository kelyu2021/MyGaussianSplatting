#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# train_neighbour_sky.py
#   train_neighbour.py + an SH environment-sky model.
#
# The sky is represented as a single set of view-direction SH coefficients
# (default degree 3 → 48 params) and composited behind the Gaussian render:
#
#     final = gaussian_rgb + (1 - alpha) * sky_rgb
#
# At pixels where the Depth-Anything-V2 inverse-depth map is exactly 0 (sky),
# we (a) push the rasterizer's accumulated alpha toward 0 with an opacity
# supervision loss and (b) skip the inverse-depth supervision (which would
# otherwise fight transparency). Once alpha → 0 over the sky, the rasterizer's
# depth contribution is 0 there, so the displayed alpha-weighted inv-depth is
# also 0 — matching the GT convention.
#

import os
import re
import copy
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import randint, choice
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.sky_model import SkySHModel
from utils.general_utils import safe_state, get_expon_lr_func
from utils.graphics_utils import getWorld2View2
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch.modules.lpips import LPIPS

_ID_RE = re.compile(r"^(\d+)")
_FACE_RE = re.compile(r"_(front|back|left|right|up|down|top|bottom)(?:\.[A-Za-z0-9]+)?$")
_VALID_DIRECTIONS = ("left", "right", "up", "front", "back")

def _image_id(cam):
    """Extract the leading numeric id from cam.image_name (e.g. '0008_front' -> 8)."""
    m = _ID_RE.match(cam.image_name)
    return int(m.group(1)) if m else None

def _camera_face(cam):
    """Return the cubemap face suffix ('front', 'back', …) from cam.image_name, or None."""
    m = _FACE_RE.search(cam.image_name)
    return m.group(1) if m else None


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _sky_mask_from_cam(cam):
    """Per-pixel sky mask from the raw mono inv-depth (1 where sky, 0 elsewhere).

    `invmonodepth_raw` is the unmodified (pre scale/offset) signal from disk
    cached by the Camera ctor — sky pixels stay at exactly 0 there, so the
    mask is robust to the depth_params rescale that happens downstream.

    Returns a (1, H, W) float tensor on CUDA, or None if no depth was loaded.
    """
    raw = getattr(cam, "invmonodepth_raw", None)
    if raw is None:
        return None
    return (raw == 0).float().to("cuda", non_blocking=True)


# ----------------------------------------------------------------------
# WGAN-GP critic (PatchGAN), ported from gopromax_neighbour/train_da2loss_critic.py
# ----------------------------------------------------------------------
class Critic(nn.Module):
    """Lightweight PatchGAN critic for WGAN-GP. (B, 3, H, W) -> (B, 1)."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        def _block(ch_in, ch_out, stride=2):
            return nn.Sequential(
                nn.Conv2d(ch_in, ch_out, 4, stride=stride, padding=1),
                nn.InstanceNorm2d(ch_out, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            )
        bc = base_channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, bc, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            _block(bc, bc * 2),
            _block(bc * 2, bc * 4),
            _block(bc * 4, bc * 8),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bc * 8, 1),
        )
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def _critic_saliency_map(critic, img):
    """|∂(-C(img))/∂img|: per-pixel gradient saliency of the critic w.r.t. the
    rendered image. Bright = "pixels the critic considers fake."

    Returns (1, H, W) — raw L2 norm of gradient across channels.
    No smoothing, no sqrt compression, no normalization. Absolute scale
    is preserved so comparisons across images/iterations are meaningful.
    Uses `torch.autograd.grad` so the critic's parameters' `.grad` is
    left untouched (no interference with the WGAN-GP optimiser state).
    """
    was_training = critic.training
    critic.eval()
    with torch.enable_grad():
        x = img.detach().clone().requires_grad_(True)
        score = critic(x.unsqueeze(0))
        grads = torch.autograd.grad(
            outputs=-score.mean(),
            inputs=x,
            create_graph=False, retain_graph=False,
        )[0]
    if was_training:
        critic.train()
    return grads.detach().abs().sum(dim=0, keepdim=True)                    # (1, H, W)


def _hot_colormap(g):
    """Single-channel [0,1] → 3-channel 'hot' colormap (black→red→yellow→white).
    Input (1,H,W); output (3,H,W). For visualising saliency heatmaps.
    """
    g = g.clamp(0.0, 1.0).squeeze(0)                                        # (H, W)
    r   = (3.0 * g).clamp(0.0, 1.0)
    grn = (3.0 * g - 1.0).clamp(0.0, 1.0)
    b   = (3.0 * g - 2.0).clamp(0.0, 1.0)
    return torch.stack([r, grn, b], dim=0)                                  # (3, H, W)


def _gradient_penalty(critic, real, fake, device):
    """WGAN-GP gradient penalty on a linear real/fake interpolation."""
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=device)
    interp = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    scores = critic(interp)
    grad = torch.autograd.grad(
        outputs=scores, inputs=interp,
        grad_outputs=torch.ones_like(scores),
        create_graph=True, retain_graph=True,
    )[0].view(B, -1)
    return ((grad.norm(2, dim=1) - 1.0) ** 2).mean()


# ----------------------------------------------------------------------
# SplatWeaver-style high-frequency prior (arXiv 2605.07287, Eq. 5)
# ----------------------------------------------------------------------
# Per-pixel HF energy from a single-level Haar DWT:
#     (LL, LH, HL, HH) = DWT(I);   HF = (sqrt(LH^2 + HL^2 + HH^2)) ↑2
# In the paper this drives expert routing over how many Gaussians to
# spawn per pixel. In per-scene 3DGS we don't have routing, so the
# transferable mechanism is to use HF as a per-pixel loss weight: high-
# frequency pixels get amplified gradients, which feeds larger viewspace
# gradients into the existing densify_grad_threshold criterion ("dense
# where complex, sparse where smooth").
_HF_KERNELS = None
def _haar_hf_map(img):
    """Single-level Haar HF energy of a (3, H, W) image. Returns (1, H, W) in [0, 1]."""
    global _HF_KERNELS
    if img.dim() == 3:
        img = img.unsqueeze(0)
    gray = (0.2126 * img[:, 0:1] + 0.7152 * img[:, 1:2] + 0.0722 * img[:, 2:3])
    _, _, H, W = gray.shape
    pad_h, pad_w = (H % 2), (W % 2)
    if pad_h or pad_w:
        gray = F.pad(gray, (0, pad_w, 0, pad_h), mode="replicate")
    if _HF_KERNELS is None or _HF_KERNELS.device != gray.device or _HF_KERNELS.dtype != gray.dtype:
        _HF_KERNELS = torch.tensor([
            [[0.5,  0.5], [-0.5, -0.5]],   # LH (horizontal)
            [[0.5, -0.5], [ 0.5, -0.5]],   # HL (vertical)
            [[0.5, -0.5], [-0.5,  0.5]],   # HH (diagonal)
        ], dtype=gray.dtype, device=gray.device).unsqueeze(1)
    sub = F.conv2d(gray, _HF_KERNELS, stride=2)                       # (1, 3, H/2, W/2)
    hf = torch.sqrt(sub.pow(2).sum(dim=1, keepdim=True) + 1e-8)       # (1, 1, H/2, W/2)
    hf = F.interpolate(hf, size=(img.shape[-2], img.shape[-1]),
                       mode="bilinear", align_corners=False)
    hf = hf / hf.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    return hf.squeeze(0)                                              # (1, H, W)


# ----------------------------------------------------------------------
# Off-path jittered cameras (ported from gopromax_neighbour/train_da2loss_critic.py)
# ----------------------------------------------------------------------
# Goal: train the splats so renders from camera centres *off* the captured
# trajectory still look photorealistic. We build a sibling "jittered"
# camera per iteration by shifting the centre by `road_width` world units
# in a chosen direction, render it, and let a WGAN-GP critic compare
# on-path render (real) vs off-path render (fake). The off-path render
# has no GT to anchor it, so the only signal pushing it toward realism is
# the adversarial term.
def _compute_trajectory_basis(cameras):
    """Derive (forward, up, lateral) unit world vectors from camera centres.

    forward : centres[last] − centres[first], normalised (walking direction).
    up      : average of each camera's world-space +up axis (= R_c2w · [0, -1, 0]).
    lateral : forward × up.
    """
    sorted_cams = sorted(cameras, key=lambda c: c.image_name)
    centres = np.array([(-c.R @ c.T).astype(np.float64) for c in sorted_cams])  # (N, 3)
    if len(centres) < 2:
        return (np.array([0.0, 0.0, 1.0]),
                np.array([0.0, -1.0, 0.0]),
                np.array([1.0, 0.0, 0.0]))
    forward = centres[-1] - centres[0]
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    up_accum = np.zeros(3, dtype=np.float64)
    for c in sorted_cams:
        up_accum += (c.R @ np.array([0.0, -1.0, 0.0])).astype(np.float64)
    up = up_accum / (np.linalg.norm(up_accum) + 1e-12)
    lateral = np.cross(forward, up)
    lateral = lateral / (np.linalg.norm(lateral) + 1e-12)
    return forward, up, lateral


def _build_jittered_camera(cam, road_width, direction, forward, up, lateral):
    """Shallow copy `cam` with centre shifted by `road_width` along `direction`.

    Orientation (R) is unchanged; only the translation T (and the derived
    world_view_transform / full_proj_transform / camera_center) move.
    `forward`/`up`/`lateral` are unit world-coord numpy vectors from
    `_compute_trajectory_basis` — keeping the shift basis at *trajectory*
    level (not per-camera) so "left" means the same world direction on
    every cubemap face.
    """
    if direction == "left":
        shift = lateral
    elif direction == "right":
        shift = -lateral
    elif direction == "up":
        shift = up
    elif direction == "front":
        shift = forward
    elif direction == "back":
        shift = -forward
    else:
        raise ValueError(f"Unknown jitter direction: {direction!r}")

    R_c2w = cam.R                                                # numpy (3, 3)
    T_w2c = cam.T                                                # numpy (3,)
    C_old = -R_c2w @ T_w2c                                       # world-space centre
    C_new = C_old + float(road_width) * shift
    T_new = -R_c2w.T @ C_new                                     # new w2c translation

    jit = copy.copy(cam)                                         # share image/depth tensors
    jit.R = R_c2w
    jit.T = T_new
    jit.world_view_transform = torch.tensor(
        getWorld2View2(R_c2w, T_new, cam.trans, cam.scale)
    ).transpose(0, 1).cuda()
    jit.full_proj_transform = (
        jit.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))
    ).squeeze(0)
    jit.camera_center = jit.world_view_transform.inverse()[3, :3]
    jit.image_name = f"jit_{direction}_{cam.image_name}"
    return jit


def _composite_render(viewpoint_cam, gaussians, sky_model, pipe, separate_sh,
                      use_trained_exp):
    """Render Gaussians with bg=0, evaluate sky SH, composite, return a dict.

    Output dict mirrors `render()` plus:
        sky_rgb         : (3, H, W) — sky color before compositing
        composited      : (3, H, W) — gauss_rgb + (1-α)·sky_rgb
        alpha           : (1, H, W) — accumulated Gaussian alpha
        invdepth_disp   : (1, H, W) — alpha-weighted inv-depth for viz (sky → 0)
    """
    bg0 = torch.zeros(3, device="cuda")
    pkg = render(viewpoint_cam, gaussians, pipe, bg0,
                 use_trained_exp=use_trained_exp, separate_sh=separate_sh)

    alpha = pkg["alpha"]
    if alpha is None:
        raise RuntimeError(
            "Sky-composite rendering requires a rasterizer that returns "
            "accumulated alpha (raster_out[3]). The installed "
            "diff_gaussian_rasterization in this env doesn't expose it."
        )

    sky_rgb = sky_model(viewpoint_cam)                                  # (3, H, W)
    gauss_rgb = pkg["render"]                                           # (3, H, W)
    composited = gauss_rgb + (1.0 - alpha) * sky_rgb                    # broadcast (1,H,W)

    # Display inv-depth: alpha-weighted so sky pixels (α → 0) are exactly 0.
    invdepth_disp = alpha * (1.0 / pkg["depth"].clamp(min=1e-6))

    pkg["sky_rgb"] = sky_rgb
    pkg["composited"] = composited
    pkg["invdepth_disp"] = invdepth_disp
    return pkg


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,
             sky_sh_degree, lambda_sky_opacity, sky_lr_init, sky_lr_final,
             critic_start_iter, critic_iters, lambda_adv, lambda_gp,
             lambda_drift, lr_critic, critic_base_channels,
             use_hf_prior, lambda_hf_loss,
             use_offroad_critic, road_width, road_width_init_frac,
             road_width_warmup_iters, jitter_directions, jitter_faces):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    # Sky model: SH (degree `sky_sh_degree`) environment, RGB at each pixel is
    # eval_sh(view-dir in world space). Composited behind the Gaussian render.
    sky_model = SkySHModel(sh_degree=sky_sh_degree).cuda()
    sky_model.training_setup(
        lr_init=sky_lr_init, lr_final=sky_lr_final, max_steps=opt.iterations,
    )
    print(f"[sky] SH degree={sky_sh_degree}  coeffs/channel={(sky_sh_degree+1)**2}  "
          f"total_params={(sky_sh_degree+1)**2 * 3}  "
          f"lambda_sky_opacity={lambda_sky_opacity}  "
          f"lr(init→final)={sky_lr_init}→{sky_lr_final}")

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        # Resume sky weights from the matching iteration if present.
        sky_ckpt = os.path.join(scene.model_path, f"sky_iter_{first_iter}.pth")
        if os.path.isfile(sky_ckpt):
            sky_model.load(sky_ckpt)
            print(f"[sky] resumed from {sky_ckpt}")

    # Note: rendering uses bg=0 unconditionally so the Gaussian contribution
    # is the pure Σ T_i α_i c_i term, and (1-α) is the room left for the sky.
    # `dataset.white_background` / `opt.random_background` are intentionally
    # ignored for the gaussian render — they don't compose well with a learned
    # sky.

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    # ── Custom train/test split: image_id % 8 == 0 → test ───────────────
    all_cams = list(scene.getTrainCameras()) + list(scene.getTestCameras())
    train_cams, test_cams = [], []
    for c in all_cams:
        iid = _image_id(c)
        if iid is not None and iid > 0 and iid % 8 == 0:
            test_cams.append(c)
        else:
            train_cams.append(c)
    test_ids = sorted({_image_id(c) for c in test_cams})
    print(f"[split] total={len(all_cams)}  train={len(train_cams)}  test={len(test_cams)}")
    print(f"[split] test image_ids (% 8 == 0): {test_ids}")
    if not train_cams:
        sys.exit("[split] ERROR: no training cameras after split.")

    # Sky-mask reliability breakdown (one-time).
    n_total = len(train_cams)
    n_with_skymask = sum(1 for c in train_cams if getattr(c, "invmonodepth_raw", None) is not None)
    print(f"[sky] train viewpoints: total={n_total}  with_skymask_loaded={n_with_skymask}")

    # ── WGAN-GP critic (off-path adversarial supervision) ────────────────
    # The critic scores the off-path (jittered) render as fake and the
    # on-path render as real. The generator (Gaussians + sky) gets
    # λ_adv·(-C(fake)) added to its loss; the critic is updated for K
    # WGAN-GP micro-steps each iter. Created iff --use_offroad_critic.
    critic = None
    critic_optimizer = None
    if use_offroad_critic:
        critic = Critic(in_channels=3, base_channels=critic_base_channels).cuda()
        critic_optimizer = torch.optim.Adam(
            critic.parameters(), lr=lr_critic, betas=(0.0, 0.9))
        if checkpoint:
            critic_ckpt = os.path.join(scene.model_path, f"critic_iter_{first_iter}.pth")
            if os.path.isfile(critic_ckpt):
                critic.load_state_dict(torch.load(critic_ckpt, map_location="cuda"))
                print(f"[critic] resumed from {critic_ckpt}")
        print(f"[critic] enabled  critic_base_channels={critic_base_channels}  "
              f"critic_start_iter={critic_start_iter}  critic_iters={critic_iters}  "
              f"lambda_adv={lambda_adv}  lambda_gp={lambda_gp}  lambda_drift={lambda_drift}  lr_critic={lr_critic}")
    else:
        print("[critic] disabled")

    if use_hf_prior:
        print(f"[hf-prior] enabled  λ_hf_loss={lambda_hf_loss}  "
              f"(Haar-DWT high-freq weighting on L1; SplatWeaver arXiv 2605.07287)")
    else:
        print("[hf-prior] disabled")

    # ── Trajectory basis (always computed; needed for jitter eval grid) ──
    # Used both by the off-path adversarial branch and by `training_report`
    # to dump per-eval "jitter_images/<cam>.png" 2×5 grids showing how the
    # current splats look from camera centres shifted ±1..4 world units.
    traj_forward, traj_up, traj_lateral = _compute_trajectory_basis(train_cams)

    if use_offroad_critic:
        if road_width <= 0:
            sys.exit("[off-road] --use_offroad_critic requires --road_width > 0 (world units).")
        for d in jitter_directions:
            if d not in _VALID_DIRECTIONS:
                sys.exit(f"[off-road] invalid jitter direction {d!r}; choose from {_VALID_DIRECTIONS}")
        jitter_face_set = set(jitter_faces) if jitter_faces else None      # None = no face filter
        if jitter_face_set is not None:
            n_eligible = sum(1 for c in train_cams if _camera_face(c) in jitter_face_set)
            print(f"[off-road] face filter: {sorted(jitter_face_set)}  "
                  f"eligible cams: {n_eligible}/{len(train_cams)}")
            if n_eligible == 0:
                sys.exit("[off-road] ERROR: no train cameras match --jitter_faces; aborting.")
        else:
            print(f"[off-road] face filter: ALL  eligible cams: {len(train_cams)}/{len(train_cams)}")
        print(f"[off-road] enabled  road_width={road_width:.3f}  "
              f"init_frac={road_width_init_frac:.3f}  warmup_iters={road_width_warmup_iters}  "
              f"directions={list(jitter_directions)}")
    else:
        jitter_face_set = None
        print("[off-road] disabled")

    # ── LPIPS network (instantiated once, reused for eval) ──────────────
    lpips_model = LPIPS(net_type='vgg').to("cuda").eval()
    for p in lpips_model.parameters():
        p.requires_grad_(False)

    viewpoint_stack = train_cams.copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    # Depth-supervision reliability breakdown (one-time, set at Camera ctor).
    n_with_depth = sum(1 for c in train_cams if getattr(c, "invdepthmap", None) is not None)
    n_reliable = sum(1 for c in train_cams if getattr(c, "depth_reliable", False))
    pct = (100.0 * n_reliable / n_total) if n_total else 0.0
    print(f"[depth-reg] train viewpoints: total={n_total}  with_depth_loaded={n_with_depth}  "
          f"depth_reliable={n_reliable} ({pct:.1f}%)  "
          f"depth_l1_weight(init→final)={opt.depth_l1_weight_init}→{opt.depth_l1_weight_final}")

    depth_branch_taken = 0
    depth_branch_skipped = 0

    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    ema_sky_opa_for_log = 0.0
    ema_psnr_for_log = 0.0
    ema_ssim_for_log = 0.0
    ema_loss_adv_for_log = 0.0
    ema_loss_critic_for_log = 0.0
    ema_w_dist_for_log = 0.0
    ema_gp_for_log = 0.0
    ema_real_score_for_log = 0.0
    ema_fake_score_for_log = 0.0
    critic_iters_run = 0
    offroad_iters_run = 0
    n_post_warmup = 0                       # iters with iteration > critic_start_iter

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    # Network-GUI preview uses composited image.
                    pkg_gui = _composite_render(
                        custom_cam, gaussians, sky_model, pipe,
                        separate_sh=SPARSE_ADAM_AVAILABLE,
                        use_trained_exp=dataset.train_test_exp,
                    )
                    net_image = pkg_gui["composited"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        sky_model.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera (from the custom train split)
        if not viewpoint_stack:
            viewpoint_stack = train_cams.copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = _composite_render(
            viewpoint_cam, gaussians, sky_model, pipe,
            separate_sh=SPARSE_ADAM_AVAILABLE,
            use_trained_exp=dataset.train_test_exp,
        )
        image = render_pkg["composited"]
        alpha = render_pkg["alpha"]                                       # (1, H, W)
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask_view = viewpoint_cam.alpha_mask.cuda()
            image = image * alpha_mask_view

        # Loss (photometric on composited image)
        gt_image = viewpoint_cam.original_image.cuda()

        # SplatWeaver-style HF prior: per-pixel weight w = 1 + λ_hf · HF(GT).
        # Amplifies L1 gradient in high-frequency regions → larger viewspace
        # gradients → existing densify_grad_threshold densifies those areas
        # more. Disabled (weight ≡ 1) unless --use_hf_prior is set.
        if use_hf_prior and lambda_hf_loss > 0:
            with torch.no_grad():
                hf_w = 1.0 + lambda_hf_loss * _haar_hf_map(gt_image)      # (1, H, W)
            Ll1 = (torch.abs(image - gt_image) * hf_w).mean()
        else:
            Ll1 = l1_loss(image, gt_image)

        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Sky-mask opacity loss: push α → 0 at pixels where the GT inv-depth is 0.
        # This is what frees the Gaussians from modelling the sky and lets the
        # SH sky take over there.
        sky_mask = _sky_mask_from_cam(viewpoint_cam)                      # (1, H, W) or None
        Ll1_sky_opa = 0.0
        if sky_mask is not None and lambda_sky_opacity > 0:
            Ll1_sky_opa_pure = (alpha * sky_mask).mean()
            Ll1_sky_opa_t = lambda_sky_opacity * Ll1_sky_opa_pure
            loss = loss + Ll1_sky_opa_t
            Ll1_sky_opa = Ll1_sky_opa_t.item()

        # Depth regularization — gated to non-sky pixels.
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = 1.0 / render_pkg["depth"].clamp(min=1e-6)
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            if sky_mask is not None:
                depth_mask = depth_mask * (1.0 - sky_mask)               # drop sky pixels

            Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss = loss + Ll1depth
            Ll1depth = Ll1depth.item()
            depth_branch_taken += 1
        else:
            Ll1depth = 0
            depth_branch_skipped += 1

        # WGAN adversarial term (off-path only).
        #   real = on-path render `image`
        #   fake = render from a centre-shifted (jittered) camera, no GT.
        # The off-path branch needs a 2nd render per iter so we keep it
        # cheap by gating on critic_start_iter and the face filter.
        loss_adv_val = 0.0
        jit_pkg = None
        jit_image = None
        offroad_eligible = (
            use_offroad_critic
            and (jitter_face_set is None
                 or _camera_face(viewpoint_cam) in jitter_face_set)
        )
        critic_active = (critic is not None and iteration >= critic_start_iter)
        if critic_active and offroad_eligible:
            ramp_t = min(1.0,
                         (iteration - critic_start_iter) / max(1, road_width_warmup_iters))
            cur_road_width = road_width * (
                road_width_init_frac + (1.0 - road_width_init_frac) * ramp_t)
            direction = choice(list(jitter_directions))
            jit_cam = _build_jittered_camera(
                viewpoint_cam, cur_road_width, direction,
                forward=traj_forward, up=traj_up, lateral=traj_lateral,
            )
            jit_pkg = _composite_render(
                jit_cam, gaussians, sky_model, pipe,
                separate_sh=SPARSE_ADAM_AVAILABLE,
                use_trained_exp=dataset.train_test_exp,
            )
            jit_image = jit_pkg["composited"]
            critic.eval()
            fake_score_g = critic(jit_image.unsqueeze(0))
            loss_adv = lambda_adv * (-fake_score_g.mean())
            loss = loss + loss_adv
            loss_adv_val = loss_adv.item()

        loss.backward()

        iter_end.record()
        iter_end.synchronize()

        with torch.no_grad():
            elapsed_ms = iter_start.elapsed_time(iter_end)
            psnr_live = psnr(image, gt_image).mean().item()
            ssim_live = float(ssim_value.item())

            # EMAs for progress bar.
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            ema_sky_opa_for_log = 0.4 * Ll1_sky_opa + 0.6 * ema_sky_opa_for_log
            ema_psnr_for_log = 0.4 * psnr_live + 0.6 * ema_psnr_for_log
            ema_ssim_for_log = 0.4 * ssim_live + 0.6 * ema_ssim_for_log

            if tb_writer and iteration % 500 == 0:
                seen = depth_branch_taken + depth_branch_skipped
                hit_pct = 100.0 * depth_branch_taken / max(seen, 1)
                gauss_n = scene.gaussians.get_xyz.shape[0]
                tb_writer.add_scalar('train/l1', Ll1.item(), iteration)
                tb_writer.add_scalar('train/ssim', ssim_live, iteration)
                tb_writer.add_scalar('train/psnr', psnr_live, iteration)
                tb_writer.add_scalar('train/total_loss', loss.item(), iteration)
                tb_writer.add_scalar('train/depth_l1', Ll1depth, iteration)
                tb_writer.add_scalar('train/sky_opacity', Ll1_sky_opa, iteration)
                tb_writer.add_scalar('train/iter_time_ms', elapsed_ms, iteration)
                tb_writer.add_scalar('depth/weight', depth_l1_weight(iteration), iteration)
                tb_writer.add_scalar('depth/hit_pct', hit_pct, iteration)
                tb_writer.add_scalar('gaussians/count', gauss_n, iteration)
                tb_writer.add_scalar('gaussians/sh_degree', scene.gaussians.active_sh_degree, iteration)
                tb_writer.add_scalar('sky/lr',
                    sky_model.optimizer.param_groups[0]['lr'] if sky_model.optimizer else 0.0,
                    iteration)

            if iteration % 500 == 0:
                seen = depth_branch_taken + depth_branch_skipped
                dpct = (100.0 * depth_branch_taken / seen) if seen else 0.0
                pb = {
                    "Loss":  f"{ema_loss_for_log:.5f}",
                    "PSNR":  f"{ema_psnr_for_log:.2f}",
                    "SSIM":  f"{ema_ssim_for_log:.4f}",
                    "DepL":  f"{ema_Ll1depth_for_log:.5f}",
                    "SkyO":  f"{ema_sky_opa_for_log:.5f}",
                    "Hit%":  f"{dpct:.1f}",
                    "G":     f"{scene.gaussians.get_xyz.shape[0]/1000:.0f}k",
                }
                if use_offroad_critic:
                    # W↑ = critic learning faster than gen. W↓ = gen catching up.
                    # GP near 0–1 is healthy; >>1 means lambda_gp too low.
                    pb["Adv"] = f"{ema_loss_adv_for_log:+.4f}"
                    pb["W"]   = f"{ema_w_dist_for_log:+.3f}"
                    pb["GP"]  = f"{ema_gp_for_log:.3f}"
                progress_bar.set_postfix(pb, refresh=False)
                progress_bar.update(500)
            if iteration == opt.iterations:
                progress_bar.close()
                seen = depth_branch_taken + depth_branch_skipped
                dpct = (100.0 * depth_branch_taken / seen) if seen else 0.0
                print(f"[depth-reg] iterations: total={seen}  depth_branch_taken={depth_branch_taken} "
                      f"({dpct:.1f}%)  depth_branch_skipped={depth_branch_skipped}")
                if use_offroad_critic:
                    crun_pct = (100.0 * critic_iters_run / max(1, n_post_warmup))
                    print(f"[critic] final EMAs:  W={ema_w_dist_for_log:+.4f}  "
                          f"real={ema_real_score_for_log:+.4f}  fake={ema_fake_score_for_log:+.4f}  "
                          f"GP={ema_gp_for_log:.4f}  L_c={ema_loss_critic_for_log:+.4f}  "
                          f"L_adv={ema_loss_adv_for_log:+.4f}")
                    print(f"[critic] iters fired: {critic_iters_run}/{n_post_warmup} "
                          f"({crun_pct:.1f}% of post-start iters)")
                    elig_pct = (100.0 * offroad_iters_run / max(1, n_post_warmup))
                    ramp_t_end = min(1.0,
                        (iteration - critic_start_iter) / max(1, road_width_warmup_iters))
                    cur_rw_end = road_width * (road_width_init_frac
                                               + (1.0 - road_width_init_frac) * ramp_t_end)
                    print(f"[off-road] final road_width={cur_rw_end:.3f}  ramp_t={ramp_t_end:.3f}  "
                          f"eligible_iters={offroad_iters_run}/{n_post_warmup} "
                          f"({elig_pct:.1f}%)")

            # Periodic eval + image logging
            training_report(tb_writer, iteration, train_cams, test_cams, scene,
                            sky_model, pipe, SPARSE_ADAM_AVAILABLE,
                            lpips_model, dataset.train_test_exp, testing_iterations,
                            traj_forward=traj_forward, traj_up=traj_up,
                            traj_lateral=traj_lateral,
                            critic=critic, jitter_face_set=jitter_face_set)
            # Critic health snapshot at every test iteration.
            if iteration in testing_iterations and use_offroad_critic:
                crun_pct = (100.0 * critic_iters_run / max(1, n_post_warmup))
                print(f"[ITER {iteration}] critic: "
                      f"W={ema_w_dist_for_log:+.4f}  "
                      f"real={ema_real_score_for_log:+.4f}  "
                      f"fake={ema_fake_score_for_log:+.4f}  "
                      f"GP={ema_gp_for_log:.4f}  "
                      f"L_c={ema_loss_critic_for_log:+.4f}  "
                      f"L_adv={ema_loss_adv_for_log:+.4f}  "
                      f"fired={critic_iters_run}/{n_post_warmup}({crun_pct:.0f}%)")
                if critic_active:
                    ramp_t_log = min(1.0,
                        (iteration - critic_start_iter) / max(1, road_width_warmup_iters))
                    cur_rw_log = road_width * (road_width_init_frac
                                               + (1.0 - road_width_init_frac) * ramp_t_log)
                    elig_pct = (100.0 * offroad_iters_run / max(1, n_post_warmup))
                    print(f"[ITER {iteration}] off-road: "
                          f"road_width={cur_rw_log:.3f}  ramp_t={ramp_t_log:.2f}  "
                          f"eligible={offroad_iters_run}/{n_post_warmup}({elig_pct:.0f}%)")
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians + sky".format(iteration))
                scene.save(iteration)
                sky_model.save(os.path.join(scene.model_path, f"sky_iter_{iteration}.pth"))

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # Off-path densification: also accumulate viewspace gradient
                # magnitudes from the jittered render. This is what turns the
                # adversarial signal into clone/split: Gaussians at the edge
                # of black holes (visible only from off-path views) get a
                # large viewspace gradient from −C(jit_image), cross
                # densify_grad_threshold, and split into children that drift
                # to cover the hole over subsequent iters. max_radii2D is
                # updated too so the prune step doesn't kill Gaussians only
                # visible from shifted viewpoints.
                if jit_pkg is not None:
                    jit_vsp = jit_pkg["viewspace_points"]
                    jit_vis = jit_pkg["visibility_filter"]
                    jit_radii = jit_pkg["radii"]
                    gaussians.max_radii2D[jit_vis] = torch.max(
                        gaussians.max_radii2D[jit_vis], jit_radii[jit_vis])
                    gaussians.add_densification_stats(jit_vsp, jit_vis)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)

                # White-background opacity reset is intentionally skipped — sky
                # compositing replaces the role white_background used to play.
                if iteration % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                sky_model.step()

        # ── Critic update (WGAN-GP, K micro-steps with detached renders) ──
        # Outside no_grad because GP needs autograd.
        #   real = on-path composited render
        #   fake = off-path (jittered) composited render
        # Only runs when the picked cam passed the face filter (we have
        # jit_image then) and we are past the warmup.
        # Per-iteration critic-side scalars (default 0 when critic didn't fire
        # this iter — face filter / pre-warmup).
        loss_critic_val = 0.0
        w_dist_val = 0.0
        gp_val = 0.0
        real_score_val = 0.0
        fake_score_val = 0.0
        run_critic_step = critic_active and jit_image is not None
        if run_critic_step:
            critic.train()
            real_img_d = image.detach().clamp(0.0, 1.0)
            fake_img_d = jit_image.detach().clamp(0.0, 1.0)
            for _ in range(max(1, critic_iters)):
                real_score = critic(real_img_d.unsqueeze(0))
                fake_score = critic(fake_img_d.unsqueeze(0))
                w_dist = real_score.mean() - fake_score.mean()
                loss_c = -w_dist
                gp = _gradient_penalty(critic, real_img_d.unsqueeze(0),
                                       fake_img_d.unsqueeze(0), real_img_d.device)
                # Epsilon-drift: anchors raw scores near 0 so the critic learns
                # to separate real/fake instead of inflating output magnitude
                # (PGGAN/Karras). Does not affect the input-gradient the
                # generator sees, only the absolute score scale.
                drift = real_score.pow(2).mean() + fake_score.pow(2).mean()
                loss_c = loss_c + lambda_gp * gp + lambda_drift * drift
                critic_optimizer.zero_grad(set_to_none=True)
                loss_c.backward()
                critic_optimizer.step()
            # Snapshot last micro-step for logging. W_dist trend is the
            # primary signal: it tends to grow while the critic is still
            # learning, then plateau / shrink as the generator catches up.
            # GP should hover near 0–1; runaway GP ⇒ critic isn't Lipschitz,
            # raise --lambda_gp. real/fake both drifting up together ⇒
            # score inflation / weak signal, raise --lambda_drift.
            loss_critic_val = loss_c.item()
            w_dist_val = w_dist.item()
            gp_val = gp.item()
            real_score_val = real_score.mean().item()
            fake_score_val = fake_score.mean().item()
            critic_iters_run += 1
        if offroad_eligible and critic_active:
            offroad_iters_run += 1
        if critic_active:
            n_post_warmup += 1

        with torch.no_grad():
            ema_loss_adv_for_log    = 0.4 * loss_adv_val    + 0.6 * ema_loss_adv_for_log
            ema_loss_critic_for_log = 0.4 * loss_critic_val + 0.6 * ema_loss_critic_for_log
            ema_w_dist_for_log      = 0.4 * w_dist_val      + 0.6 * ema_w_dist_for_log
            ema_gp_for_log          = 0.4 * gp_val          + 0.6 * ema_gp_for_log
            ema_real_score_for_log  = 0.4 * real_score_val  + 0.6 * ema_real_score_for_log
            ema_fake_score_for_log  = 0.4 * fake_score_val  + 0.6 * ema_fake_score_for_log

            if tb_writer and iteration % 500 == 0:
                # Generator-side adversarial term + reconstruction balance.
                tb_writer.add_scalar('train/loss_adv', loss_adv_val, iteration)
                # Critic loss components (raw, this iter):
                #   loss_c_total  = −W + λ_gp · GP
                tb_writer.add_scalar('critic/loss_total',  loss_critic_val, iteration)
                tb_writer.add_scalar('critic/w_dist',      w_dist_val,      iteration)
                tb_writer.add_scalar('critic/real_score',  real_score_val,  iteration)
                tb_writer.add_scalar('critic/fake_score',  fake_score_val,  iteration)
                tb_writer.add_scalar('critic/gp_raw',      gp_val,          iteration)
                tb_writer.add_scalar('critic/gp_weighted', lambda_gp * gp_val, iteration)
                # Curriculum + branch firing rate (only after warmup).
                if critic_active:
                    tb_writer.add_scalar('critic/fired',
                                         1.0 if run_critic_step else 0.0, iteration)
                    if use_offroad_critic:
                        ramp_t_log = min(1.0,
                                         (iteration - critic_start_iter) /
                                         max(1, road_width_warmup_iters))
                        cur_rw_log = road_width * (road_width_init_frac
                                                   + (1.0 - road_width_init_frac) * ramp_t_log)
                        tb_writer.add_scalar('offroad/cur_road_width', cur_rw_log, iteration)
                        tb_writer.add_scalar('offroad/ramp_t',          ramp_t_log, iteration)
                        tb_writer.add_scalar('offroad/eligible_pct',
                                             100.0 * offroad_iters_run /
                                             max(1, n_post_warmup),
                                             iteration)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                sky_model.save(os.path.join(scene.model_path, f"sky_iter_{iteration}.pth"))
                if critic is not None:
                    torch.save(critic.state_dict(),
                               os.path.join(scene.model_path, f"critic_iter_{iteration}.pth"))

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def _depth_to_gray(d):
    """Normalize a depth-like map to a 3-channel grayscale [0,1] tensor for logging."""
    d = d.detach().float()
    if d.dim() == 3 and d.shape[0] > 1:
        d = d[0:1]
    if d.dim() == 3:
        d = d.squeeze(0)
    d = torch.where(torch.isfinite(d), d, torch.zeros_like(d))
    valid = d > 0
    if valid.any():
        v = d[valid]
        lo = torch.quantile(v, 0.02)
        hi = torch.quantile(v, 0.98)
        denom = (hi - lo).clamp(min=1e-8)
        d_norm = ((d - lo) / denom).clamp(0.0, 1.0)
        d_norm = d_norm * valid.float()
    else:
        d_norm = torch.zeros_like(d)
    return d_norm.unsqueeze(0).expand(3, -1, -1).contiguous()


def _depth_saliency(depth):
    """Sobel gradient magnitude of a depth map → (3,H,W). Accepts (1,H,W) or
    (3,H,W); only the first channel is used. Raw absolute scale is preserved
    (no per-image normalization) so comparisons across columns are meaningful.
    """
    gray = depth[:1].unsqueeze(0)                                          # (1,1,H,W)
    dtype, device = gray.dtype, gray.device
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=dtype, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                            dtype=dtype, device=device).view(1, 1, 3, 3)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    mag = (gx ** 2 + gy ** 2).sqrt().squeeze(0)                            # (1,H,W)
    return mag.expand(3, -1, -1).contiguous()                              # (3,H,W)


def _depth_preproc(d):
    """Preprocess raw inv-depth: inf/nan→0, squeeze to (1,H,W). No normalization."""
    d = d.detach().float()
    if d.dim() == 3 and d.shape[0] > 1:
        d = d[0:1]
    if d.dim() == 2:
        d = d.unsqueeze(0)
    return torch.where(torch.isfinite(d), d, torch.zeros_like(d))          # (1,H,W)


def _joint_depth_to_gray(raw_list):
    """Jointly normalize a list of (1,H,W) raw depths → list of (3,H,W) in [0,1].
    All images share the same 2nd–98th percentile range so brightness is comparable
    across columns within the row."""
    valid_cats = [d.reshape(-1)[d.reshape(-1) > 0] for d in raw_list]
    valid_cats = [v for v in valid_cats if v.numel() > 0]
    if valid_cats:
        all_v = torch.cat(valid_cats)
        lo = torch.quantile(all_v, 0.02)
        hi = torch.quantile(all_v, 0.98)
    else:
        lo = torch.tensor(0.0, device=raw_list[0].device)
        hi = torch.tensor(1.0, device=raw_list[0].device)
    denom = (hi - lo).clamp(min=1e-8)
    result = []
    for d in raw_list:
        valid = d > 0
        d_norm = ((d - lo) / denom).clamp(0.0, 1.0) * valid.float()
        result.append(d_norm.expand(3, -1, -1).contiguous())
    return result


def _joint_scale(tensors):
    """Scale a list of (C,H,W) tensors by their shared global maximum → [0,1].
    Preserves relative magnitudes across all images in the row."""
    g_max = max(t.amax().item() for t in tensors)
    denom = max(g_max, 1e-8)
    return [(t / denom).clamp(0.0, 1.0) for t in tensors]


def _to_3ch(x):
    """Promote any (H,W) / (1,H,W) / (3,H,W) tensor to (3,H,W) in [0,1]."""
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.shape[0] == 1:
        x = x.expand(3, -1, -1)
    return x.clamp(0.0, 1.0).contiguous()


def _make_log_grid(gt_rgb, gt_depth, sky_mask_viz,
                   pred_rgb, pred_depth, pred_opacity,
                   sky_rgb, gauss_only_rgb):
    """Assemble a (3, 3H, 3W) composite:
        row 1: GT_RGB       | GT_DEPTH        | SKY_MASK
        row 2: PRED_RGB     | PRED_DEPTH      | PRED_ALPHA
        row 3: SKY_RGB      | GAUSS_ONLY_RGB  | (blank)
    """
    blank = torch.zeros_like(_to_3ch(gt_rgb))
    row1 = torch.cat([_to_3ch(gt_rgb),     _to_3ch(gt_depth),       _to_3ch(sky_mask_viz)], dim=-1)
    row2 = torch.cat([_to_3ch(pred_rgb),   _to_3ch(pred_depth),     _to_3ch(pred_opacity)], dim=-1)
    row3 = torch.cat([_to_3ch(sky_rgb),    _to_3ch(gauss_only_rgb), blank],                 dim=-1)
    return torch.cat([row1, row2, row3], dim=-2)


def training_report(tb_writer, iteration, train_cams, test_cams, scene: Scene,
                    sky_model, pipe, separate_sh,
                    lpips_model, train_test_exp, testing_iterations,
                    traj_forward=None, traj_up=None, traj_lateral=None,
                    critic=None, jitter_face_set=None):
    if iteration not in testing_iterations:
        return

    torch.cuda.empty_cache()

    # n_train_sample = min(5, len(train_cams))
    n_train_sample = len(train_cams)
    if n_train_sample > 0:
        stride = max(1, len(train_cams) // n_train_sample)
        train_sample = train_cams[::stride][:n_train_sample]
    else:
        train_sample = []

    validation_configs = (
        {'name': 'test',  'cameras': test_cams},
        {'name': 'train', 'cameras': train_sample},
    )

    for config in validation_configs:
        cams = config['cameras']
        if not cams:
            continue

        l1_sum = psnr_sum = ssim_sum = lpips_sum = 0.0
        n = len(cams)
        # Jitter grid is dumped only on the test split — train cams already
        # have GT on-path so the visualisation isn't informative. Face filter
        # mirrors the off-path supervision: lateral shifts only make sense on
        # forward / backward-facing cubemap faces (side faces shift along
        # their own optical axis, which we don't optimise for).
        jitter_face_eff = (jitter_face_set if jitter_face_set is not None
                           else {"front", "back"})
        do_jitter_split = (config['name'] == 'test')
        for i, viewpoint in enumerate(cams):
            pkg = _composite_render(
                viewpoint, scene.gaussians, sky_model, pipe,
                separate_sh=separate_sh, use_trained_exp=train_test_exp,
            )
            rendered = torch.clamp(pkg["composited"], 0.0, 1.0)
            alpha = pkg["alpha"]                                          # (1,H,W)
            sky_rgb = pkg["sky_rgb"]                                      # (3,H,W)
            gauss_only = pkg["render"]                                    # (3,H,W) bg=0
            invdepth_disp = pkg["invdepth_disp"]                          # (1,H,W)

            gt = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            if train_test_exp:
                rendered      = rendered[..., rendered.shape[-1] // 2:]
                gt            = gt[..., gt.shape[-1] // 2:]
                invdepth_disp = invdepth_disp[..., invdepth_disp.shape[-1] // 2:]
                alpha         = alpha[..., alpha.shape[-1] // 2:]
                sky_rgb       = sky_rgb[..., sky_rgb.shape[-1] // 2:]
                gauss_only    = gauss_only[..., gauss_only.shape[-1] // 2:]

            l1_sum    += torch.abs(rendered - gt).mean().item()
            psnr_sum  += psnr(rendered, gt).mean().item()
            ssim_sum  += ssim(rendered, gt).item()
            lpips_sum += lpips_model(rendered.unsqueeze(0), gt.unsqueeze(0)).item()

            if viewpoint.invdepthmap is not None:
                gt_depth_t = viewpoint.invdepthmap
                if train_test_exp:
                    gt_depth_t = gt_depth_t[..., gt_depth_t.shape[-1] // 2:]
            else:
                gt_depth_t = torch.zeros_like(rendered[:1])

            sky_mask_viz = _sky_mask_from_cam(viewpoint)
            if sky_mask_viz is None:
                sky_mask_viz = torch.zeros_like(rendered[:1])
            elif train_test_exp:
                sky_mask_viz = sky_mask_viz[..., sky_mask_viz.shape[-1] // 2:]
            sky_mask_viz = 1.0 - sky_mask_viz   # ← invert: 1 where non-sky, 0 where sky

            grid = _make_log_grid(
                gt_rgb          = gt,
                gt_depth        = _depth_to_gray(gt_depth_t),
                sky_mask_viz    = sky_mask_viz,
                pred_rgb        = rendered,
                pred_depth      = _depth_to_gray(invdepth_disp),
                pred_opacity    = alpha,
                sky_rgb         = sky_rgb,
                gauss_only_rgb  = gauss_only,
            )
            log_dir = os.path.join(scene.model_path, "log_images", f"iter_{iteration:06d}", config['name'])
            os.makedirs(log_dir, exist_ok=True)
            img_path = os.path.join(log_dir, f"{viewpoint.image_name}.png")
            grid_np = (grid.permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(grid_np).save(img_path)

            # ── Jitter-eval grid ──────────────────────────────────────
            # For all eligible test cams, dump a 5-row × 9-col image (4 rows
            # when no critic). Columns are ordered by physical camera position,
            # on-path in the centre:
            #   col order:  L+4u | L+3u | L+2u | L+1u | on-path | R+1u | R+2u | R+3u | R+4u
            #   row1: render   (black non-sky ⇒ hole)
            #   row2: depth    (log1p α·inv-depth, joint-normalised across the row)
            #   row3: depth-grad × critic saliency (fused; Sobel|∇raw-depth| gating
            #         critic interest. Plain depth gradient if no critic)
            #   row4: opacity  (accumulated α)
            #   row5: critic heatmap (|∂(-C)/∂pixels|, joint-scaled; critic only)
            # Reading guide:
            #   render  : black non-sky region ⇒ hole.
            #   depth   : alpha-weighted inv-depth; dark region surrounded by
            #             structured depth ⇒ missing geometry.
            #   depth-grad×critic : bright only where a depth edge and critic
            #             interest coincide ⇒ candidate splat sites on geometry
            #             boundaries the critic flags as fake.
            #   opacity : accumulated α; α≈0 outside sky ⇒ no splats covering
            #             this pixel from the shifted view. Fills with α≈1 as
            #             densification lands new splats.
            #   heatmap : |∂(-C(render))/∂pixels|; "hot" colormap (black →
            #             red → orange → yellow → white). White = where the
            #             critic most wants new splats added.
            have_basis = (traj_forward is not None
                          and traj_up is not None
                          and traj_lateral is not None)
            face_ok = (_camera_face(viewpoint) in jitter_face_eff)
            if (do_jitter_split and face_ok and have_basis):
                jit_distances = (1.0, 2.0, 3.0, 4.0)
                def _render_jitter(direction, dist):
                    jc = _build_jittered_camera(
                        viewpoint, dist, direction,
                        forward=traj_forward, up=traj_up, lateral=traj_lateral,
                    )
                    jp = _composite_render(
                        jc, scene.gaussians, sky_model, pipe,
                        separate_sh=separate_sh, use_trained_exp=train_test_exp,
                    )
                    rgb_   = torch.clamp(jp["composited"], 0.0, 1.0)
                    inv_d_ = jp["invdepth_disp"]
                    alp_   = jp["alpha"]
                    if train_test_exp:
                        rgb_   = rgb_[...,   rgb_.shape[-1]   // 2:]
                        inv_d_ = inv_d_[..., inv_d_.shape[-1] // 2:]
                        alp_   = alp_[...,   alp_.shape[-1]   // 2:]
                    return rgb_, inv_d_, alp_
                left_packs  = [_render_jitter("left",  d) for d in jit_distances]
                right_packs = [_render_jitter("right", d) for d in jit_distances]
                # Physical left→right order: L+4u..L+1u | on-path | R+1u..R+4u.
                # left_packs is [+1u..+4u], so reverse it for the left half.
                left_renders   = [p[0] for p in reversed(left_packs)]
                right_renders  = [p[0] for p in right_packs]
                left_opacities  = [_to_3ch(p[2]) for p in reversed(left_packs)]
                right_opacities = [_to_3ch(p[2]) for p in right_packs]
                on_path_opa     = _to_3ch(alpha)

                # ── Row 1: render ──
                row_render = torch.cat(left_renders + [rendered] + right_renders, dim=-1)

                # ── Row 2: depth — log1p, jointly normalised across all 9 cols ──
                gt_raw_d    = _depth_preproc(gt_depth_t)
                left_raw_d  = [_depth_preproc(p[1]) for p in reversed(left_packs)]
                right_raw_d = [_depth_preproc(p[1]) for p in right_packs]
                raw_sweep   = left_raw_d + [gt_raw_d] + right_raw_d
                depth_joint = _joint_depth_to_gray([torch.log1p(d) for d in raw_sweep])
                row_depth   = torch.cat(depth_joint, dim=-1)

                # Critic saliency sweep (raw |∂(-C)/∂pixels|), shared by the
                # fused row 3 and the row 5 heatmap. None if no critic.
                if critic is not None:
                    sal_sweep = ([_critic_saliency_map(critic, r) for r in left_renders]
                                 + [_critic_saliency_map(critic, rendered)]
                                 + [_critic_saliency_map(critic, r) for r in right_renders])
                else:
                    sal_sweep = None

                # ── Row 3: depth-gradient signal. Sobel on raw inv-depth, then
                #    (if a critic exists) fused multiplicatively with the critic
                #    saliency so depth edges GATE where the critic wants splats —
                #    bright only where a depth discontinuity AND critic interest
                #    coincide. Each modality is joint-scaled to [0,1] first so
                #    neither dominates by raw scale; the product is re-scaled for
                #    visibility. Falls back to plain depth gradient w/o critic. ──
                depth_g = [_depth_saliency(d)[:1] for d in raw_sweep]       # (1,H,W) each
                if sal_sweep is not None:
                    dg_n  = _joint_scale(depth_g)
                    cs_n  = _joint_scale(sal_sweep)
                    fused = _joint_scale([dg * cs for dg, cs in zip(dg_n, cs_n)])
                    row_dsal = torch.cat([_to_3ch(f) for f in fused], dim=-1)
                else:
                    dsal     = _joint_scale(depth_g)
                    row_dsal = torch.cat([_to_3ch(d) for d in dsal], dim=-1)

                # ── Row 4: opacity ──
                row_opa = torch.cat(left_opacities + [on_path_opa] + right_opacities, dim=-1)

                if sal_sweep is not None:
                    # ── Row 5: critic heatmap — jointly scale raw saliencies,
                    #    then colormap ──
                    heats = _joint_scale(sal_sweep)
                    row_heat = torch.cat([_hot_colormap(s) for s in heats], dim=-1)
                    jit_grid = torch.cat(
                        [row_render, row_depth, row_dsal, row_opa, row_heat], dim=-2)
                else:
                    jit_grid = torch.cat(
                        [row_render, row_depth, row_dsal, row_opa], dim=-2)
                jit_dir = os.path.join(scene.model_path, "log_images",
                                       f"iter_{iteration:06d}", "jitter_images")
                os.makedirs(jit_dir, exist_ok=True)
                jit_path = os.path.join(
                    jit_dir, f"{config['name']}_{viewpoint.image_name}.png")
                jg_np = (jit_grid.permute(1, 2, 0).clamp(0.0, 1.0)
                         .cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(jg_np).save(jit_path)

        l1_mean    = l1_sum    / n
        psnr_mean  = psnr_sum  / n
        ssim_mean  = ssim_sum  / n
        lpips_mean = lpips_sum / n

        print(f"\n[ITER {iteration}] {config['name']:<5s}: "
              f"L1={l1_mean:.5f}  PSNR={psnr_mean:.3f}  "
              f"SSIM={ssim_mean:.4f}  LPIPS={lpips_mean:.4f}  ({n} views)")

        if tb_writer:
            tb_writer.add_scalar(f"{config['name']}/l1",    l1_mean,    iteration)
            tb_writer.add_scalar(f"{config['name']}/psnr",  psnr_mean,  iteration)
            tb_writer.add_scalar(f"{config['name']}/ssim",  ssim_mean,  iteration)
            tb_writer.add_scalar(f"{config['name']}/lpips", lpips_mean, iteration)

    if tb_writer:
        tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        tb_writer.add_scalar('scene/total_points', scene.gaussians.get_xyz.shape[0], iteration)

    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters (with SH sky model)")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=None,
                        help="Iterations at which to run full eval. Defaults to every 1000 iters up to --iterations.")
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    # Sky-model knobs.
    parser.add_argument("--sky_sh_degree", type=int, default=6,
                        help="SH degree for the environment sky model (0..3).")
    parser.add_argument("--lambda_sky_opacity", type=float, default=0.05,
                        help="Weight of the sky-mask opacity-suppression loss "
                             "(pushes α → 0 where GT inv-depth == 0).")
    parser.add_argument("--sky_lr_init", type=float, default=1e-2)
    parser.add_argument("--sky_lr_final", type=float, default=1e-4)

    # WGAN-GP critic (ported from gopromax_neighbour/train_da2loss_critic.py).
    # Only used by --use_offroad_critic; the critic is built iff that flag is set.
    parser.add_argument("--critic_start_iter", type=int, default=3000,
                        help="Iteration at which the adversarial term and critic updates kick in.")
    parser.add_argument("--critic_iters", type=int, default=1,
                        help="Critic micro-updates per training iteration (K in WGAN-GP).")
    parser.add_argument("--lambda_adv", type=float, default=0.01,
                        help="Weight of the adversarial (-C(fake)) term on the generator loss.")
    parser.add_argument("--lambda_gp", type=float, default=10.0,
                        help="WGAN-GP gradient-penalty weight.")
    parser.add_argument("--lambda_drift", type=float, default=1e-3,
                        help="Epsilon-drift weight: penalizes critic score magnitude "
                             "(eps*(real^2+fake^2)) to anchor scores near 0 and prevent "
                             "output-magnitude drift. PGGAN/Karras; typical 1e-3.")
    parser.add_argument("--lr_critic", type=float, default=1e-4)
    parser.add_argument("--critic_base_channels", type=int, default=64)

    # SplatWeaver high-frequency densification prior (arXiv 2605.07287, Eq. 5).
    parser.add_argument("--use_hf_prior", action="store_true",
                        help="Weight the L1 loss by a Haar-DWT high-frequency energy map of GT, "
                             "biasing densification toward complex regions.")
    parser.add_argument("--lambda_hf_loss", type=float, default=1.0,
                        help="HF amplification: per-pixel L1 weight = 1 + λ_hf · HF_norm(GT).")

    # Off-path adversarial supervision (ported from train_da2loss_critic.py).
    # Trains the splats to render well from camera centres shifted off the
    # captured trajectory (e.g. several world-units left/right). This flag
    # is the master toggle: it builds the WGAN-GP critic and pairs
    # (real, fake) = (on-path render, off-path render).
    parser.add_argument("--use_offroad_critic", action="store_true",
                        help="Enable off-path jittered-camera adversarial supervision "
                             "(also builds the WGAN-GP critic).")
    parser.add_argument("--road_width", type=float, default=0.0,
                        help="Target lateral shift of the jittered camera, in world units.")
    parser.add_argument("--road_width_init_frac", type=float, default=0.1,
                        help="Curriculum: start at road_width · init_frac, ramp to road_width.")
    parser.add_argument("--road_width_warmup_iters", type=int, default=5000,
                        help="Iterations (from critic_start_iter) to ramp init_frac → 1.0.")
    parser.add_argument("--jitter_directions", nargs="+", default=["left", "right"],
                        choices=list(_VALID_DIRECTIONS),
                        help="Set of shift directions sampled uniformly each iter.")
    parser.add_argument("--jitter_faces", nargs="*", default=[],
                        help="Cubemap-face suffixes that are eligible for off-path supervision "
                             "(e.g. front back). Empty = all cameras eligible.")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.test_iterations is None:
        args.test_iterations = list(range(1000, args.iterations + 1, 1000))
    if args.iterations not in args.test_iterations:
        args.test_iterations.append(args.iterations)
    args.test_iterations = sorted(set(args.test_iterations))

    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args), op.extract(args), pp.extract(args),
        args.test_iterations, args.save_iterations, args.checkpoint_iterations,
        args.start_checkpoint, args.debug_from,
        sky_sh_degree=args.sky_sh_degree,
        lambda_sky_opacity=args.lambda_sky_opacity,
        sky_lr_init=args.sky_lr_init,
        sky_lr_final=args.sky_lr_final,
        critic_start_iter=args.critic_start_iter,
        critic_iters=args.critic_iters,
        lambda_adv=args.lambda_adv,
        lambda_gp=args.lambda_gp,
        lambda_drift=args.lambda_drift,
        lr_critic=args.lr_critic,
        critic_base_channels=args.critic_base_channels,
        use_hf_prior=args.use_hf_prior,
        lambda_hf_loss=args.lambda_hf_loss,
        use_offroad_critic=args.use_offroad_critic,
        road_width=args.road_width,
        road_width_init_frac=args.road_width_init_frac,
        road_width_warmup_iters=args.road_width_warmup_iters,
        jitter_directions=args.jitter_directions,
        jitter_faces=args.jitter_faces,
    )

    print("\nTraining complete.")
