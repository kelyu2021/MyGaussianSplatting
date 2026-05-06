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
import shutil
import argparse
import csv
from pathlib import Path
from random import shuffle, seed as set_seed, choice as random_choice
from collections import OrderedDict

import imageio.v2 as imageio
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
# Use the Depth Anything V2 (MiDaS-style) scale-and-shift-invariant loss
# defined in train_da2loss.py so the depth supervision here matches the
# original DA-v2 paper formulation.
from train_da2loss import depth_mono_loss  # noqa: E402

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    pass

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ═══════════════════════════════════════════════════════════════════════════
#  §0  3×3 Log Image Helper
# ═══════════════════════════════════════════════════════════════════════════

def save_log_images_gan(
    log_dir: str, epoch: int,
    gt_on, rendered_on, depth_on, acc_on,
    rendered_off, depth_off, acc_off,
    sky_mask=None, moving_mask=None, mono_depth=None,
):
    """Save a 3×3 grid to {log_dir}/epoch_{epoch:04d}.png.

    Row 1: GT RGB          | GT mask (sky=red, moving=green) | GT depth (mono)
    Row 2: Pred RGB        | Pred alpha (acc)                | Pred depth
    Row 3: Jittered RGB    | Jittered alpha (acc)            | Jittered depth

    All depth panels: greyscale, light = near, dark = far, empty/sky = black.
    """
    import numpy as np
    from PIL import Image as _Image

    def _to_rgb_uint8(t):
        arr = t.detach().cpu().float().numpy()
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        else:
            arr = np.transpose(arr, (1, 2, 0))
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            elif arr.shape[-1] > 3:
                arr = arr[..., :3]
        return np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    def _acc_to_grey(acc_t):
        a = acc_t.detach().cpu().numpy().squeeze()
        grey = (np.clip(a, 0, 1) * 255).astype(np.uint8)
        return np.stack([grey] * 3, axis=-1)

    def _depth_to_grey(depth_t_or_np, valid_mask=None, is_disparity=False,
                       pct_lo: float = 0.0, pct_hi: float = 1.0):
        """Greyscale depth panel: near = light, far = dark.

        If ``is_disparity`` is True, the input is treated as inverse depth
        (larger value = closer), so the brightness mapping is *not* inverted.
        Otherwise the input is metric depth (larger value = farther) and the
        mapping is inverted so that close points still appear bright.

        ``pct_lo``/``pct_hi`` (in [0, 1]) select percentile bounds over the
        valid pixels, suppressing the influence of a few outliers (e.g.
        stray near-Gaussian leaks into sky tiles).
        """
        if torch.is_tensor(depth_t_or_np):
            d = depth_t_or_np.detach().cpu().numpy().squeeze()
        else:
            d = np.asarray(depth_t_or_np).squeeze().astype(np.float32)
        out = np.zeros(d.shape, dtype=np.uint8)
        if valid_mask is None:
            valid_mask = np.isfinite(d) & (d > 0)
        if valid_mask.any():
            v = d[valid_mask]
            if pct_lo > 0.0 or pct_hi < 1.0:
                d_min = float(np.quantile(v, pct_lo))
                d_max = float(np.quantile(v, pct_hi))
            else:
                d_min = float(v.min())
                d_max = float(v.max())
            if d_max - d_min > 1e-6:
                norm = np.clip((d - d_min) / (d_max - d_min), 0.0, 1.0)
                if is_disparity:
                    scaled = norm * 255.0           # near (large) -> light
                else:
                    scaled = (1.0 - norm) * 255.0   # near (small) -> light
            else:
                scaled = np.full_like(d, 255.0)
            out[valid_mask] = np.clip(scaled, 0, 255).astype(np.uint8)[valid_mask]
        return np.stack([out] * 3, axis=-1)

    def _erode_non_sky(sky_mask_arg, shape_hw):
        """Return a (H, W) bool mask of non-sky pixels eroded by 5px.

        ``sky_mask_arg`` follows the dataset convention: 1 = non-sky/valid,
        0 = sky. Returns ``None`` if no mask is available.
        """
        if sky_mask_arg is None:
            return None
        if torch.is_tensor(sky_mask_arg):
            gm = sky_mask_arg.detach().float().cpu().numpy().squeeze()
        else:
            gm = np.asarray(sky_mask_arg).astype(np.float32).squeeze()
        if gm.shape != shape_hw:
            return None
        non_sky = (gm > 0.5).astype(np.float32)
        k = 5
        pad = k // 2
        padded = np.pad(non_sky, pad, mode="edge")
        eroded = np.ones_like(non_sky)
        for di in range(k):
            for dj in range(k):
                eroded = np.minimum(
                    eroded,
                    padded[di:di + non_sky.shape[0],
                           dj:dj + non_sky.shape[1]])
        return eroded > 0.5

    def _pred_depth_panel(depth_premul_t, acc_t, sky_mask_arg=None,
                          use_sky_mask: bool = True,
                          acc_cutoff: float = 5e-2):
        """Convert premul depth + acc to a disparity greyscale panel.

        Mirrors the convention in train_da2loss.py to suppress sky-area
        white patches caused by stray near-Gaussian leaks.
        """
        dp = depth_premul_t.detach().cpu().numpy().squeeze().astype(np.float32)
        a = acc_t.detach().cpu().numpy().squeeze().astype(np.float32)
        valid = (a > acc_cutoff) & np.isfinite(dp)
        if use_sky_mask:
            eroded = _erode_non_sky(sky_mask_arg, valid.shape)
            if eroded is not None:
                valid = valid & eroded
        # Expected depth = premul / acc on valid pixels.
        depth = np.zeros_like(dp)
        depth[valid] = dp[valid] / np.maximum(a[valid], 1e-10)
        # Disparity (1/d) so brightness = near.
        disp = np.zeros_like(depth)
        disp[valid] = 1.0 / np.maximum(depth[valid], 1e-6)
        return _depth_to_grey(disp, valid_mask=valid, is_disparity=True,
                              pct_lo=0.02, pct_hi=0.98)

    H = int(gt_on.shape[-2])
    W = int(gt_on.shape[-1])

    # ── Row 1: Ground truth ───────────────────────────────────────────
    gt_rgb = _to_rgb_uint8(gt_on[:3])

    # Mask: R=sky, G=moving, B=0
    mask_img = np.zeros((H, W, 3), dtype=np.uint8)
    if sky_mask is not None:
        s = sky_mask.detach().cpu().float().numpy().squeeze()
        mask_img[..., 0] = ((1.0 - np.clip(s, 0, 1)) * 255).astype(np.uint8)
    if moving_mask is not None:
        m = moving_mask.detach().cpu().float().numpy().squeeze()
        mask_img[..., 1] = (np.clip(m, 0, 1) * 255).astype(np.uint8)

    if mono_depth is not None:
        # DA-v2 outputs are disparity (larger = closer); flip the brightness
        # mapping so this panel matches the rendered-depth panels below
        # (near = light, far = dark). Use percentile bounds to suppress
        # outliers — same as the predicted depth panels.
        gt_depth_img = _depth_to_grey(mono_depth, is_disparity=True,
                                       pct_lo=0.02, pct_hi=0.98)
    else:
        gt_depth_img = np.zeros((H, W, 3), dtype=np.uint8)

    # ── Row 2: On-road prediction ─────────────────────────────────────
    # Use the GT sky mask to suppress white patches from stray near-Gaussians
    # leaking into sky tiles (consistent with train_da2loss.py).
    pred_rgb = _to_rgb_uint8(rendered_on)
    pred_alpha_img = _acc_to_grey(acc_on)
    pred_depth_img = _pred_depth_panel(depth_on, acc_on,
                                        sky_mask_arg=sky_mask,
                                        use_sky_mask=True)

    # ── Row 3: Jittered prediction ────────────────────────────────────
    # Off-road view doesn't have a pixel-aligned GT sky mask, but for the
    # small lateral shifts used here the GT sky mask is still a good
    # approximation of where the sky is. Re-using it (same as row 2)
    # together with a stricter acc cutoff suppresses the white patches
    # caused by stray near-Gaussians leaking into sky tiles.
    jit_rgb = _to_rgb_uint8(rendered_off)
    jit_alpha_img = _acc_to_grey(acc_off)
    jit_depth_img = _pred_depth_panel(depth_off, acc_off,
                                       sky_mask_arg=sky_mask,
                                       use_sky_mask=True,
                                       acc_cutoff=1e-1)

    # ── Compose grid ──────────────────────────────────────────────────
    row1 = np.concatenate([gt_rgb,   mask_img,       gt_depth_img],   axis=1)
    row2 = np.concatenate([pred_rgb, pred_alpha_img, pred_depth_img], axis=1)
    row3 = np.concatenate([jit_rgb,  jit_alpha_img,  jit_depth_img],  axis=1)
    grid = np.concatenate([row1, row2, row3], axis=0)

    os.makedirs(log_dir, exist_ok=True)
    _Image.fromarray(grid).save(
        os.path.join(log_dir, f"epoch_{epoch:04d}.png"))


def _tensor_to_video_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert a rendered RGB tensor [C,H,W] in [0,1] to a video frame."""
    arr = t.detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
    return (arr * 255).astype(np.uint8)


@torch.no_grad()
def _save_eval_jitter_videos(
    cam: Camera,
    gaussians: GaussianModel,
    bg_color: torch.Tensor,
    image_on: torch.Tensor,
    up_dir: np.ndarray,
    road_width: float,
    directions: list[str],
    forward_dir: np.ndarray | None,
    lateral_dir: np.ndarray | None,
    save_root: str,
    steps: int = 10,
    fps: int = 10,
) -> dict[str, dict]:
    """Save outward/return jitter videos for one eval camera.

    The return leg reuses the outward rendered viewpoints in reverse order.
    The final endpoint package for each direction is returned so the existing
    composite eval image can reuse it instead of rendering that endpoint again.
    """
    if steps <= 0 or road_width <= 0:
        return {}

    cam_dir = os.path.join(save_root, cam.image_name)
    os.makedirs(cam_dir, exist_ok=True)

    base_frame = _tensor_to_video_uint8(image_on)
    distances = [road_width * (i + 1) / steps for i in range(steps)]
    endpoint_pkgs: dict[str, dict] = {}
    outward_frames: dict[str, list[np.ndarray]] = {}

    for direction in directions:
        frames = []
        for step_idx, dist in enumerate(distances, start=1):
            jit_cam = build_jittered_camera(
                cam, up_dir, dist,
                lateral_sign=direction,
                forward=forward_dir,
                lateral=lateral_dir,
            )
            pkg = render(jit_cam, gaussians, bg_color)
            frames.append(_tensor_to_video_uint8(pkg["rgb"]))
            if step_idx == steps:
                endpoint_pkgs[direction] = pkg

        outward_frames[direction] = frames

        imageio.mimwrite(
            os.path.join(cam_dir, f"on_path_to_{direction}.mp4"),
            [base_frame] + frames,
            fps=fps,
        )
        imageio.mimwrite(
            os.path.join(cam_dir, f"{direction}_to_on_path.mp4"),
            list(reversed(frames)) + [base_frame],
            fps=fps,
        )

    # Combined loop for easy review on repeat:
    # on_path_to_right + right_to_on_path + on_path_to_left + left_to_on_path
    # (Skip duplicate boundary frames so playback is smooth.)
    if ("right" in outward_frames) and ("left" in outward_frames):
        right_out = outward_frames["right"]
        left_out = outward_frames["left"]

        def _return_leg(out_frames: list[np.ndarray]) -> list[np.ndarray]:
            # If out_frames = [f1, ..., fN], outward segment ends at fN.
            # Return segment should be [f(N-1), ..., f1, base] to avoid
            # duplicating fN at the join.
            if len(out_frames) <= 1:
                return [base_frame]
            return list(reversed(out_frames[:-1])) + [base_frame]

        combined_frames = (
            [base_frame] +
            right_out +
            _return_leg(right_out) +
            left_out +
            _return_leg(left_out)
        )

        imageio.mimwrite(
            os.path.join(cam_dir, f"{cam.image_name}_wobble.mp4"),
            combined_frames,
            fps=fps,
        )

    return endpoint_pkgs


# ═══════════════════════════════════════════════════════════════════════════
#  §0b  GAN-aware Evaluation (8-panel composite per test camera)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_gan(
    test_cameras, gaussians, bg_color, epoch,
    up_dir: np.ndarray, road_width: float, lateral_sign: float,
    forward_dir: np.ndarray | None = None,
    lateral_dir: np.ndarray | None = None,
    tb_writer=None,
    eval_csv_path: str | None = None, split: str = "test",
    n_points: int | None = None,
    save_dir: str | None = None,
    jitter_faces: tuple[str, ...] | None = None,
    jitter_directions: tuple[str, ...] | None = None,
):
    """Evaluate on test cameras with 3x3 composite images.

    Only cameras whose face is in ``jitter_faces`` are evaluated (when
    given), and one composite is saved per (camera, jitter direction)
    pair drawn from ``jitter_directions``.
    """
    if not test_cameras:
        return {}

    # Filter cameras by face
    if jitter_faces is not None:
        face_set = set(jitter_faces)
        eval_cameras = [c for c in test_cameras
                        if _parse_face(c.image_name)[1] in face_set]
    else:
        eval_cameras = list(test_cameras)

    # Resolve directions to render at eval
    if jitter_directions is not None and len(jitter_directions) > 0:
        directions = list(jitter_directions)
    elif lateral_sign is None:
        directions = ["left", "right"]
    elif isinstance(lateral_sign, str):
        directions = [lateral_sign]
    else:
        directions = ["left" if lateral_sign >= 0 else "right"]

    l1_list, psnr_list, ssim_list = [], [], []

    img_dir = None
    if save_dir is not None:
        img_dir = os.path.join(save_dir, split, f"epoch_{epoch}")
        os.makedirs(img_dir, exist_ok=True)
    video_dir = None
    if img_dir is not None:
        video_dir = os.path.join(img_dir, "jitter_videos")
        os.makedirs(video_dir, exist_ok=True)

    for cam in eval_cameras:
        gt = cam.original_image.cuda(non_blocking=True)
        sky_mask = cam.guidance.get("mask")
        if sky_mask is not None:
            sky_mask = sky_mask.cuda(non_blocking=True)
        moving_mask = cam.guidance.get("moving_mask")
        if moving_mask is not None:
            moving_mask = moving_mask.cuda(non_blocking=True)

        mask = sky_mask
        if mask is not None and moving_mask is not None:
            mask = mask & (~moving_mask)
        elif moving_mask is not None:
            mask = ~moving_mask

        # Render on-road (once per camera; metrics computed once)
        pkg_on = render(cam, gaussians, bg_color)
        image_on = pkg_on["rgb"]

        # Metrics (on-road)
        l1_list.append(l1_loss(image_on, gt, mask).item())
        psnr_list.append(psnr(image_on, gt, mask).item())
        ssim_list.append(ssim(image_on, gt, mask=mask).item())

        endpoint_pkgs = {}
        if video_dir is not None:
            endpoint_pkgs = _save_eval_jitter_videos(
                cam, gaussians, bg_color, image_on,
                up_dir=up_dir,
                road_width=road_width,
                directions=directions,
                forward_dir=forward_dir,
                lateral_dir=lateral_dir,
                save_root=video_dir,
                steps=10,
                fps=10,
            )

        # Render off-road for each requested direction and save composite
        for direction in directions:
            pkg_off = endpoint_pkgs.get(direction)
            if pkg_off is None:
                jit_cam = build_jittered_camera(
                    cam, up_dir, road_width,
                    lateral_sign=direction,
                    forward=forward_dir,
                    lateral=lateral_dir,
                )
                pkg_off = render(jit_cam, gaussians, bg_color)
            image_off = pkg_off["rgb"]

            if img_dir is not None:
                name = cam.image_name
                mono_depth = cam.guidance.get("mono_depth")
                save_log_images_gan(
                    img_dir, 0,
                    gt, image_on, pkg_on["depth"], pkg_on["acc"],
                    image_off, pkg_off["depth"], pkg_off["acc"],
                    sky_mask=sky_mask,
                    moving_mask=moving_mask,
                    mono_depth=mono_depth,
                )
                generic = os.path.join(img_dir, "epoch_0000.png")
                target = os.path.join(img_dir,
                                      f"{name}_jitter_{direction}.png")
                if os.path.isfile(generic):
                    os.rename(generic, target)

    metrics = {
        "l1_loss": np.mean(l1_list),
        "psnr": np.mean(psnr_list),
        "ssim": np.mean(ssim_list),
    }
    if n_points is not None:
        metrics["n_points"] = n_points

    print(f"  [EVAL {split} epoch {epoch}] "
          f"L1={metrics['l1_loss']:.4f}  "
          f"PSNR={metrics['psnr']:.2f}  "
          f"SSIM={metrics['ssim']:.4f}")

    if tb_writer is not None:
        for k, v in metrics.items():
            tb_writer.add_scalar(f"eval_{split}/{k}", v, epoch)

    if eval_csv_path is not None:
        file_exists = os.path.isfile(eval_csv_path)
        row = {"split": split, "epoch": epoch}
        row.update(metrics)
        fieldnames = ["split", "epoch", "l1_loss", "psnr", "ssim", "n_points"]
        with open(eval_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=fieldnames, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    return metrics


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
    lateral_sign: float | str | None = None,
    forward: np.ndarray | None = None,
    lateral: np.ndarray | None = None,
) -> Camera:
    """Build an off-road version of an on-road camera.

    Shifts the camera by ``road_width`` metres in a chosen direction.

    The shift basis is **trajectory-level** (consistent across all cubemap
    faces) when ``forward`` and ``lateral`` are provided, so a ``left``
    shift on a ``_back``-facing cubemap camera moves the same way in
    world space as a ``left`` shift on a ``_front``-facing one.
    If they are not provided, a per-camera basis derived from the camera's
    own forward axis is used (legacy behaviour).

    lateral_sign : one of "left", "right", "up", "front", "back" (string),
                   +1.0 (left) / -1.0 (right) (float),
                   or None = randomly picks a direction.
    """
    R_c2w = cam.R                # (3,3) numpy
    T_w2c = cam.T                # (3,)  numpy

    # Camera forward direction in world coords (camera -Z axis)
    cam_forward = R_c2w @ np.array([0.0, 0.0, -1.0])

    # Pick the shift basis.  Trajectory-level basis keeps the meaning of
    # "left"/"right"/"front"/"back" consistent across cubemap faces.
    if forward is not None and lateral is not None:
        ref_forward = np.asarray(forward, dtype=np.float64)
        ref_forward = ref_forward / (np.linalg.norm(ref_forward) + 1e-12)
        ref_lateral = np.asarray(lateral, dtype=np.float64)
        ref_lateral = ref_lateral / (np.linalg.norm(ref_lateral) + 1e-12)
    else:
        ref_forward = cam_forward / (np.linalg.norm(cam_forward) + 1e-12)
        ref_lateral = np.cross(ref_forward, up)
        ref_lateral = ref_lateral / (np.linalg.norm(ref_lateral) + 1e-12)

    # Resolve the actual direction string
    if isinstance(lateral_sign, str):
        direction = lateral_sign
    elif lateral_sign is None:
        direction = random_choice(["left", "right", "up", "front", "back"])
    else:
        direction = "left" if lateral_sign >= 0 else "right"

    if direction == "left":
        shift_vec = ref_lateral
    elif direction == "right":
        shift_vec = -ref_lateral
    elif direction == "up":
        shift_vec = up / (np.linalg.norm(up) + 1e-12)
    elif direction == "front":
        shift_vec = ref_forward
    else:  # back
        shift_vec = -ref_forward

    # Old camera centre in world coords
    C_old = -R_c2w @ T_w2c

    # Shift
    C_new = C_old + road_width * shift_vec

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
        image_name=f"jitter_{direction}_{cam.image_name}",
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
                 lateral_sign: float,
                 pretrained_epoch: int | None, gan_epochs: int,
                 critic_iters: int, lambda_gp: float,
                 lambda_recon: float, lambda_dssim: float,
                 lr_critic: float, lr_generator: float,
                 road_width_warmup_epochs: int,
                 road_width_init_frac: float,
                 jitter_faces: tuple[str, ...] = ("front", "back"),
                 jitter_directions_default: tuple[str, ...] = (
                     "left", "right")):
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

    # Explicitly initialise the CUDA context before any .backward() call
    # to avoid "no current CUDA context" cuBLAS warnings.
    torch.cuda.set_device(0)
    torch.cuda.init()

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
        depth_dir=data_cfg.get("depth_dir", ""),
        split_test=data_cfg.get("split_test", 8),
        workspace=workspace,
        max_frames=data_cfg.get("max_frames", 0),
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

    # ── Filter training cameras to the faces we actually jitter ───────
    jitter_faces_set = set(jitter_faces)

    def _camera_face(c) -> str:
        return _parse_face(c.image_name)[1]

    gan_train_cameras = [c for c in train_cameras
                         if _camera_face(c) in jitter_faces_set]
    n_total = len(train_cameras)
    n_kept = len(gan_train_cameras)
    print(f"GAN training will use {n_kept}/{n_total} cameras whose face "
          f"is in {sorted(jitter_faces_set)} "
          f"(skipping {n_total - n_kept} side faces).")
    if not gan_train_cameras:
        sys.exit("ERROR: No training cameras match --jitter_faces; aborting.")

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
    print(f"  lateral : {lateral_dir}  (road_width = {road_width:.2f} m, "
          f"lateral_sign = {lateral_sign})")

    # ── Load pre-trained Gaussians ────────────────────────────────────
    model_path = os.path.join(
        workspace, model_root, cfg["task"], cfg["exp_name"])
    trained_model_dir = os.path.join(model_path, "trained_model")

    sh_degree = cfg.get("model", {}).get("sh_degree",
                cfg.get("model", {}).get("gaussian", {}).get("sh_degree", 3))
    gaussians, loaded_epoch, ckpt_state = _load_checkpoint(
        trained_model_dir, pretrained_epoch, sh_degree)

    print(f"Loaded pre-trained model from epoch {loaded_epoch}")

    # ── Copy input.ply from pre-trained model to GAN output ───────────
    src_input_ply = os.path.join(model_path, "input.ply")
    dst_input_ply = os.path.join(dirs["model_path"], "input.ply")
    if os.path.isfile(src_input_ply):
        shutil.copy2(src_input_ply, dst_input_ply)
        print(f"Copied input.ply -> {dst_input_ply}")
    else:
        print(f"Warning: {src_input_ply} not found, skipping input.ply copy")

    # ── Setup Gaussian optimizer for fine-tuning ──────────────────────
    # Use lower learning rates for fine-tuning
    ft_optim_cfg = copy.deepcopy(optim_cfg)
    ft_optim_cfg["position_lr_init"] = lr_generator
    ft_optim_cfg["position_lr_final"] = lr_generator * 0.1
    ft_optim_cfg["position_lr_max_epochs"] = gan_epochs
    ft_optim_cfg["position_lr_max_steps"] = (
        gan_epochs * len(train_cameras))
    ft_optim_cfg["feature_lr"] = lr_generator * 5.0
    ft_optim_cfg["opacity_lr"] = lr_generator * 5.0
    ft_optim_cfg["scaling_lr"] = lr_generator * 2.0
    ft_optim_cfg["rotation_lr"] = lr_generator * 1.0

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

    # ── Critic warm-up ──────────────────────────────────────────────
    # Train the critic alone for a few epochs so it provides a
    # meaningful gradient signal before the generator starts updating.
    # Use the initial (small) road_width so the critic doesn't see
    # extreme out-of-distribution renders from the very first step.
    warmup_epochs = 10
    warmup_road_width = road_width * road_width_init_frac
    print(f"\nCritic warm-up: {warmup_epochs} epochs "
          f"(generator frozen, road_width={warmup_road_width:.3f} m) ...")
    for wu_ep in range(1, warmup_epochs + 1):
        wu_stack = list(gan_train_cameras)
        shuffle(wu_stack)
        wu_loss = 0.0
        wu_w_dist = 0.0
        _wu_directions = (
            list(jitter_directions_default)
            if lateral_sign is None
            else [lateral_sign if isinstance(lateral_sign, str)
                  else ("left" if lateral_sign >= 0 else "right")]
        )
        for wi, cam in enumerate(wu_stack):
            for direction in _wu_directions:
                jit_cam = build_jittered_camera(cam, up_dir,
                                                 warmup_road_width,
                                                 lateral_sign=direction,
                                                 forward=forward_dir,
                                                 lateral=lateral_dir)
                with torch.no_grad():
                    real_img = render(cam, gaussians, bg_color)["rgb"].detach()
                    fake_img = render(jit_cam, gaussians, bg_color)["rgb"].detach()
                real_score = critic(real_img.unsqueeze(0))
                fake_score = critic(fake_img.unsqueeze(0))
                loss_c = fake_score.mean() - real_score.mean()
                gp = gradient_penalty(critic, real_img.unsqueeze(0),
                                      fake_img.unsqueeze(0), real_img.device)
                loss_c = loss_c + lambda_gp * gp
                critic_optimizer.zero_grad()
                loss_c.backward()
                critic_optimizer.step()
                wu_loss += loss_c.item()
                wu_w_dist += (real_score.mean() - fake_score.mean()).item()
            if wi % max(1, len(wu_stack) // 4) == 0:
                print(f"    warm-up {wu_ep}/{warmup_epochs} "
                      f"[{wi+1}/{len(wu_stack)}] "
                      f"loss={loss_c.item():.4f}  "
                      f"real={real_score.mean().item():.4f}  "
                      f"fake={fake_score.mean().item():.4f}  "
                      f"GP={gp.item():.4f}")
        n_wu = len(wu_stack) * len(_wu_directions)
        print(f"  warm-up epoch {wu_ep}/{warmup_epochs}  "
              f"critic_loss={wu_loss / n_wu:.4f}  "
              f"W_dist={wu_w_dist / n_wu:.4f}")

    # ── Training ──────────────────────────────────────────────────────
    print(f"\nGAN fine-tuning: {gan_epochs} epochs × "
          f"{len(gan_train_cameras)} cameras/epoch")
    print(f"  jitter_faces      = {sorted(jitter_faces_set)}")
    print(f"  jitter_directions = {list(jitter_directions_default)}")
    print(f"  road_width        = {road_width} m  "
          f"(curriculum: start {road_width * road_width_init_frac:.3f}, "
          f"ramp over {road_width_warmup_epochs} epochs)")
    print(f"  critic_iters = {critic_iters}")
    print(f"  lambda_gp    = {lambda_gp}")
    print(f"  lambda_recon = {lambda_recon}")
    print(f"  lambda_dssim = {lambda_dssim}")
    print(f"  lr_critic    = {lr_critic}")
    print(f"  lr_generator = {lr_generator}")

    log_every_n_steps = max(1, len(gan_train_cameras) // 6)  # ~6 logs per epoch

    step = 0
    progress = tqdm(range(gan_epochs), desc="GAN Epochs", unit="ep")

    for epoch in range(1, gan_epochs + 1):
        viewpoint_stack = list(gan_train_cameras)
        shuffle(viewpoint_stack)
        n_cameras = len(viewpoint_stack)

        # ── Curriculum: gradually grow road_width from init_frac → 1.0 ──
        ramp_t = min(1.0, epoch / max(1, road_width_warmup_epochs))
        cur_road_width = road_width * (
            road_width_init_frac
            + (1.0 - road_width_init_frac) * ramp_t)
        if epoch == 1 or (epoch - 1) % 10 == 0:
            tqdm.write(
                f"  [E{epoch}] road_width curriculum: "
                f"{cur_road_width:.3f} m "
                f"({cur_road_width / road_width * 100:.0f}% of target)")

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
        log_sky_mask = log_moving_mask = log_mono_depth = None

        _jitter_directions = (
            list(jitter_directions_default)
            if lateral_sign is None
            else [lateral_sign if isinstance(lateral_sign, str)
                  else ("left" if lateral_sign >= 0 else "right")]
        )

        for cam_idx, cam in enumerate(viewpoint_stack):

            # ── Load masks (once per camera) ──────────────────────
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

            for dir_idx, direction in enumerate(_jitter_directions):
                step += 1

                # ── Build jittered camera for this direction ───────
                jit_cam = build_jittered_camera(
                    cam, up_dir, cur_road_width,
                    lateral_sign=direction,
                    forward=forward_dir,
                    lateral=lateral_dir,
                )

                # global step index across all cameras × directions
                global_step_idx = cam_idx * len(_jitter_directions) + dir_idx

                # ==================================================
                #  CRITIC STEP  (train critic, freeze Gaussians)
                # ==================================================
                if global_step_idx % (critic_iters + 1) < critic_iters:
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

                    if cam_idx % log_every_n_steps == 0 and dir_idx == 0:
                        tqdm.write(
                            f"  [E{epoch} C {cam_idx+1}/{n_cameras} {direction}] "
                            f"critic_loss={loss_c.item():.4f}  "
                            f"W_dist={w_dist.item():.4f}  "
                            f"real_score={real_score.mean().item():.4f}  "
                            f"fake_score={fake_score.mean().item():.4f}  "
                            f"GP={gp.item():.4f}")

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
                    # Clamp to prevent large adversarial gradients
                    loss_adv = loss_adv.clamp(-50.0, 50.0)

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

                    # --- Monocular depth supervision (on-road view) ---
                    lambda_depth = optim_cfg.get("lambda_depth", 0.0)
                    depth_loss = torch.tensor(0.0, device="cuda")
                    if lambda_depth > 0 and "mono_depth" in cam.guidance:
                        mono_depth = cam.guidance["mono_depth"]
                        mono_depth = (mono_depth.cuda(non_blocking=True)
                                      if not mono_depth.is_cuda else mono_depth)
                        depth_loss = lambda_depth * depth_mono_loss(
                            real_pkg["depth"], mono_depth,
                            real_pkg["acc"], mask)

                    # --- Total generator loss ---
                    loss_g = (loss_adv +
                              lambda_recon * loss_recon +
                              sh_reg +
                              sky_loss +
                              depth_loss)

                    loss_g.backward()

                    # Clip Gaussian gradients to prevent destructive updates
                    all_params = []
                    for pg in gaussians.optimizer.param_groups:
                        all_params.extend(pg["params"])
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)

                    gaussians.update_optimizer()

                    ep_gen_loss += loss_g.item()
                    ep_recon_loss += loss_recon.item()

                    # Logging metrics (detached)
                    with torch.no_grad():
                        cur_psnr_on = psnr(real_img, gt_image, mask).item()
                        cur_psnr_off = psnr(fake_img, gt_image, mask).item()
                        ep_psnr_on += cur_psnr_on
                        ep_psnr_off += cur_psnr_off

                    if cam_idx % log_every_n_steps == 0 and dir_idx == 0:
                        tqdm.write(
                            f"  [E{epoch} G {cam_idx+1}/{n_cameras} {direction}] "
                            f"loss_g={loss_g.item():.4f}  "
                            f"adv={loss_adv.item():.4f}  "
                            f"recon={loss_recon.item():.4f} "
                            f"(L1={Ll1.item():.4f})  "
                            f"sh_reg={sh_reg.item():.6f}  "
                            f"sky={sky_loss.item():.6f}  "
                            f"depth={depth_loss.item():.6f}  "
                            f"PSNR_on={cur_psnr_on:.2f}  "
                            f"PSNR_off={cur_psnr_off:.2f}")

                    # Stash for end-of-epoch log images (last direction)
                    if dir_idx == len(_jitter_directions) - 1:
                        log_gt_on = gt_image.detach()
                        log_render_on = real_img.detach()
                        log_depth_on = real_pkg["depth"].detach()
                        log_acc_on = real_pkg["acc"].detach()
                        log_render_off = fake_img.detach()
                        log_depth_off = fake_pkg["depth"].detach()
                        log_acc_off = fake_pkg["acc"].detach()
                        log_sky_mask = sky_mask
                        log_moving_mask = moving_mask
                        log_mono_depth = cam.guidance.get("mono_depth")

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

        # ── Detailed epoch summary ────────────────────────────────
        with torch.no_grad():
            xyz = gaussians.get_xyz
            opac = gaussians.get_opacity
            scale = gaussians.get_scaling
            tqdm.write(
                f"\n{'─'*72}\n"
                f"  EPOCH {epoch}/{gan_epochs} SUMMARY\n"
                f"  critic_steps={n_crit_steps}  gen_steps={n_gen_steps}\n"
                f"  Loss  │ critic={avg_c:.4f}  gen={avg_g:.4f}  "
                f"recon={avg_r:.4f}  W_dist={avg_w:.4f}\n"
                f"  PSNR  │ on-road={avg_psnr_on:.2f}  "
                f"off-road={avg_psnr_off:.2f}\n"
                f"  Gauss │ N={gaussians.num_points:,}  "
                f"xyz=[{xyz.min().item():.2f}, {xyz.max().item():.2f}]  "
                f"opacity=[{opac.min().item():.3f}, {opac.max().item():.3f}] "
                f"mean={opac.mean().item():.3f}  "
                f"scale=[{scale.min().item():.4f}, {scale.max().item():.4f}] "
                f"mean={scale.mean().item():.4f}\n"
                f"{'─'*72}")

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
                    mono_depth=log_mono_depth,
                )
            except Exception:
                pass

        # ── Evaluation every 10 epochs ────────────────────────────
        if epoch % 10 == 0 or epoch == gan_epochs:
            with torch.no_grad():
                evaluate_gan(test_cameras, gaussians, bg_color,
                         epoch,
                         up_dir=up_dir,
                         road_width=road_width,
                         lateral_sign=lateral_sign,
                         forward_dir=forward_dir,
                         lateral_dir=lateral_dir,
                         tb_writer=tb_writer,
                         eval_csv_path=dirs["eval_csv_path"],
                         split="test",
                         n_points=gaussians.num_points,
                         save_dir=dirs["model_path"],
                         jitter_faces=tuple(jitter_faces_set),
                         jitter_directions=tuple(
                             jitter_directions_default))

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
        "--lateral_sign", type=float, default=None,
        help="+1 shift left, -1 shift right, omit to randomly choose left/right/up per camera.")
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
    parser.add_argument(
        "--road_width_warmup_epochs", type=int, default=50,
        help="Epochs over which the perturbation magnitude ramps from "
             "--road_width_init_frac × road_width up to road_width "
             "(default: 50).")
    parser.add_argument(
        "--road_width_init_frac", type=float, default=0.1,
        help="Initial perturbation fraction at epoch 1 (default: 0.1, i.e. "
             "start at 10%% of road_width).")
    parser.add_argument(
        "--jitter_faces", type=str, nargs="+",
        default=["front", "back"],
        help="Cubemap faces that get jittered during GAN training. Other "
             "faces are skipped to accelerate training (default: front back).")
    parser.add_argument(
        "--jitter_directions", type=str, nargs="+",
        default=["left", "right", "up"],
        choices=["left", "right", "up", "front", "back"],
        help="Directions to jitter the camera in (used when --lateral_sign "
             "is omitted). Default: left right up.")
    parser.add_argument(
        "--output_dir", type=str, default="output",
        help="Root output directory (default: output).")
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

    cfg["output_root"] = args.output_dir

    training_gan(
        cfg,
        model_root=args.model_root,
        road_width=args.road_width,
        lateral_sign=args.lateral_sign,
        pretrained_epoch=args.epoch,
        gan_epochs=args.gan_epochs,
        critic_iters=args.critic_iters,
        lambda_gp=args.lambda_gp,
        lambda_recon=args.lambda_recon,
        lambda_dssim=args.lambda_dssim,
        lr_critic=args.lr_critic,
        lr_generator=args.lr_generator,
        road_width_warmup_epochs=args.road_width_warmup_epochs,
        road_width_init_frac=args.road_width_init_frac,
        jitter_faces=tuple(args.jitter_faces),
        jitter_directions_default=tuple(args.jitter_directions),
    )


if __name__ == "__main__":
    main()
