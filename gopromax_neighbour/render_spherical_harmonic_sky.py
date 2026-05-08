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
sys.path.insert(0, str(_SCRIPT_DIR.parent))
sys.path.insert(0, str(_SCRIPT_DIR))

from train_da2loss import (                                              # noqa: E402
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
#  Spherical Harmonics Sky Model
# ═══════════════════════════════════════════════════════════════════════════

class SphericalHarmonicSky(torch.nn.Module):
    """A sky model represented by Spherical Harmonics."""

    def __init__(self, sh_degree: int, sh_coeffs: list[float] | None = None):
        super().__init__()
        self.sh_degree = sh_degree
        # Make coefficients learnable so we can fit them to GT sky pixels.
        self.sh_coeffs = torch.nn.Parameter(
            self._init_sh_coeffs(sh_coeffs), requires_grad=True)

    def _init_sh_coeffs(self, coeffs: list[float] | None) -> torch.Tensor:
        """Initialize SH coefficients."""
        if coeffs is not None and len(coeffs) > 0:
            # Accept any length of coefficients
            # Reshape assuming coefficients are interleaved as [R0, R1, R2, ..., G0, G1, G2, ..., B0, B1, B2, ...]
            coeffs_tensor = torch.tensor(coeffs, dtype=torch.float32)
            n_coeffs_per_channel = len(coeffs) // 3
            if len(coeffs) % 3 == 0:
                return coeffs_tensor.reshape(3, n_coeffs_per_channel)
            else:
                # If not evenly divisible, just reshape as-is and warn
                print(f"Warning: SH coefficients length {len(coeffs)} not divisible by 3 (RGB). "
                      f"Padding to nearest multiple of 3.")
                # Pad to multiple of 3
                pad_size = ((len(coeffs) + 2) // 3) * 3 - len(coeffs)
                coeffs_padded = torch.cat([coeffs_tensor, torch.zeros(pad_size)])
                return coeffs_padded.reshape(3, -1)
        
        # Default: bright top, darker bottom
        sh_coeffs = torch.zeros(3, (self.sh_degree + 1) ** 2)
        sh_coeffs[:, 0] = 0.8  # L0,0 (ambient)
        if self.sh_degree > 0:
            sh_coeffs[:, 1] = 0.2  # L1,-1
            sh_coeffs[:, 2] = 0.2  # L1,0
            sh_coeffs[:, 3] = 0.2  # L1,1
        return sh_coeffs

    def forward(self, dirs: torch.Tensor) -> torch.Tensor:
        """Evaluate sky color for given directions.
        
        Args:
            dirs: direction vectors of shape (N, 3), normalized
        
        Returns:
            sky colors of shape (N, 3)
        """
        N = dirs.shape[0]
        device = dirs.device
        dtype = dirs.dtype
        
        # Extract direction components: shape (N,)
        x = dirs[:, 0]
        y = dirs[:, 1]
        z = dirs[:, 2]
        
        # SH evaluation constants
        C0 = 0.28209479177387814
        C1 = 0.4886025119029199
        
        # Initialize result: shape (3, N) for RGB channels × N pixels
        result = torch.zeros(3, N, device=device, dtype=dtype)
        
        # Get number of coefficients per channel
        n_coeffs = self.sh_coeffs.shape[1]
        
        # L0,0 term: (3, 1) broadcasts to (3, N)
        if n_coeffs > 0:
            result = result + C0 * self.sh_coeffs[:, 0:1]
        
        # L1 terms (if we have >= 4 coefficients and degree >= 1)
        if self.sh_degree > 0 and n_coeffs >= 4:
            # Each term: (3, 1) * (1, N) = (3, N)
            result = result + C1 * self.sh_coeffs[:, 1:2] * (-y.unsqueeze(0))
            result = result + C1 * self.sh_coeffs[:, 2:3] * z.unsqueeze(0)
            result = result + C1 * self.sh_coeffs[:, 3:4] * (-x.unsqueeze(0))
        
        # L2 terms (if we have >= 9 coefficients and degree >= 2)
        if self.sh_degree > 1 and n_coeffs >= 9:
            C2_0 = 1.0925484305920792
            C2_1 = -1.0925484305920792
            C2_2 = 0.31539156525252005
            C2_3 = -1.0925484305920792
            C2_4 = 0.5462742152960396
            
            xx = x * x
            yy = y * y
            zz = z * z
            xy = x * y
            yz = y * z
            xz = x * z
            
            result = result + C2_0 * self.sh_coeffs[:, 4:5] * (xy.unsqueeze(0))
            result = result + C2_1 * self.sh_coeffs[:, 5:6] * (yz.unsqueeze(0))
            result = result + C2_2 * self.sh_coeffs[:, 6:7] * ((2 * zz - xx - yy).unsqueeze(0))
            result = result + C2_3 * self.sh_coeffs[:, 7:8] * (xz.unsqueeze(0))
            result = result + C2_4 * self.sh_coeffs[:, 8:9] * ((xx - yy).unsqueeze(0))
        
        # Transpose from (3, N) to (N, 3)
        return result.t()

# ══════════════════════════════════════════════════════════════════════════
#  Sky compositing helpers
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _camera_view_dirs(camera) -> torch.Tensor:
    """Per-pixel world-space view directions, shape (H, W, 3)."""
    W = int(camera.image_width)
    H = int(camera.image_height)
    yy, xx = torch.meshgrid(
        torch.arange(H, device="cuda", dtype=torch.float32) + 0.5,
        torch.arange(W, device="cuda", dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    dirs = torch.stack(
        [(xx - W / 2) / camera.K[0, 0],
         (yy - H / 2) / camera.K[1, 1],
         torch.ones_like(xx)], dim=-1)
    dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True)
    R_t = torch.from_numpy(camera.R).float().to("cuda")
    # Same convention as the rest of the codebase (R.T @ d_cam).
    dirs = (R_t.T @ dirs.reshape(-1, 3).T).T.reshape(H, W, 3)
    return dirs


def _compute_sky_bg(camera, sky_model: "SphericalHarmonicSky") -> torch.Tensor:
    """Evaluate the SH sky for every pixel of `camera`. Returns (3, H, W)."""
    dirs = _camera_view_dirs(camera)
    H, W = dirs.shape[:2]
    bg = sky_model(dirs.reshape(-1, 3)).reshape(H, W, 3).permute(2, 0, 1)
    return bg


def _composite_sky(rgb: torch.Tensor, acc: torch.Tensor,
                   sky_bg: torch.Tensor) -> torch.Tensor:
    """Alpha-composite the sky behind the rendered Gaussians.

    rgb    : (3, H, W) Gaussian render against a black background
    acc    : (1, H, W) accumulated Gaussian opacity
    sky_bg : (3, H, W) per-pixel sky color
    """
    a = acc[:1] if acc.dim() == 3 else acc.unsqueeze(0)  # (1, H, W)
    return rgb + (1.0 - a) * sky_bg


def _fit_sky_model(sky_model: "SphericalHarmonicSky",
                   cameras: list,
                   n_iters: int = 300,
                   lr: float = 1e-2,
                   samples_per_cam: int = 4096) -> None:
    """Fit SH coefficients to ground-truth sky pixels.

    Sky pixels are where the foreground mask is 0/False (the same pixels that
    were excluded during Gaussian training).
    """
    # Collect (direction, gt_rgb) pairs once.
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        for cam in cameras:
            mask = cam.guidance.get("mask")
            if mask is None:
                continue
            m = mask.to("cuda")
            if m.dim() == 3:
                m = m[0]
            m_bool = m.bool()
            sky_pix = ~m_bool                       # True where sky
            if not sky_pix.any():
                continue
            dirs = _camera_view_dirs(cam)           # (H, W, 3)
            gt = cam.original_image[:3].to("cuda")  # (3, H, W)
            pairs.append((dirs[sky_pix].contiguous(),
                          gt.permute(1, 2, 0)[sky_pix].contiguous()))

    if not pairs:
        print("  No sky pixels found in any camera – skipping SH fit.")
        sky_model.sh_coeffs.requires_grad_(False)
        return

    total_pix = sum(p[0].shape[0] for p in pairs)
    print(f"  Fitting sky SH on {total_pix:,} sky pixels "
          f"from {len(pairs)} cameras…")

    sky_model.sh_coeffs.requires_grad_(True)
    optimizer = torch.optim.Adam([sky_model.sh_coeffs], lr=lr)

    last_loss = float("nan")
    for _ in tqdm(range(n_iters), desc="Fitting sky SH"):
        optimizer.zero_grad()
        loss = 0.0
        for dirs_i, gt_i in pairs:
            n = dirs_i.shape[0]
            if n > samples_per_cam:
                sel = torch.randint(0, n, (samples_per_cam,), device="cuda")
                d, g = dirs_i[sel], gt_i[sel]
            else:
                d, g = dirs_i, gt_i
            pred = sky_model(d)
            loss = loss + torch.nn.functional.l1_loss(pred, g)
        loss = loss / len(pairs)
        loss.backward()
        optimizer.step()
        last_loss = loss.item()

    print(f"  Final sky-fit L1: {last_loss:.4f}")
    print(f"  Fitted SH coefficients (R, G, B):\n{sky_model.sh_coeffs.detach().cpu().numpy()}")
    sky_model.sh_coeffs.requires_grad_(False)

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
                     sh_degree: int) -> tuple[GaussianModel, int, dict | None]:
    """Load a GaussianModel from a checkpoint.

    Returns (gaussians, loaded_epoch, sky_model_state_or_None).
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
    sky_state = state.get("sky_model_state", None)
    if sky_state is not None:
        print("  Co-trained sky model found in checkpoint.")
    else:
        print("  No sky model in checkpoint – will fit from GT sky pixels.")
    print(f"  Loaded epoch {loaded_epoch}, "
          f"{gaussians.num_points:,} Gaussians, "
          f"SH degree {gaussians.active_sh_degree}")
    return gaussians, loaded_epoch, sky_state


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluate mode – per-image rendering for train / test sets
# ═══════════════════════════════════════════════════════════════════════════

def render_sets(cfg: dict, config_path: str, epoch: int | None = None,
                skip_train: bool = False, skip_test: bool = False,
                sky_sh_coeffs: list[float] | None = None,
                output_dir: str | None = None,
                fit_sky: bool = True,
                fit_iters: int = 300,
                fit_lr: float = 1e-2):
    """Render and save per-image outputs for train and test camera sets."""
    workspace = os.getcwd()
    data_cfg = cfg["data"]

    white_bg = data_cfg.get("white_background", False)
    bg_color = torch.tensor(
        [1, 1, 1] if white_bg else [0, 0, 0],
        dtype=torch.float32, device="cuda")

    sky_model = None
    if sky_sh_coeffs is not None or fit_sky:
        sh_degree = cfg.get("model", {}).get("sh_degree", 3)
        sky_model = SphericalHarmonicSky(sh_degree, sky_sh_coeffs).to("cuda")

    # Determine trained model directory from config file location
    config_dir = os.path.dirname(os.path.abspath(config_path))
    trained_model_dir = os.path.join(config_dir, "trained_model")

    # Determine save directory
    if output_dir is not None:
        save_path_base = os.path.join(workspace, output_dir)
    else:
        save_path_base = os.path.join(config_dir, "render_with_sh_sky")

    # NOTE: torch.no_grad() is moved INSIDE so the optional SH-fitting step
    # below can use autograd while the actual rendering stays grad-free.
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
    gaussians, loaded_epoch, sky_ckpt_state = _load_checkpoint(
        trained_model_dir, epoch, sh_degree)

    # Build camera sets
    splits = []
    print("Loading training cameras …")
    train_cameras = [load_camera(ci) for ci in tqdm(scene_info.train_cameras)]
    if not skip_train:
        splits.append(("train", train_cameras))
    if not skip_test:
        print("Loading test cameras …")
        test_cameras = [
            load_camera(ci) for ci in tqdm(scene_info.test_cameras)]
        splits.append(("test", test_cameras))

    # Use co-trained sky model from checkpoint if available;
    # otherwise fit from GT sky pixels (old behaviour).
    if sky_model is not None:
        if sky_ckpt_state is not None and sky_sh_coeffs is None:
            sky_model.load_state_dict(sky_ckpt_state)
            sky_model.eval()
            print("  Using co-trained sky model from checkpoint.")
        elif fit_sky and sky_sh_coeffs is None:
            print("Fitting sky SH coefficients to GT sky pixels…")
            _fit_sky_model(sky_model, train_cameras,
                           n_iters=fit_iters, lr=fit_lr)

    with torch.no_grad():
        times: list[float] = []

        for split_name, cameras in splits:
            # Limit to max_frames if specified
            # max_frames represents camera positions; multiply by 4 for 4 directions
            max_frames = data_cfg.get("max_frames", None)
            if max_frames is not None:
                cameras = cameras[:max_frames * 4]
                print(f"Limited {split_name} to first {max_frames} positions ({len(cameras)} frames)")
            
            save_dir = os.path.join(
                save_path_base, split_name,
                f"ours_epoch_{loaded_epoch}",
            )
            os.makedirs(save_dir, exist_ok=True)

            for idx, camera in enumerate(tqdm(cameras,
                                              desc=f"Rendering {split_name}")):
                torch.cuda.synchronize()
                t0 = time.time()

                # Always render against a constant black background, then
                # composite the SH sky behind the Gaussians using the
                # accumulated alpha returned by the rasterizer.
                result = render(camera, gaussians, bg_color)
                if sky_model is not None:
                    sky_bg = _compute_sky_bg(camera, sky_model)
                    result['rgb'] = _composite_sky(
                        result['rgb'], result['acc'], sky_bg)

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
                acc_d = result['acc'].clamp(min=1e-6)
                depth_t = result['depth_premul'] / acc_d
                depth = depth_t.detach().permute(1, 2, 0).cpu().numpy()
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

def render_trajectory(cfg: dict, config_path: str, epoch: int | None = None, fps: int = 10,
                      sky_sh_coeffs: list[float] | None = None,
                      output_dir: str | None = None,
                      fit_sky: bool = True,
                      fit_iters: int = 300,
                      fit_lr: float = 1e-2):
    """Render all frames in order and produce trajectory videos."""
    workspace = os.getcwd()
    data_cfg = cfg["data"]

    white_bg = data_cfg.get("white_background", False)
    bg_color = torch.tensor(
        [1, 1, 1] if white_bg else [0, 0, 0],
        dtype=torch.float32, device="cuda")

    sky_model = None
    if sky_sh_coeffs is not None or fit_sky:
        sh_degree = cfg.get("model", {}).get("sh_degree", 3)
        sky_model = SphericalHarmonicSky(sh_degree, sky_sh_coeffs).to("cuda")

    # Determine trained model directory from config file location
    config_dir = os.path.dirname(os.path.abspath(config_path))
    trained_model_dir = os.path.join(config_dir, "trained_model")

    # Determine save directory
    if output_dir is not None:
        save_path_base = os.path.join(workspace, output_dir)
    else:
        save_path_base = os.path.join(config_dir, "render_with_sh_sky")

    # Load scene data (outside no_grad so SH-fitting can use autograd).
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
    gaussians, loaded_epoch, sky_ckpt_state = _load_checkpoint(
        trained_model_dir, epoch, sh_degree)

    save_dir = os.path.join(
        save_path_base, 'trajectory',
        f"ours_epoch_{loaded_epoch}",
    )
    os.makedirs(save_dir, exist_ok=True)

    # Build all cameras (train + test), sorted by uid
    print("Loading all cameras …")
    all_cam_infos = scene_info.train_cameras + scene_info.test_cameras
    all_cameras = [load_camera(ci) for ci in tqdm(all_cam_infos)]
    all_cameras = sorted(all_cameras, key=lambda c: c.id)

    # Use co-trained sky model from checkpoint if available;
    # otherwise fit from GT sky pixels (old behaviour).
    if sky_model is not None:
        if sky_ckpt_state is not None and sky_sh_coeffs is None:
            sky_model.load_state_dict(sky_ckpt_state)
            sky_model.eval()
            print("  Using co-trained sky model from checkpoint.")
        elif fit_sky and sky_sh_coeffs is None:
            print("Fitting sky SH coefficients to GT sky pixels…")
            train_cameras = [load_camera(ci) for ci in scene_info.train_cameras]
            _fit_sky_model(sky_model, train_cameras,
                           n_iters=fit_iters, lr=fit_lr)

    with torch.no_grad():
        # Limit to max_frames if specified
        # max_frames represents camera positions; multiply by 4 for 4 directions
        max_frames = data_cfg.get("max_frames", None)
        if max_frames is not None:
            all_cameras = all_cameras[:max_frames * 4]
            print(f"Limited to first {max_frames} positions ({len(all_cameras)} frames)")

        rgbs_gt, rgbs = [], []
        depths, diffs = [], []
        cams_list = []

        for idx, camera in enumerate(tqdm(all_cameras,
                                          desc="Rendering Trajectory")):
            # Render against constant black bg, then composite the SH sky
            # behind the Gaussians using the accumulated alpha.
            result = render(camera, gaussians, bg_color)
            if sky_model is not None:
                sky_bg = _compute_sky_bg(camera, sky_model)
                result['rgb'] = _composite_sky(
                    result['rgb'], result['acc'], sky_bg)

            # Extract camera direction from image name
            name = camera.image_name
            image_name_lower = name.lower()
            if 'front' in image_name_lower:
                cam_id = 0
            elif 'right' in image_name_lower:
                cam_id = 1
            elif 'back' in image_name_lower:
                cam_id = 2
            elif 'left' in image_name_lower:
                cam_id = 3
            else:
                # Fallback: try to get from camera metadata or default to cycling
                cam_id = camera.meta.get('cam', idx % 4) if hasattr(camera, 'meta') else idx % 4
            cams_list.append(cam_id)

            # Accumulate frames
            rgbs_gt.append(_tensor_to_uint8(camera.original_image[:3]))
            rgbs.append(_tensor_to_uint8(result['rgb']))

            acc_d = result['acc'].clamp(min=1e-6)
            depth = (result['depth_premul'] / acc_d).detach().permute(1, 2, 0).cpu().numpy()
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
#  make_video mode – stitch existing *_rgb.png files into a strip video
# ══════════════════════════════════════════════════════════════════════════

def make_panoramic_video(render_dir: str, fps: int = 10,
                        output_path: str | None = None) -> None:
    """Stitch per-frame *_rgb.png files from a rendered output directory into
    a left|front|right|back panoramic strip video.

    Args:
        render_dir   : Directory containing the *_rgb.png files
                       (e.g. render_with_sh_sky/train/ours_epoch_150/).
        fps          : Output video frame rate.
        output_path  : Where to write the .mp4. Defaults to
                       {render_dir}/panoramic_rgb.mp4.
    """
    render_dir = os.path.abspath(render_dir)
    if output_path is None:
        output_path = os.path.join(render_dir, "panoramic_rgb.mp4")

    # Collect all *_rgb.png files.
    rgb_files = sorted(Path(render_dir).glob("*_rgb.png"))
    if not rgb_files:
        raise FileNotFoundError(f"No *_rgb.png files found in {render_dir}")

    # Build mapping: frame_id -> {face: path}
    import re
    face_aliases = {
        "left":  "left",
        "front": "front",
        "right": "right",
        "back":  "back",
    }
    frame_faces: dict[str, dict[str, Path]] = {}
    for p in rgb_files:
        # Expect stem like "0002_front" or "some_name_front"
        stem = p.stem  # e.g. "0002_front_rgb" -> remove trailing "_rgb"
        if stem.endswith("_rgb"):
            stem = stem[:-4]
        # Last token is the face name
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        frame_id, face = parts[0], parts[1].lower()
        if face not in face_aliases:
            continue
        frame_faces.setdefault(frame_id, {})[face] = p

    # Keep only frames that have all 4 faces.
    face_order = ["left", "front", "right", "back"]
    complete_frames = sorted(
        fid for fid, faces in frame_faces.items()
        if all(f in faces for f in face_order)
    )
    if not complete_frames:
        raise ValueError(
            "No frames with all four faces (left/front/right/back) found.")

    print(f"Found {len(complete_frames)} complete frames in {render_dir}")

    strip_frames: list[np.ndarray] = []
    for fid in tqdm(complete_frames, desc="Building panoramic strip"):
        panels = [imageio.imread(str(frame_faces[fid][f])) for f in face_order]
        # Ensure all panels have the same height (resize if needed).
        h_ref = panels[0].shape[0]
        resized = []
        for panel in panels:
            if panel.shape[0] != h_ref:
                panel = cv2.resize(panel, (panel.shape[1], h_ref))
            resized.append(panel)
        strip_frames.append(np.concatenate(resized, axis=1))

    imageio.mimwrite(output_path, strip_frames, fps=fps)
    print(f"Saved panoramic video → {output_path}")
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GoPro Max Neighbour – Rendering / Evaluation")
    parser.add_argument(
        "--config", default="configs/gopromax_neighbour.yaml",
        help="Path to YAML config file.")
    parser.add_argument(
        "--mode", choices=["evaluate", "trajectory", "make_video"],
        nargs='+', default=["evaluate"],
        help="One or more modes to run in order: evaluate, trajectory, make_video.")
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
    parser.add_argument(
        "--sky_sh_coeffs", type=float, nargs='+', default=None,
        help="Manual SH coefficients for the sky (skips fitting if provided).")
    parser.add_argument(
        "--no_fit_sky", action="store_true",
        help="Disable sky entirely (render against plain background).")
    parser.add_argument(
        "--fit_iters", type=int, default=300,
        help="Number of optimisation steps for sky SH fitting.")
    parser.add_argument(
        "--fit_lr", type=float, default=1e-2,
        help="Adam learning rate for sky SH fitting.")
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save the output. "
             "For make_video mode, this is the directory containing *_rgb.png files.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.output_dir is None:
        # Default output dir to folder of config file.
        output_dir = os.path.join(os.path.dirname(args.config), "render_with_sh_sky")
    else:
        output_dir = args.output_dir
    print(f"Rendering results to {output_dir}")

    fit_sky = (not args.no_fit_sky) and (args.sky_sh_coeffs is None)

    for mode in args.mode:
        if mode == "make_video":
            # If --output_dir points directly to a *_rgb.png directory, use it.
            # Otherwise auto-discover the latest ours_epoch_* subfolder under
            # {output_dir}/train/ (the default evaluate output layout).
            video_dir = args.output_dir
            if video_dir is None or not any(Path(video_dir).glob("*_rgb.png")):
                search_root = Path(output_dir) / "train"
                epoch_dirs = sorted(search_root.glob("ours_epoch_*"))
                if not epoch_dirs:
                    raise FileNotFoundError(
                        f"No ours_epoch_* directories found under {search_root}. "
                        "Run evaluate first or pass --output_dir pointing at the images folder.")
                video_dir = str(epoch_dirs[-1])
                print(f"make_video: using {video_dir}")
            make_panoramic_video(video_dir, fps=args.fps)
        elif mode == "evaluate":
            render_sets(cfg, args.config, epoch=args.epoch,
                        skip_train=args.skip_train,
                        skip_test=args.skip_test,
                        sky_sh_coeffs=args.sky_sh_coeffs,
                        output_dir=output_dir,
                        fit_sky=fit_sky,
                        fit_iters=args.fit_iters,
                        fit_lr=args.fit_lr)
        elif mode == "trajectory":
            render_trajectory(cfg, args.config, epoch=args.epoch, fps=args.fps,
                              sky_sh_coeffs=args.sky_sh_coeffs,
                              output_dir=output_dir,
                              fit_sky=fit_sky,
                              fit_iters=args.fit_iters,
                              fit_lr=args.fit_lr)
