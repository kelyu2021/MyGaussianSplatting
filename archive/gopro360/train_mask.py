"""
GoPro 360 Gaussian Splatting – Training Script
================================================

Entry point for training Gaussian Splatting on GoPro 360° COLMAP data.
All heavy lifting is delegated to modules under ``gopro360/lib/``.

Usage
-----
    cd MyGaussianSplatting/gopro360
    python train.py --cfg_file configs/gopro360.yaml

Outputs
-------
    a. Checkpoints      →  {model_path}/trained_model/
    b. Saved PLY        →  {model_path}/point_cloud/
    c. Log Images       →  {model_path}/log_images/
    d. TensorBoard Logs →  {record_dir}/
"""

from __future__ import annotations

import os
import sys
import copy
from pathlib import Path
from random import shuffle

# Reduce CUDA memory fragmentation
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════
#  Path setup – gopro360/ is the project root
# ═══════════════════════════════════════════════════════════════════════════
_SCRIPT_DIR = Path(__file__).resolve().parent              # gopro360/
sys.path.insert(0, str(_SCRIPT_DIR))

# ── Library imports (all from gopro360/lib/) ─────────────────────────────
from lib.config import cfg                                              # noqa: E402
from lib.utils.loss_utils import l1_loss, psnr, ssim                    # noqa: E402
from lib.utils.general_utils import inverse_sigmoid, safe_state         # noqa: E402
from lib.utils.cfg_utils import save_cfg                                # noqa: E402
from lib.utils.camera_utils import Camera                               # noqa: E402
from lib.utils.sh_utils import RGB2SH                                   # noqa: E402
from lib.utils.system_utils import searchForMaxCheckpoint               # noqa: E402
from lib.models.street_gaussian_renderer import StreetGaussianRenderer  # noqa: E402
from lib.models.street_gaussian_model import StreetGaussianModel        # noqa: E402
from lib.models.scene import Scene                                      # noqa: E402
from lib.datasets.gopro360_dataset import GoPro360Dataset               # noqa: E402
from lib.utils.training_utils import (                                  # noqa: E402
    prepare_output_and_logger,
    training_report,
    save_log_images,
    MetricCSVLogger,
)
from diff_gaussian_rasterization import (                               # noqa: E402
    GaussianRasterizationSettings, GaussianRasterizer,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Roof-Hole Seeding
# ═══════════════════════════════════════════════════════════════════════════

_roof_mask_cache: dict = {}   # cam image_name → (H, W) bool tensor on CPU


def _load_roof_mask(image_name: str, roof_mask_dir: str,
                    device: str = "cpu") -> torch.Tensor | None:
    """Load the roof-only mask for a camera.  Cached across calls.

    Returns a bool tensor (H, W) where True = roof (masked / hole).
    """
    if image_name in _roof_mask_cache:
        return _roof_mask_cache[image_name]

    from PIL import Image as _PILImage
    p = Path(roof_mask_dir) / image_name
    if not p.exists():
        _roof_mask_cache[image_name] = None
        return None
    arr = np.array(_PILImage.open(str(p)).convert("L"))
    # 0 = roof (masked), 255 = valid  →  True where roof
    roof = torch.from_numpy(arr < 128)
    _roof_mask_cache[image_name] = roof
    return roof


def _roof_mask_candidate_names(cam: Camera) -> list[str]:
    """Return candidate roof-mask filenames for a real or virtual camera."""
    names: list[str] = []

    def _add(name: str | None):
        if not name:
            return
        stem = Path(name).stem
        names.append(f"{stem}.png")
        names.append(stem)

    _add(getattr(cam, "image_name", None))

    meta = getattr(cam, "meta", {}) or {}
    for src_name in meta.get("roof_mask_source_names", []):
        _add(src_name)

    deduped: list[str] = []
    seen = set()
    for name in names:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    return deduped


def _get_roof_mask_for_camera(cam: Camera, roof_mask_dir: str) -> torch.Tensor | None:
    for candidate in _roof_mask_candidate_names(cam):
        roof = _load_roof_mask(candidate, roof_mask_dir)
        if roof is not None:
            return roof
    return None


def _camera_to_c2w(cam: Camera) -> np.ndarray:
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = cam.R.T
    w2c[:3, 3] = np.asarray(cam.T, dtype=np.float32)
    return np.linalg.inv(w2c)


def _project_to_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1.0
        rot = u @ vt
    return rot.astype(np.float32)


def _interpolate_c2w(cam_a: Camera, cam_b: Camera, alpha: float) -> np.ndarray:
    c2w_a = _camera_to_c2w(cam_a)
    c2w_b = _camera_to_c2w(cam_b)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 3] = (1.0 - alpha) * c2w_a[:3, 3] + alpha * c2w_b[:3, 3]
    blended_rot = (1.0 - alpha) * c2w_a[:3, :3] + alpha * c2w_b[:3, :3]
    c2w[:3, :3] = _project_to_rotation(blended_rot)
    return c2w


def _make_virtual_hole_seed_camera(cam_a: Camera, cam_b: Camera, alpha: float) -> Camera:
    """Create a pose-interpolated camera used only for hole seeding."""
    new_cam = copy.deepcopy(cam_a)
    new_cam.set_extrinsic(_interpolate_c2w(cam_a, cam_b, alpha))

    meta = dict(getattr(new_cam, "meta", {}) or {})
    meta["is_virtual_hole_seed"] = True
    meta["roof_mask_source_names"] = [cam_a.image_name, cam_b.image_name]
    new_cam.meta = meta

    alpha_tag = int(round(alpha * 1000.0))
    new_cam.image_name = (
        f"{cam_a.image_name}__virt__{cam_b.image_name}__{alpha_tag:03d}"
    )
    return new_cam


def build_virtual_hole_seed_cameras(
    train_cameras,
    alphas,
    max_virtual_cameras=0,
    max_frame_gap=2,
):
    """Interpolate neighboring training poses to seed holes from extra viewpoints."""
    if not alphas or max_virtual_cameras == 0:
        return []

    grouped: dict[int, list[tuple[int, Camera]]] = {}
    for cam in train_cameras:
        meta = getattr(cam, "meta", {}) or {}
        frame_idx = meta.get("frame_idx")
        face_id = meta.get("cam")
        if frame_idx is None or face_id is None:
            continue
        grouped.setdefault(face_id, []).append((frame_idx, cam))

    virtual_cameras = []
    for face_cams in grouped.values():
        face_cams.sort(key=lambda item: item[0])
        for (frame_a, cam_a), (frame_b, cam_b) in zip(face_cams, face_cams[1:]):
            frame_gap = frame_b - frame_a
            if frame_gap <= 0 or frame_gap > max_frame_gap:
                continue
            for alpha in alphas:
                if alpha <= 0.0 or alpha >= 1.0:
                    continue
                virtual_cameras.append(
                    _make_virtual_hole_seed_camera(cam_a, cam_b, alpha)
                )

    if max_virtual_cameras > 0 and len(virtual_cameras) > max_virtual_cameras:
        step = max(1, len(virtual_cameras) // max_virtual_cameras)
        virtual_cameras = virtual_cameras[::step][:max_virtual_cameras]

    return virtual_cameras


def _prepare_2d_mask(mask_tensor, height: int, width: int, device: str):
    if mask_tensor is None:
        return None

    mask_2d = mask_tensor
    if not isinstance(mask_2d, torch.Tensor):
        mask_2d = torch.as_tensor(mask_2d)
    if mask_2d.ndim == 3:
        mask_2d = mask_2d[0]
    mask_2d = mask_2d.to(device)

    if mask_2d.shape[-2:] != (height, width):
        mask_2d = F.interpolate(
            mask_2d.float().unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode="nearest",
        )[0, 0]

    return mask_2d > 0.5


def _build_support_valid_mask(cam: Camera, roof_mask_dir: str, device: str):
    height, width = int(cam.image_height), int(cam.image_width)
    valid_mask = None

    if "mask" in cam.guidance:
        valid_mask = _prepare_2d_mask(cam.guidance["mask"], height, width, device)

    roof_mask = _get_roof_mask_for_camera(cam, roof_mask_dir)
    if roof_mask is not None:
        roof_mask = _prepare_2d_mask(roof_mask, height, width, device)
        valid_mask = (~roof_mask) if valid_mask is None else (valid_mask & ~roof_mask)

    return valid_mask


def _select_support_cameras(
    source_cam: Camera,
    train_cameras,
    max_support_cameras: int,
    max_support_frame_gap: int,
):
    source_meta = getattr(source_cam, "meta", {}) or {}
    source_frame = source_meta.get("frame_idx")
    source_face = source_meta.get("cam")

    scored = []
    for candidate in train_cameras:
        if candidate is source_cam:
            continue

        candidate_meta = getattr(candidate, "meta", {}) or {}
        candidate_frame = candidate_meta.get("frame_idx")
        candidate_face = candidate_meta.get("cam")

        if source_frame is not None and candidate_frame is not None:
            frame_gap = abs(candidate_frame - source_frame)
            if max_support_frame_gap > 0 and frame_gap > max_support_frame_gap:
                continue
        else:
            frame_gap = 0

        face_penalty = 0 if source_face == candidate_face else 1
        scored.append(((face_penalty, frame_gap, candidate.id), candidate))

    scored.sort(key=lambda item: item[0])
    return [cam for _, cam in scored[:max_support_cameras]]


@torch.no_grad()
def _sample_seed_rgb_from_support_views(
    seed_xyz: torch.Tensor,
    source_cam: Camera,
    train_cameras,
    roof_mask_dir: str,
    device: str,
    max_support_cameras: int,
    max_support_frame_gap: int,
):
    support_cameras = _select_support_cameras(
        source_cam,
        train_cameras,
        max_support_cameras=max_support_cameras,
        max_support_frame_gap=max_support_frame_gap,
    )
    if not support_cameras:
        return None, None

    num_seeds = seed_xyz.shape[0]
    rgb_accum = torch.zeros((num_seeds, 3), dtype=torch.float32, device=device)
    weight_accum = torch.zeros((num_seeds, 1), dtype=torch.float32, device=device)
    source_meta = getattr(source_cam, "meta", {}) or {}
    source_frame = source_meta.get("frame_idx")

    for support_cam in support_cameras:
        height, width = int(support_cam.image_height), int(support_cam.image_width)
        image = support_cam.original_image.to(device, non_blocking=True)
        valid_mask = _build_support_valid_mask(support_cam, roof_mask_dir, device)

        R_c2w = torch.tensor(support_cam.R, dtype=torch.float32, device=device)
        R_w2c = R_c2w.T
        t_w2c = torch.tensor(support_cam.T, dtype=torch.float32, device=device)
        K = (support_cam.K.to(device).float()
             if isinstance(support_cam.K, torch.Tensor)
             else torch.tensor(support_cam.K, dtype=torch.float32, device=device))

        cam_pts = seed_xyz @ R_w2c.T + t_w2c.unsqueeze(0)
        depth = cam_pts[:, 2]
        px = (K @ cam_pts.T).T
        u = px[:, 0] / depth.clamp(min=1e-6)
        v = px[:, 1] / depth.clamp(min=1e-6)

        visible = (
            (depth > 0.1)
            & (u >= 0)
            & (u < width)
            & (v >= 0)
            & (v < height)
        )
        keep = visible.nonzero(as_tuple=True)[0]
        if keep.numel() == 0:
            continue

        u_i = u[keep].round().long().clamp(0, width - 1)
        v_i = v[keep].round().long().clamp(0, height - 1)

        if valid_mask is not None:
            pixel_valid = valid_mask[v_i, u_i]
            if not pixel_valid.any():
                continue
            keep = keep[pixel_valid]
            u_i = u_i[pixel_valid]
            v_i = v_i[pixel_valid]

        rgb = image[:, v_i, u_i].permute(1, 0)
        weight = 1.0 / depth[keep].clamp(min=1.0)

        support_meta = getattr(support_cam, "meta", {}) or {}
        support_frame = support_meta.get("frame_idx")
        if source_frame is not None and support_frame is not None:
            frame_gap = abs(support_frame - source_frame)
            weight = weight / (1.0 + 0.05 * frame_gap)

        rgb_accum[keep] += rgb * weight.unsqueeze(1)
        weight_accum[keep] += weight.unsqueeze(1)

    valid_rgb = weight_accum[:, 0] > 0
    if not valid_rgb.any():
        return None, None

    rgb = torch.zeros_like(rgb_accum)
    rgb[valid_rgb] = rgb_accum[valid_rgb] / weight_accum[valid_rgb]
    return rgb.clamp(0.0, 1.0), valid_rgb


@torch.no_grad()
def seed_hole_gaussians(gaussians, train_cameras, device="cuda",
                        max_seeds=70000, boundary_kernel=21,
                        roof_mask_dir="colmap/output/masks_roof_depth",
                        max_total_gaussians=10_000_000,
                        seed_cameras=None):
    """Seed new Gaussians in the roof-hole region during training.

    Instead of shifting boundary Gaussians toward an abstract centroid,
    this directly **unprojects roof-mask pixels into 3D** at the estimated
    road-surface depth.  For each camera with a roof mask:

    1. Project existing Gaussians → find depth of boundary pixels (road
       surface just outside the roof mask edge).
    2. Use median boundary depth as the estimated depth for masked pixels.
    3. Sample a grid of pixels inside the roof mask.
    4. Unproject them to 3D at the estimated depth.
    5. Copy colour / scale / rotation from the nearest boundary Gaussian.

    This places seeds exactly where the hole is in 3D, at the correct
    road-surface depth, so the optimiser can refine them using views where
    that surface is visible.

    Returns the number of seeded Gaussians.
    """
    bg = gaussians.background
    xyz = bg._xyz.detach()   # (N, 3)  — stay on original device first
    opacity = torch.sigmoid(bg._opacity.detach())  # (N, 1)
    N = xyz.shape[0]

    # ── Guard: skip if already at capacity ────────────────────────────
    if max_total_gaussians > 0 and N >= max_total_gaussians:
        print(f"  [seed_hole] Skipping: {N:,} Gaussians already at cap "
              f"({max_total_gaussians:,})")
        return 0
    # Adapt budget to remaining headroom
    headroom = max_total_gaussians - N if max_total_gaussians > 0 else max_seeds
    max_seeds = min(max_seeds, headroom)

    # ── Cameras with roof masks ──────────────────────────────────────
    cam_mask_pairs = []
    candidate_cameras = seed_cameras if seed_cameras is not None else train_cameras
    for c in candidate_cameras:
        roof = _get_roof_mask_for_camera(c, roof_mask_dir)
        if roof is not None and roof.any():
            cam_mask_pairs.append((c, roof))
    if not cam_mask_pairs:
        return 0

    # Subsample cameras for efficiency (up to 16 to save memory)
    if len(cam_mask_pairs) > 16:
        step = max(1, len(cam_mask_pairs) // 16)
        cam_mask_pairs = cam_mask_pairs[::step][:16]

    # Budget per camera
    seeds_per_cam = max(100, max_seeds // len(cam_mask_pairs))
    pad = boundary_kernel // 2
    max_support_cameras = int(getattr(cfg.optim, "hole_multiview_support_cameras", 12))
    max_support_frame_gap = int(getattr(cfg.optim, "hole_multiview_support_frame_gap", 48))

    all_new_xyz = []
    all_new_rgb = []
    all_new_rgb_valid = []

    for cam, roof_cpu in cam_mask_pairs:
        roof = roof_cpu.to(device)            # (H, W) bool, True=roof
        H, W = roof.shape
        valid_zone = ~roof

        # ── Project Gaussians into this camera ────────────────────────
        R_c2w = torch.tensor(cam.R, dtype=torch.float32, device=device)
        R_w2c = R_c2w.T
        t_w2c = torch.tensor(cam.T, dtype=torch.float32, device=device)
        K = (cam.K.to(device).float() if isinstance(cam.K, torch.Tensor)
             else torch.tensor(cam.K, dtype=torch.float32, device=device))
        K_inv = torch.linalg.inv(K)

        cam_pts = xyz @ R_w2c.T + t_w2c.unsqueeze(0)   # (N, 3)
        depth   = cam_pts[:, 2]
        px      = (K @ cam_pts.T).T
        u       = px[:, 0] / depth.clamp(min=1e-6)
        v       = px[:, 1] / depth.clamp(min=1e-6)

        vis = ((depth > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
               & (opacity[:, 0] > 0.05))
        vi  = vis.nonzero(as_tuple=True)[0]
        u_i = u[vi].long().clamp(0, W - 1)
        v_i = v[vi].long().clamp(0, H - 1)

        # ── Boundary zone (dilated roof ∩ valid) ─────────────────────
        roof_f  = roof.float().unsqueeze(0).unsqueeze(0)
        dilated = F.max_pool2d(roof_f, kernel_size=boundary_kernel,
                               stride=1, padding=pad)
        boundary_zone = (dilated[0, 0] > 0.5) & valid_zone

        # Gaussians in the boundary zone
        in_boundary = boundary_zone[v_i, u_i]
        bd_global_idx = vi[in_boundary]
        bd_depths     = depth[bd_global_idx]

        if len(bd_depths) < 10:
            continue

        # ── Estimate road depth at roof pixels ────────────────────────
        # Use median depth of boundary Gaussians (robust to outliers)
        road_depth = bd_depths.median()

        bd_xyz = xyz[bd_global_idx]   # (B, 3)

        # ── Sample pixels inside the roof mask ────────────────────────
        roof_pixels = roof.nonzero(as_tuple=False)  # (M, 2) — (v, u)
        if len(roof_pixels) > seeds_per_cam:
            perm = torch.randperm(len(roof_pixels), device=device)[:seeds_per_cam]
            roof_pixels = roof_pixels[perm]

        # ── Unproject to 3D ───────────────────────────────────────────
        # pixel (u, v) + depth → camera coords → world coords
        uv1 = torch.stack([
            roof_pixels[:, 1].float(),   # u
            roof_pixels[:, 0].float(),   # v
            torch.ones(len(roof_pixels), device=device),
        ], dim=1)                           # (S, 3)
        cam_coords = (K_inv @ uv1.T).T * road_depth   # (S, 3)
        # camera coords → world coords:  X_world = R_c2w @ X_cam - R_c2w @ t_w2c
        world_coords = (R_c2w @ cam_coords.T).T - (R_c2w @ t_w2c).unsqueeze(0)

        # ── Snap y to nearest boundary Gaussian's road height ─────────
        # Use x,z only so altitude differences don't skew matching.
        seed_xz = world_coords[:, [0, 2]]             # (S, 2)
        bd_xz   = bd_xyz[:, [0, 2]]                   # (B, 2)
        dists   = torch.cdist(seed_xz, bd_xz)         # (S, B)
        _, nn   = dists.min(dim=1)                     # (S,)
        world_coords[:, 1] = bd_xyz[nn, 1]

        seed_rgb, seed_rgb_valid = _sample_seed_rgb_from_support_views(
            world_coords,
            source_cam=cam,
            train_cameras=train_cameras,
            roof_mask_dir=roof_mask_dir,
            device=device,
            max_support_cameras=max_support_cameras,
            max_support_frame_gap=max_support_frame_gap,
        )
        if seed_rgb is None or seed_rgb_valid is None:
            seed_rgb = torch.zeros((world_coords.shape[0], 3), dtype=torch.float32)
            seed_rgb_valid = torch.zeros(world_coords.shape[0], dtype=torch.bool)

        all_new_xyz.append(world_coords.cpu())
        all_new_rgb.append(seed_rgb.cpu())
        all_new_rgb_valid.append(seed_rgb_valid.cpu())
        # Free per-camera GPU tensors
        del roof, cam_pts, depth, px, u, v, vis, vi, u_i, v_i
        del roof_f, dilated, boundary_zone, in_boundary
        del bd_global_idx, bd_depths, roof_pixels, uv1, cam_coords
        del world_coords, bd_xyz, dists, nn, seed_rgb, seed_rgb_valid
        torch.cuda.empty_cache()

    if not all_new_xyz:
        return 0

    new_xyz   = torch.cat(all_new_xyz, dim=0)  # CPU
    new_rgb = torch.cat(all_new_rgb, dim=0)
    new_rgb_valid = torch.cat(all_new_rgb_valid, dim=0)
    # Cap seed count
    if new_xyz.shape[0] > max_seeds:
        perm    = torch.randperm(new_xyz.shape[0])[:max_seeds]
        new_xyz = new_xyz[perm]
        new_rgb = new_rgb[perm]
        new_rgb_valid = new_rgb_valid[perm]

    # ── Phase 2: K-NN interpolation from ALL nearby Gaussians ─────────
    # By searching all road-level Gaussians (not just boundary ones from
    # one camera), we naturally pick up Gaussians placed / optimised by
    # other viewpoints that see the same ground area without the roof.
    K = 4
    new_xyz_gpu = new_xyz.cuda()

    # Filter existing Gaussians to a road-level y-band around the seeds
    seed_y_med = new_xyz_gpu[:, 1].median()
    seed_y_span = (new_xyz_gpu[:, 1].max() - new_xyz_gpu[:, 1].min()).clamp(min=1.0)
    y_margin = seed_y_span * 0.5
    road_mask = ((xyz[:, 1] >= seed_y_med - y_margin)
                 & (xyz[:, 1] <= seed_y_med + y_margin)
                 & (opacity[:, 0] > 0.05))
    road_global_idx = road_mask.nonzero(as_tuple=True)[0]

    # Subsample if too large (keep cdist memory bounded)
    MAX_ROAD = 100_000
    if len(road_global_idx) > MAX_ROAD:
        perm = torch.randperm(len(road_global_idx), device=device)[:MAX_ROAD]
        road_global_idx = road_global_idx[perm]

    road_xz = xyz[road_global_idx][:, [0, 2]]          # (R, 2)
    seed_xz = new_xyz_gpu[:, [0, 2]]                    # (S, 2)

    # Batched K-NN in horizontal (x, z) plane
    BATCH = 2000
    knn_global_list, knn_w_list = [], []
    for i in range(0, len(seed_xz), BATCH):
        d = torch.cdist(seed_xz[i:i + BATCH], road_xz)  # (B, R)
        topk_d, topk_local = d.topk(K, dim=1, largest=False)
        w = 1.0 / topk_d.clamp(min=1e-6)
        w = w / w.sum(dim=1, keepdim=True)
        knn_global_list.append(road_global_idx[topk_local])
        knn_w_list.append(w)
        del d

    knn_idx = torch.cat(knn_global_list, dim=0)   # (S, K) global indices
    knn_w   = torch.cat(knn_w_list, dim=0)         # (S, K) weights

    def _idw_gather(param, idx, w):
        """Inverse-distance-weighted gather: param(N,*D) idx(S,K) w(S,K)->(S,*D)"""
        g = param[idx]          # (S, K, *D)
        ww = w
        for _ in range(g.dim() - 2):
            ww = ww.unsqueeze(-1)
        return (g * ww).sum(dim=1)

    n_to_seed = new_xyz_gpu.shape[0]
    new_features_dc = _idw_gather(bg._features_dc.detach(), knn_idx, knn_w)
    new_features_rest = _idw_gather(bg._features_rest.detach(), knn_idx, knn_w)
    new_opacity = _idw_gather(bg._opacity.detach(), knn_idx, knn_w)
    new_scaling = _idw_gather(bg._scaling.detach(), knn_idx, knn_w)
    new_rotation = F.normalize(
        _idw_gather(bg._rotation.detach(), knn_idx, knn_w), dim=-1)
    new_semantic = _idw_gather(bg._semantic.detach(), knn_idx, knn_w)

    new_rgb_gpu = new_rgb.to(device)
    new_rgb_valid_gpu = new_rgb_valid.to(device)
    if new_rgb_valid_gpu.any():
        multiview_dc = RGB2SH(new_rgb_gpu).unsqueeze(1)
        new_features_dc = torch.where(
            new_rgb_valid_gpu[:, None, None],
            multiview_dc,
            new_features_dc,
        )

        min_seed_opacity = float(getattr(cfg.optim, "hole_seed_min_opacity", 0.08))
        seed_opacity = torch.sigmoid(new_opacity)
        seed_opacity = torch.maximum(
            seed_opacity,
            torch.full_like(seed_opacity, min_seed_opacity),
        )
        new_opacity = torch.where(
            new_rgb_valid_gpu[:, None],
            inverse_sigmoid(seed_opacity.clamp(max=0.999)),
            new_opacity,
        )

    new_tensors = {
        "xyz":      new_xyz_gpu,
        "f_dc":     new_features_dc,
        "f_rest":   new_features_rest,
        "opacity":  new_opacity,
        "scaling":  new_scaling,
        "rotation": new_rotation,
        "semantic": new_semantic,
    }

    bg.densification_postfix(new_tensors)
    torch.cuda.empty_cache()
    return n_to_seed


# ═══════════════════════════════════════════════════════════════════════════
#  Multi-GPU helpers
# ═══════════════════════════════════════════════════════════════════════════

_GAUSS_PARAM_NAMES = [
    '_xyz', '_features_dc', '_features_rest',
    '_opacity', '_scaling', '_rotation', '_semantic',
]


def _forward_loss_backward_secondary(
    device: str,
    cam: Camera,
    gaussians: StreetGaussianModel,
    optim_args,
    roof_mask_dir: str,
):
    """Render + loss + backward on a secondary GPU.

    Clones Gaussian parameters to *device*, runs the CUDA rasteriser,
    computes the same losses as the main loop (minus colour-correction),
    calls ``backward()``, and returns gradients transferred to cuda:0.
    """
    bg = gaussians.background

    # Set CUDA device context so custom CUDA kernels target the right GPU
    with torch.cuda.device(device):
        # Clone raw parameters to target device (leaf tensors with grad)
        params: dict[str, torch.Tensor] = {}
        for name in _GAUSS_PARAM_NAMES:
            p = getattr(bg, name).data
            params[name] = p.to(device).detach().requires_grad_(True)

        means3D = params['_xyz']
        N = means3D.shape[0]
        if N == 0:
            return {}, {}

        # Activations (match GaussianModel properties)
        opacity = torch.sigmoid(params['_opacity'])
        scales = torch.exp(params['_scaling'])
        rotations = F.normalize(params['_rotation'], dim=-1)
        shs = torch.cat([params['_features_dc'], params['_features_rest']], dim=1)

        # Camera tensors on target device
        world_view_transform = cam.world_view_transform.to(device)
        full_proj_transform = cam.full_proj_transform.to(device)
        camera_center = cam.camera_center.to(device)

        # Rasteriser on target device
        white_bg = cfg.data.white_background
        bg_color = torch.tensor(
            [1, 1, 1] if white_bg else [0, 0, 0],
            dtype=torch.float32, device=device)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(cam.image_height),
            image_width=int(cam.image_width),
            tanfovx=math.tan(cam.FoVx * 0.5),
            tanfovy=math.tan(cam.FoVy * 0.5),
            bg=bg_color,
            scale_modifier=cfg.render.scaling_modifier,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            sh_degree=gaussians.active_sh_degree,
            campos=camera_center,
            prefiltered=False,
            debug=cfg.render.debug,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        screenspace_points = torch.zeros(
            (N, 3), requires_grad=True, device=device).float() + 0
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass

        # Semantic features (if enabled)
        semantics_tensor = None
        if cfg.data.get('use_semantic', False):
            semantic_mode = cfg.model.gaussian.get('semantic_mode', 'logits')
            sem = params['_semantic']
            if semantic_mode == 'probabilities':
                sem = F.softmax(sem, dim=1)
            semantics_tensor = sem

        # Rasterise
        rendered_color, radii, rendered_depth, rendered_acc, rendered_feature = rasterizer(
            means3D=means3D,
            means2D=screenspace_points,
            opacities=opacity,
            shs=shs,
            colors_precomp=None,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
            semantics=semantics_tensor,
        )

        image = rendered_color
        acc = rendered_acc

        # Ground truth + masks on target device
        gt_image = cam.original_image.to(device, non_blocking=True)
        mask = None
        if "mask" in cam.guidance:
            mask = cam.guidance["mask"].to(device, non_blocking=True)

        roof_mask_2d = _load_roof_mask(cam.image_name + ".png", roof_mask_dir)
        if roof_mask_2d is None:
            roof_mask_2d = _load_roof_mask(cam.image_name, roof_mask_dir)
        if roof_mask_2d is not None:
            roof_mask_2d = roof_mask_2d.to(device)
            rH, rW = roof_mask_2d.shape
            iH, iW = gt_image.shape[1], gt_image.shape[2]
            if rH != iH or rW != iW:
                roof_mask_2d = F.interpolate(
                    roof_mask_2d.float().unsqueeze(0).unsqueeze(0),
                    size=(iH, iW), mode='nearest'
                )[0, 0] > 0.5

        # ── Loss (same terms as main loop, minus colour correction) ───
        scalar_dict: dict = {}
        lambda_l1 = getattr(optim_args, "lambda_l1", 1.0)
        Ll1 = l1_loss(image, gt_image, mask)
        scalar_dict["l1_loss"] = Ll1.item()
        loss = (
            (1.0 - optim_args.lambda_dssim) * lambda_l1 * Ll1
            + optim_args.lambda_dssim * (1.0 - ssim(image, gt_image, mask=mask))
        )

        lambda_sh = getattr(optim_args, "lambda_sh_reg", 1e-3)
        if lambda_sh > 0:
            sh_rest = params['_features_rest']
            loss += lambda_sh * (sh_rest ** 2).mean()

        lambda_sky_acc = getattr(optim_args, "lambda_sky_acc", 1e-2)
        if lambda_sky_acc > 0 and mask is not None:
            if roof_mask_2d is not None:
                sky_only = (1 - mask.float()) * (1 - roof_mask_2d.float())
            else:
                sky_only = 1 - mask.float()
            loss += lambda_sky_acc * (acc * sky_only).mean()

        scalar_dict["loss"] = loss.item()

        loss.backward()
        torch.cuda.synchronize(device)

        # Collect gradients → cuda:0
        grads: dict[str, torch.Tensor] = {}
        for name in _GAUSS_PARAM_NAMES:
            p = params[name]
            if p.grad is not None:
                grads[name] = p.grad.to('cuda:0')

        return grads, scalar_dict


# ═══════════════════════════════════════════════════════════════════════════
#  Training Loop
# ═══════════════════════════════════════════════════════════════════════════

def training():
    """Main training loop for GoPro 360 Gaussian Splatting.

    The loop is **epoch-based**: each epoch iterates through every training
    camera exactly once (shuffled).  Epoch-based config values are used
    directly — no conversion to iterations.
    """
    training_args = cfg.train
    optim_args = cfg.optim
    data_args = cfg.data

    tb_writer = prepare_output_and_logger()
    csv_logger = MetricCSVLogger(cfg.model_path)

    # ── Data & model ──────────────────────────────────────────────────
    dataset = GoPro360Dataset()
    gaussians = StreetGaussianModel(dataset.scene_info.metadata)
    scene = Scene(gaussians=gaussians, dataset=dataset)

    # ── Epoch-based schedule (used directly, no conversion) ─────────
    train_cameras = scene.getTrainCameras()
    cams_per_epoch = len(train_cameras)
    num_epochs = training_args.epochs

    hole_virtual_alphas = [
        float(alpha)
        for alpha in getattr(optim_args, "hole_virtual_view_alphas", [])
    ]
    hole_virtual_max_cameras = int(
        getattr(optim_args, "hole_virtual_max_cameras", 0)
    )
    hole_virtual_max_frame_gap = int(
        getattr(optim_args, "hole_virtual_max_frame_gap", 2)
    )
    virtual_seed_cameras = build_virtual_hole_seed_cameras(
        train_cameras,
        alphas=hole_virtual_alphas,
        max_virtual_cameras=hole_virtual_max_cameras,
        max_frame_gap=hole_virtual_max_frame_gap,
    )
    hole_seed_cameras = train_cameras + virtual_seed_cameras

    test_epochs_set = set(training_args.test_epochs)
    save_epochs_set = set(training_args.save_epochs)
    checkpoint_epochs_set = set(training_args.checkpoint_epochs)

    densify_from_epoch = optim_args.densify_from_epoch
    densify_until_epoch = optim_args.densify_until_epoch
    opacity_reset_interval = optim_args.opacity_reset_epoch_interval
    hole_seed_interval = optim_args.hole_seed_epoch_interval
    hole_seed_until_epoch = optim_args.hole_seed_until_epoch
    prune_interval_epochs = optim_args.prune_epoch_interval

    # position_lr_max_steps must be in iterations for the LR scheduler
    position_lr_max_steps = optim_args.position_lr_max_epochs * cams_per_epoch
    cfg.optim.position_lr_max_steps = position_lr_max_steps

    print(f"Epoch-based training: {num_epochs} epochs × "
          f"{cams_per_epoch} cameras/epoch")
    print(f"  densify:    epoch {densify_from_epoch}–{densify_until_epoch}")
    print(f"  opacity reset every {opacity_reset_interval} epochs")
    print(f"  position LR decay over {optim_args.position_lr_max_epochs} epochs")
    print(f"  hole seeding every {hole_seed_interval}"
          f" epochs until epoch {hole_seed_until_epoch}")
    if virtual_seed_cameras:
        print("  hole seeding viewpoints: "
              f"{len(train_cameras)} real + {len(virtual_seed_cameras)} virtual")
    print(f"  pruning every {prune_interval_epochs} epochs")
    print(f"  test/eval at epochs {sorted(test_epochs_set)}")
    print(f"  checkpoints at epochs {sorted(checkpoint_epochs_set)}")
    print(f"  save PLY at epochs {sorted(save_epochs_set)}")

    # ── Now set up optimiser (uses position_lr_max_steps from cfg) ────
    gaussians.training_setup()

    # ── Resume from checkpoint ────────────────────────────────────────
    start_epoch = 0
    step = 0               # running step counter (for LR scheduler only)
    try:
        loaded_prefix, loaded_num = (
            searchForMaxCheckpoint(cfg.trained_model_dir)
            if cfg.loaded_iter == -1
            else ('epoch', cfg.loaded_iter))
        ckpt_path = os.path.join(
            cfg.trained_model_dir, f"{loaded_prefix}_{loaded_num}.pth"
        )
        state = torch.load(ckpt_path)
        start_epoch = state.get("epoch", loaded_num)
        step = state.get("step", state.get("iter", start_epoch * cams_per_epoch))
        print(f"Resuming from {ckpt_path}  (epoch {start_epoch})")
        gaussians.load_state_dict(state)
    except Exception:
        pass

    print(f"Starting from epoch {start_epoch}")
    save_cfg(cfg, cfg.model_path, epoch=start_epoch)

    renderer = StreetGaussianRenderer()

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end   = torch.cuda.Event(enable_timing=True)

    ema_loss = 0.0
    ema_psnr = 0.0
    ema_ssim = 0.0

    progress = tqdm(range(start_epoch, num_epochs), initial=start_epoch,
                    total=num_epochs, desc="Epochs", unit="ep")

    # ── Multi-GPU setup ───────────────────────────────────────────────
    _n_gpus = torch.cuda.device_count()
    if _n_gpus > 1:
        _executor = ThreadPoolExecutor(max_workers=_n_gpus - 1)
        _roof_mask_dir = getattr(
            data_args, 'roof_mask_dir', 'colmap/output/masks_roof_depth')
        print(f"Multi-GPU training: {_n_gpus} GPUs, "
              f"{_n_gpus} views per step (gradient accumulation)")
    else:
        _executor = None

    # ══════════════════════════════════════════════════════════════════
    #  EPOCH LOOP
    # ══════════════════════════════════════════════════════════════════
    for epoch in range(start_epoch + 1, num_epochs + 1):

        viewpoint_stack = list(train_cameras)
        shuffle(viewpoint_stack)

        # ── Camera loop (one step per camera) ─────────────────────
        for cam_idx, cam in enumerate(viewpoint_stack):
            step += 1

            iter_start.record()
            gaussians.update_learning_rate(step)

            # ── Launch secondary GPU work (multi-GPU) ─────────────────
            secondary_futures = []
            if _executor is not None:
                for gpu_id in range(1, _n_gpus):
                    # Pick next camera from this epoch's stack (or random)
                    sec_idx = (cam_idx + gpu_id) % len(viewpoint_stack) \
                              if viewpoint_stack else 0
                    sec_cam = (viewpoint_stack[sec_idx]
                               if viewpoint_stack
                               else train_cameras[gpu_id % len(train_cameras)])
                    secondary_futures.append(_executor.submit(
                        _forward_loss_backward_secondary,
                        f'cuda:{gpu_id}', sec_cam, gaussians,
                        optim_args, _roof_mask_dir))

            gt_image = cam.original_image
            gt_image = gt_image.cuda(non_blocking=True) if not gt_image.is_cuda else gt_image

            # ── Mask (sky + roof) ─────────────────────────────────────
            if "mask" in cam.guidance:
                mask = cam.guidance["mask"]
                mask = mask.cuda(non_blocking=True) if not mask.is_cuda else mask
            else:
                mask = None

            # ── Roof-only mask (for separating sky from roof) ─────────
            roof_mask_dir = getattr(
                data_args, 'roof_mask_dir', 'colmap/output/masks_roof_depth')
            roof_mask_2d = _load_roof_mask(
                cam.image_name + ".png", roof_mask_dir)
            if roof_mask_2d is None:
                roof_mask_2d = _load_roof_mask(cam.image_name, roof_mask_dir)
            if roof_mask_2d is not None:
                roof_mask_2d = roof_mask_2d.to(gt_image.device)
                rH, rW = roof_mask_2d.shape
                iH, iW = gt_image.shape[1], gt_image.shape[2]
                if rH != iH or rW != iW:
                    roof_mask_2d = F.interpolate(
                        roof_mask_2d.float().unsqueeze(0).unsqueeze(0),
                        size=(iH, iW), mode='nearest'
                    )[0, 0] > 0.5
            else:
                roof_mask_2d = None

            # ── Render ────────────────────────────────────────────────
            render_pkg = renderer.render(cam, gaussians)
            image = render_pkg["rgb"]
            acc   = render_pkg["acc"]
            depth = render_pkg["depth"]
            viewspace_pts = render_pkg["viewspace_points"]
            visibility    = render_pkg["visibility_filter"]
            radii         = render_pkg["radii"]

            scalar_dict: dict = {}

            # ── RGB loss (L1 + D-SSIM) ───────────────────────────────
            lambda_l1 = getattr(optim_args, "lambda_l1", 1.0)
            Ll1 = l1_loss(image, gt_image, mask)
            scalar_dict["l1_loss"] = Ll1.item()

            loss = (
                (1.0 - optim_args.lambda_dssim) * lambda_l1 * Ll1
                + optim_args.lambda_dssim * (1.0 - ssim(image, gt_image, mask=mask))
            )

            # ── SH higher-order regularisation ────────────────────────
            lambda_sh = getattr(optim_args, "lambda_sh_reg", 1e-3)
            if lambda_sh > 0:
                sh_rest = gaussians.background._features_rest
                sh_reg = lambda_sh * (sh_rest ** 2).mean()
                scalar_dict["sh_reg_loss"] = sh_reg.item()
                loss += sh_reg

            # ── Sky opacity loss: acc should be 0 where sky (NOT roof)
            lambda_sky_acc = getattr(optim_args, "lambda_sky_acc", 1e-2)
            if lambda_sky_acc > 0 and mask is not None:
                if roof_mask_2d is not None:
                    sky_only = (1 - mask.float()) * (1 - roof_mask_2d.float())
                else:
                    sky_only = 1 - mask.float()
                sky_acc_loss = lambda_sky_acc * (acc * sky_only).mean()
                scalar_dict["sky_acc_loss"] = sky_acc_loss.item()
                loss += sky_acc_loss

            # ── Colour-correction regularisation ──────────────────────
            lambda_cc = getattr(optim_args, "lambda_color_correction", 0.0)
            if lambda_cc > 0 and getattr(gaussians, "use_color_correction", False):
                cc_loss = gaussians.color_correction.regularization_loss(cam)
                scalar_dict["color_correction_reg_loss"] = cc_loss.item()
                loss += lambda_cc * cc_loss

            scalar_dict["loss"] = loss.item()

            # ── Compute SSIM for logging (detached) ──────────────────
            with torch.no_grad():
                ssim_val = ssim(image, gt_image, mask=mask).item()
                scalar_dict["ssim"] = ssim_val

            loss.backward()

            # ── Accumulate secondary GPU gradients (multi-GPU) ────────
            if secondary_futures:
                for future in secondary_futures:
                    sec_grads, _ = future.result()
                    bg_model = gaussians.background
                    for pname, grad in sec_grads.items():
                        param = getattr(bg_model, pname)
                        if param.grad is not None:
                            param.grad.add_(grad)
                bg_model = gaussians.background
                for pname in _GAUSS_PARAM_NAMES:
                    param = getattr(bg_model, pname)
                    if param.grad is not None:
                        param.grad.div_(_n_gpus)

            iter_end.record()

            # ── Book-keeping (no grad) ────────────────────────────────
            with torch.no_grad():
                tensor_dict: dict = {}

                cur_psnr = psnr(image, gt_image, mask).mean().float()
                scalar_dict["psnr"] = cur_psnr.item()

                epoch_frac = epoch - 1 + (cam_idx + 1) / cams_per_epoch

                if cam_idx % 10 == 0:
                    ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
                    ema_psnr = 0.4 * cur_psnr.item() + 0.6 * ema_psnr
                    ema_ssim = 0.4 * ssim_val + 0.6 * ema_ssim

                    progress.set_postfix({
                        "Exp":   f"{cfg.task}-{cfg.exp_name}",
                        "Loss":  f"{ema_loss:.7f}",
                        "PSNR":  f"{ema_psnr:.4f}",
                        "SSIM":  f"{ema_ssim:.4f}",
                    })

                # ── CSV logging (every 10 cameras) ───────────────────
                if cam_idx % 10 == 0:
                    csv_logger.log_train(epoch_frac, scalar_dict, {
                        "ema_loss": ema_loss,
                        "ema_psnr": ema_psnr,
                        "ema_ssim": ema_ssim,
                    })

                # ── Adaptive density control ──────────────────────────
                if epoch <= densify_until_epoch:
                    gaussians.set_visibility(
                        include_list=list(
                            set(gaussians.model_name_id.keys()) - {"sky"}
                        )
                    )
                    gaussians.set_max_radii2D(radii, visibility)
                    gaussians.add_densification_stats(viewspace_pts, visibility)

                    prune_big = epoch > opacity_reset_interval
                    if (epoch > densify_from_epoch
                            and cam_idx % optim_args.densification_interval == 0):
                        s, t = gaussians.densify_and_prune(
                            max_grad=optim_args.densify_grad_threshold,
                            min_opacity=optim_args.min_opacity,
                            prune_big_points=prune_big,
                        )
                        scalar_dict.update(s)
                        tensor_dict.update(t)

                if epoch <= densify_until_epoch:
                    if (data_args.white_background
                            and epoch == densify_from_epoch and cam_idx == 0):
                        gaussians.reset_opacity()

                # ── TensorBoard ───────────────────────────────────────
                if tb_writer:
                    try:
                        for k, v in scalar_dict.items():
                            tb_writer.add_scalar(f"train/{k}", v, step)
                        for k, v in tensor_dict.items():
                            tb_writer.add_histogram(f"train/{k}", v, step)
                    except Exception:
                        pass

                # ── Optimiser step ────────────────────────────────────
                gaussians.update_optimizer()

        # ══════════════════════════════════════════════════════════════
        #  END OF EPOCH — epoch-level actions
        # ══════════════════════════════════════════════════════════════
        with torch.no_grad():
            progress.update(1)

            # ── Save log images ───────────────────────────────────────
            if epoch % 20 == 0:
                save_log_images(epoch, gt_image, image, depth, acc)

            # ── Save PLY snapshot ─────────────────────────────────────
            if epoch in save_epochs_set:
                print(f"\n[EPOCH {epoch}] Saving Gaussians")
                scene.save(epoch, prefix="epoch")

            # ── Opacity reset ─────────────────────────────────────────
            if (epoch <= densify_until_epoch
                    and opacity_reset_interval > 0
                    and epoch % opacity_reset_interval == 0):
                gaussians.reset_opacity()

            # ── Periodic low-opacity pruning ──────────────────────────
            if (epoch > densify_until_epoch
                    and prune_interval_epochs > 0
                    and epoch % prune_interval_epochs == 0):
                prune_min_op = optim_args.prune_min_opacity
                bg_model = gaussians.background
                prune_mask = (
                    bg_model.get_opacity < prune_min_op
                ).squeeze()
                n_before = bg_model._xyz.shape[0]
                if prune_mask.any():
                    bg_model.prune_points(prune_mask)
                    torch.cuda.empty_cache()
                    n_after = bg_model._xyz.shape[0]
                    print(f"\n[EPOCH {epoch}] Pruned "
                          f"{n_before - n_after:,} low-opacity "
                          f"Gaussians (total: {n_after:,})")

            # ── Seed Gaussians in the roof-hole region ────────────────
            just_reset = (epoch <= densify_until_epoch
                          and opacity_reset_interval > 0
                          and epoch % opacity_reset_interval == 0)
            should_seed = (
                hole_seed_interval > 0
                and epoch >= densify_from_epoch
                and epoch <= hole_seed_until_epoch
                and (epoch % hole_seed_interval == 0 or just_reset)
            )
            if should_seed:
                roof_mask_dir = getattr(
                    data_args, 'roof_mask_dir',
                    'colmap/output/masks_roof_depth')
                n_seeded = seed_hole_gaussians(
                    gaussians, train_cameras,
                    max_seeds=optim_args.hole_max_seeds,
                    roof_mask_dir=roof_mask_dir,
                    max_total_gaussians=optim_args.max_total_gaussians,
                    seed_cameras=hole_seed_cameras,
                )
                n_total = gaussians.background._xyz.shape[0]
                if n_seeded > 0:
                    print(f"\n[EPOCH {epoch}] Seeded "
                          f"{n_seeded} Gaussians in roof hole "
                          f"(total: {n_total:,})")
                    torch.cuda.empty_cache()

            # ── Test-set evaluation ───────────────────────────────────
            if epoch in test_epochs_set:
                training_report(
                    tb_writer,
                    scene, renderer,
                    csv_logger=csv_logger,
                    epoch=epoch,
                    step=step,
                )

            # ── Save checkpoint ───────────────────────────────────────
            is_final = (epoch == num_epochs)
            if epoch in checkpoint_epochs_set or is_final:
                print(f"\n[EPOCH {epoch}] Saving Checkpoint")
                sd = gaussians.save_state_dict(is_final=is_final)
                sd["epoch"] = epoch
                sd["step"] = step
                ckpt_path = os.path.join(
                    cfg.trained_model_dir, f"epoch_{epoch}.pth"
                )
                torch.save(sd, ckpt_path)


# ═══════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Optimizing  {cfg.model_path}")
    safe_state(cfg.train.quiet)
    torch.autograd.set_detect_anomaly(cfg.train.detect_anomaly)
    training()
    print("\nTraining complete.")
    print(f"Run  python visualize_metrics.py --model_path {cfg.model_path}  "
          "to plot training curves.")
