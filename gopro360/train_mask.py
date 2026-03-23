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
from pathlib import Path
from random import randint, shuffle

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
from lib.utils.general_utils import safe_state                          # noqa: E402
from lib.utils.cfg_utils import save_cfg                                # noqa: E402
from lib.utils.camera_utils import Camera                               # noqa: E402
from lib.utils.system_utils import searchForMaxIteration                # noqa: E402
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


@torch.no_grad()
def seed_hole_gaussians(gaussians, train_cameras, device="cuda",
                        max_seeds=70000, boundary_kernel=21,
                        roof_mask_dir="colmap/output/masks_roof_depth",
                        max_total_gaussians=10_000_000):
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
    for c in train_cameras:
        roof = _load_roof_mask(c.image_name + ".png", roof_mask_dir)
        if roof is None:
            roof = _load_roof_mask(c.image_name, roof_mask_dir)
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

    all_new_xyz = []
    all_donor_idx = []    # index into global Gaussian array for colour etc.

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

        # ── Find nearest boundary Gaussian for each seed (donor) ─────
        bd_xyz = xyz[bd_global_idx]   # (B, 3)
        dists  = torch.cdist(world_coords, bd_xyz)   # (S, B)
        _, nn  = dists.min(dim=1)                     # (S,)
        donors = bd_global_idx[nn]                    # global index

        all_new_xyz.append(world_coords.cpu())
        all_donor_idx.append(donors.cpu())
        # Free per-camera GPU tensors
        del roof, cam_pts, depth, px, u, v, vis, vi, u_i, v_i
        del roof_f, dilated, boundary_zone, in_boundary
        del bd_global_idx, bd_depths, roof_pixels, uv1, cam_coords
        del world_coords, bd_xyz, dists, nn, donors
        torch.cuda.empty_cache()

    if not all_new_xyz:
        return 0

    new_xyz   = torch.cat(all_new_xyz, dim=0)  # CPU
    donor_idx = torch.cat(all_donor_idx, dim=0)  # CPU

    # Deduplicate: if seeds from different cameras land on nearly the
    # same 3D point, keep only one (grid-based dedup)
    if new_xyz.shape[0] > max_seeds:
        perm    = torch.randperm(new_xyz.shape[0])[:max_seeds]
        new_xyz   = new_xyz[perm]
        donor_idx = donor_idx[perm]

    n_to_seed = new_xyz.shape[0]
    new_tensors = {
        "xyz":      new_xyz.cuda(),
        "f_dc":     bg._features_dc[donor_idx].detach().clone().cuda(),
        "f_rest":   bg._features_rest[donor_idx].detach().clone().cuda(),
        "opacity":  bg._opacity[donor_idx].detach().clone().cuda(),
        "scaling":  bg._scaling[donor_idx].detach().clone().cuda(),
        "rotation": bg._rotation[donor_idx].detach().clone().cuda(),
        "semantic": bg._semantic[donor_idx].detach().clone().cuda(),
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
    iteration: int,
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

    If ``cfg.train.epochs`` is set, the loop is **epoch-based**: each epoch
    iterates through every training camera exactly once (shuffled).  Otherwise
    the legacy iteration-based loop with random sampling is used.

    All densification / LR-schedule thresholds still use the global iteration
    counter so existing configs remain compatible.
    """
    training_args = cfg.train
    optim_args    = cfg.optim
    data_args     = cfg.data

    start_iter = 0
    tb_writer  = prepare_output_and_logger()
    csv_logger = MetricCSVLogger(cfg.model_path)

    # ── Data & model ──────────────────────────────────────────────────
    dataset   = GoPro360Dataset()
    gaussians = StreetGaussianModel(dataset.scene_info.metadata)
    scene     = Scene(gaussians=gaussians, dataset=dataset)
    gaussians.training_setup()

    # ── Resume from checkpoint ────────────────────────────────────────
    try:
        loaded_iter = (searchForMaxIteration(cfg.trained_model_dir)
                       if cfg.loaded_iter == -1 else cfg.loaded_iter)
        ckpt_path = os.path.join(
            cfg.trained_model_dir, f"iteration_{loaded_iter}.pth"
        )
        state = torch.load(ckpt_path)
        start_iter = state["iter"]
        print(f"Resuming from {ckpt_path}  (iter {start_iter})")
        gaussians.load_state_dict(state)
    except Exception:
        pass

    print(f"Starting from iteration {start_iter}")
    save_cfg(cfg, cfg.model_path, epoch=start_iter)

    renderer = StreetGaussianRenderer()

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end   = torch.cuda.Event(enable_timing=True)

    ema_loss = 0.0
    ema_psnr = 0.0
    ema_ssim = 0.0

    # ── Determine loop bounds ─────────────────────────────────────────
    train_cameras = scene.getTrainCameras()
    cams_per_epoch = len(train_cameras)

    num_epochs  = training_args.get("epochs", 0)
    use_epochs  = num_epochs > 0

    if use_epochs:
        total_iters = num_epochs * cams_per_epoch
        print(f"Epoch-based training: {num_epochs} epochs × "
              f"{cams_per_epoch} cameras = {total_iters} iterations")
    else:
        total_iters = training_args.iterations
        print(f"Iteration-based training: {total_iters} iterations "
              f"({total_iters / cams_per_epoch:.1f} epochs)")

    progress = tqdm(range(start_iter, total_iters), initial=start_iter,
                    total=total_iters)
    start_iter += 1

    # Build initial shuffled stack (will be refilled each epoch)
    viewpoint_stack: list = []

    # ── Multi-GPU setup ───────────────────────────────────────────────
    _n_gpus = torch.cuda.device_count()
    if _n_gpus > 1:
        _executor = ThreadPoolExecutor(max_workers=_n_gpus - 1)
        _roof_mask_dir = getattr(
            data_args, 'roof_mask_dir', 'colmap/output/masks_roof_depth')
        print(f"Multi-GPU training: {_n_gpus} GPUs, "
              f"{_n_gpus} views per iteration (gradient accumulation)")
    else:
        _executor = None

    for iteration in range(start_iter, total_iters + 1):

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # ── Get training camera (epoch-aware) ─────────────────────────
        if not viewpoint_stack:
            viewpoint_stack = list(train_cameras)
            shuffle(viewpoint_stack)

        if use_epochs:
            cam: Camera = viewpoint_stack.pop(0)      # sequential within epoch
        else:
            cam: Camera = viewpoint_stack.pop(         # random (legacy)
                randint(0, len(viewpoint_stack) - 1)
            )

        # ── Launch secondary GPU work (multi-GPU) ─────────────────
        secondary_futures = []
        if _executor is not None:
            for gpu_id in range(1, _n_gpus):
                if not viewpoint_stack:
                    viewpoint_stack = list(train_cameras)
                    shuffle(viewpoint_stack)
                if use_epochs:
                    sec_cam = viewpoint_stack.pop(0)
                else:
                    sec_cam = viewpoint_stack.pop(
                        randint(0, len(viewpoint_stack) - 1))
                secondary_futures.append(_executor.submit(
                    _forward_loss_backward_secondary,
                    f'cuda:{gpu_id}', sec_cam, gaussians,
                    optim_args, iteration, _roof_mask_dir))

        gt_image = cam.original_image
        gt_image = gt_image.cuda(non_blocking=True) if not gt_image.is_cuda else gt_image

        # ── Mask (sky + roof) ─────────────────────────────────────────
        if "mask" in cam.guidance:
            mask = cam.guidance["mask"]
            mask = mask.cuda(non_blocking=True) if not mask.is_cuda else mask
        else:
            mask = None

        # ── Roof-only mask (for separating sky from roof) ─────────────
        roof_mask_dir = getattr(
            data_args, 'roof_mask_dir', 'colmap/output/masks_roof_depth')
        roof_mask_2d = _load_roof_mask(
            cam.image_name + ".png", roof_mask_dir)
        if roof_mask_2d is None:
            roof_mask_2d = _load_roof_mask(cam.image_name, roof_mask_dir)
        if roof_mask_2d is not None:
            roof_mask_2d = roof_mask_2d.to(gt_image.device)  # (H, W) True=roof
            # Resize to match training image if needed
            rH, rW = roof_mask_2d.shape
            iH, iW = gt_image.shape[1], gt_image.shape[2]
            if rH != iH or rW != iW:
                roof_mask_2d = F.interpolate(
                    roof_mask_2d.float().unsqueeze(0).unsqueeze(0),
                    size=(iH, iW), mode='nearest'
                )[0, 0] > 0.5
        else:
            roof_mask_2d = None

        # ── Render ────────────────────────────────────────────────────
        render_pkg = renderer.render(cam, gaussians)
        image = render_pkg["rgb"]
        acc   = render_pkg["acc"]
        depth = render_pkg["depth"]
        viewspace_pts = render_pkg["viewspace_points"]
        visibility    = render_pkg["visibility_filter"]
        radii         = render_pkg["radii"]

        scalar_dict: dict = {}

        # ── RGB loss (L1 + D-SSIM) ───────────────────────────────────
        lambda_l1 = getattr(optim_args, "lambda_l1", 1.0)
        Ll1 = l1_loss(image, gt_image, mask)
        scalar_dict["l1_loss"] = Ll1.item()

        loss = (
            (1.0 - optim_args.lambda_dssim) * lambda_l1 * Ll1
            + optim_args.lambda_dssim * (1.0 - ssim(image, gt_image, mask=mask))
        )

        # ── SH higher-order regularisation ────────────────────────────
        lambda_sh = getattr(optim_args, "lambda_sh_reg", 1e-3)
        if lambda_sh > 0:
            sh_rest = gaussians.background._features_rest
            sh_reg = lambda_sh * (sh_rest ** 2).mean()
            scalar_dict["sh_reg_loss"] = sh_reg.item()
            loss += sh_reg

        # ── Sky opacity loss: acc should be 0 where sky (NOT roof) ──
        # The combined mask has mask=0 for both sky and roof.  We must
        # NOT penalise accumulation in the roof region — that's where
        # seeded road Gaussians live and need to keep their opacity.
        lambda_sky_acc = getattr(optim_args, "lambda_sky_acc", 1e-2)
        if lambda_sky_acc > 0 and mask is not None:
            if roof_mask_2d is not None:
                # sky-only = masked in combined mask AND not roof
                sky_only = (1 - mask.float()) * (1 - roof_mask_2d.float())
            else:
                sky_only = 1 - mask.float()
            sky_acc_loss = lambda_sky_acc * (acc * sky_only).mean()
            scalar_dict["sky_acc_loss"] = sky_acc_loss.item()
            loss += sky_acc_loss

        # ── Depth ranking (monotonicity) loss ─────────────────────────
        # Depth Anything V2 outputs disparity-like values (larger = closer),
        # so we invert (1/d) to get a pseudo-depth with larger = farther,
        # matching the Gaussian renderer's depth convention.
        # lambda_depth_rank = getattr(optim_args, "lambda_depth_rank", 1e-4)
        # depth_rank_warmup = getattr(optim_args, "depth_rank_warmup", 1000)
        # if (lambda_depth_rank > 0
        #         and iteration > depth_rank_warmup
        #         and "lidar_depth" in cam.guidance):
        #     gt_depth = cam.guidance["lidar_depth"]
        #     gt_depth = gt_depth.cuda(non_blocking=True) if not gt_depth.is_cuda else gt_depth
        #     # Invert: DA V2 gives disparity (large = close) → 1/d = depth (large = far)
        #     gt_depth = 1.0 / gt_depth.clamp(min=1e-3)
        #     # valid = non-sky & finite GT depth
        #     valid = torch.isfinite(gt_depth) & (gt_depth > 0)
        #     if mask is not None:
        #         valid = valid & (mask > 0.5)
        #     valid_idx = valid.flatten().nonzero(as_tuple=False).squeeze(-1)
        #     n_pairs = min(1024, len(valid_idx) // 2)
        #     if n_pairs > 0:
        #         perm = torch.randperm(len(valid_idx), device=valid_idx.device)[:n_pairs * 2]
        #         idx = valid_idx[perm].view(2, n_pairs)
        #         gt_flat = gt_depth.flatten()
        #         pred_flat = depth.flatten()
        #         gt_diff = gt_flat[idx[0]] - gt_flat[idx[1]]
        #         pred_diff = pred_flat[idx[0]] - pred_flat[idx[1]]
        #         # hinge: penalise when predicted ordering disagrees with GT
        #         depth_rank_loss = lambda_depth_rank * torch.relu(-gt_diff * pred_diff).mean()
        #         scalar_dict["depth_rank_loss"] = depth_rank_loss.item()
        #         loss += depth_rank_loss

        # ── Colour-correction regularisation ──────────────────────────
        lambda_cc = getattr(optim_args, "lambda_color_correction", 0.0)
        if lambda_cc > 0 and getattr(gaussians, "use_color_correction", False):
            cc_loss = gaussians.color_correction.regularization_loss(cam)
            scalar_dict["color_correction_reg_loss"] = cc_loss.item()
            loss += lambda_cc * cc_loss

        scalar_dict["loss"] = loss.item()

        # ── Compute SSIM for logging (detached) ──────────────────────
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

        # ── Save log images (every 1 000 iterations) ─────────────────
        if iteration % 1000 == 0:
            save_log_images(iteration, gt_image, image, depth, acc)

        # ── Book-keeping (no grad) ───────────────────────────────────
        with torch.no_grad():
            tensor_dict: dict = {}

            cur_psnr = psnr(image, gt_image, mask).mean().float()
            scalar_dict["psnr"] = cur_psnr.item()

            if iteration % 10 == 0:
                ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
                ema_psnr = 0.4 * cur_psnr.item() + 0.6 * ema_psnr
                ema_ssim = 0.4 * ssim_val + 0.6 * ema_ssim

                epoch_num = (iteration - 1) // cams_per_epoch + 1
                progress.set_postfix({
                    "Exp":   f"{cfg.task}-{cfg.exp_name}",
                    "Epoch": f"{epoch_num}",
                    "Loss":  f"{ema_loss:.7f}",
                    "PSNR":  f"{ema_psnr:.4f}",
                    "SSIM":  f"{ema_ssim:.4f}",
                })
            progress.update(1)

            # ── CSV logging (every 10 iterations) ─────────────────────
            if iteration % 10 == 0:
                csv_logger.log_train(iteration, scalar_dict, {
                    "ema_loss": ema_loss,
                    "ema_psnr": ema_psnr,
                    "ema_ssim": ema_ssim,
                })

            # ── Save PLY snapshot ─────────────────────────────────────
            if iteration in training_args.save_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            # ── Adaptive density control ──────────────────────────────
            if iteration < optim_args.densify_until_iter:
                gaussians.set_visibility(
                    include_list=list(
                        set(gaussians.model_name_id.keys()) - {"sky"}
                    )
                )
                gaussians.set_max_radii2D(radii, visibility)
                gaussians.add_densification_stats(viewspace_pts, visibility)

                prune_big = iteration > optim_args.opacity_reset_interval
                if (iteration > optim_args.densify_from_iter
                        and iteration % optim_args.densification_interval == 0):
                    s, t = gaussians.densify_and_prune(
                        max_grad=optim_args.densify_grad_threshold,
                        min_opacity=optim_args.min_opacity,
                        prune_big_points=prune_big,
                    )
                    scalar_dict.update(s)
                    tensor_dict.update(t)

            if iteration < optim_args.densify_until_iter:
                if iteration % optim_args.opacity_reset_interval == 0:
                    gaussians.reset_opacity()
                if (data_args.white_background
                        and iteration == optim_args.densify_from_iter):
                    gaussians.reset_opacity()

            # ── Periodic low-opacity pruning (after densification) ─
            # Reclaims memory from seeded Gaussians that didn't converge.
            prune_interval = getattr(optim_args, 'prune_interval', 5000)
            if (iteration > optim_args.densify_until_iter
                    and prune_interval > 0
                    and iteration % prune_interval == 0):
                prune_min_op = getattr(
                    optim_args, 'prune_min_opacity', 0.005)
                bg_model = gaussians.background
                prune_mask = (
                    bg_model.get_opacity < prune_min_op
                ).squeeze()
                n_before = bg_model._xyz.shape[0]
                if prune_mask.any():
                    bg_model.prune_points(prune_mask)
                    torch.cuda.empty_cache()
                    n_after = bg_model._xyz.shape[0]
                    print(f"\n[ITER {iteration}] Pruned "
                          f"{n_before - n_after:,} low-opacity "
                          f"Gaussians (total: {n_after:,})")

            # ── Seed Gaussians in the roof-hole region ────────────────
            # Runs independently of densify_until_iter so seeds keep
            # being injected through the first half of training.
            hole_seed_interval = getattr(
                optim_args, 'hole_seed_interval', 1000)
            hole_seed_until = getattr(
                optim_args, 'hole_seed_until_iter',
                min(50000,
                    max(optim_args.densify_until_iter, total_iters // 4)))
            just_reset_opacity = (
                iteration < optim_args.densify_until_iter
                and iteration % optim_args.opacity_reset_interval == 0
            )
            should_seed = (
                hole_seed_interval > 0
                and iteration >= 500
                and iteration < hole_seed_until
                and (iteration % hole_seed_interval == 0
                     or just_reset_opacity)
            )
            if should_seed:
                roof_mask_dir = getattr(
                    data_args, 'roof_mask_dir',
                    'colmap/output/masks_roof_depth')
                n_seeded = seed_hole_gaussians(
                    gaussians, train_cameras,
                    max_seeds=getattr(
                        optim_args, 'hole_max_seeds', 70000),
                    roof_mask_dir=roof_mask_dir,
                    max_total_gaussians=getattr(
                        optim_args, 'max_total_gaussians', 10_000_000),
                )
                n_total = gaussians.background._xyz.shape[0]
                if n_seeded > 0:
                    scalar_dict["hole_seeds"] = n_seeded
                    scalar_dict["total_gaussians"] = n_total
                    print(f"\n[ITER {iteration}] Seeded "
                          f"{n_seeded} Gaussians in roof hole "
                          f"(total: {n_total:,})")
                    torch.cuda.empty_cache()

            # ── TensorBoard & evaluation ──────────────────────────────
            training_report(
                tb_writer, iteration,
                scalar_dict, tensor_dict,
                training_args.test_iterations,
                scene, renderer,                csv_logger=csv_logger,            )

            # ── Optimiser step ────────────────────────────────────────
            if iteration < total_iters:
                gaussians.update_optimizer()

            # ── Save checkpoint ───────────────────────────────────────
            # Save at configured iterations AND at end of training
            should_save = (iteration in training_args.checkpoint_iterations
                           or iteration == total_iters)
            if should_save:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                sd = gaussians.save_state_dict(
                    is_final=(iteration == total_iters)
                )
                sd["iter"] = iteration
                ckpt_path = os.path.join(
                    cfg.trained_model_dir, f"iteration_{iteration}.pth"
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
