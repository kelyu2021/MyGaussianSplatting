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

import torch
import numpy as np
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

        gt_image = cam.original_image
        gt_image = gt_image.cuda(non_blocking=True) if not gt_image.is_cuda else gt_image

        mask = (cam.guidance["mask"]
                if "mask" in cam.guidance
                else torch.ones_like(gt_image[0:1]).bool())
        mask = mask.cuda(non_blocking=True) if not mask.is_cuda else mask

        sky_mask = lidar_depth = obj_bound = None
        if "sky_mask" in cam.guidance:
            sky_mask = cam.guidance["sky_mask"]
            sky_mask = sky_mask.cuda(non_blocking=True) if not sky_mask.is_cuda else sky_mask
        if "lidar_depth" in cam.guidance:
            lidar_depth = cam.guidance["lidar_depth"]
            lidar_depth = lidar_depth.cuda(non_blocking=True) if not lidar_depth.is_cuda else lidar_depth
        if "obj_bound" in cam.guidance:
            obj_bound = cam.guidance["obj_bound"]
            obj_bound = obj_bound.cuda(non_blocking=True) if not obj_bound.is_cuda else obj_bound

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

        # ── Sky loss ──────────────────────────────────────────────────
        if (optim_args.lambda_sky > 0
                and gaussians.include_sky
                and sky_mask is not None):
            acc_clamped = torch.clamp(acc, min=1e-6, max=1.0 - 1e-6)
            sky_loss = torch.where(
                sky_mask,
                -torch.log(1.0 - acc_clamped),
                -torch.log(acc_clamped),
            ).mean()
            sky_scale = getattr(optim_args, "lambda_sky_scale", [])
            if len(sky_scale) > 0:
                sky_loss *= sky_scale[cam.meta["cam"]]
            scalar_dict["sky_loss"] = sky_loss.item()
            loss += optim_args.lambda_sky * sky_loss

        # ── LiDAR depth loss (not used for GoPro 360) ────────────────
        if optim_args.lambda_depth_lidar > 0 and lidar_depth is not None:
            depth_mask = torch.logical_and(lidar_depth > 0.0, mask)
            expected_depth = depth / (acc + 1e-10)
            d_err = torch.abs(expected_depth[depth_mask] - lidar_depth[depth_mask])
            d_err, _ = torch.topk(d_err, int(0.95 * d_err.numel()), largest=False)
            lidar_loss = d_err.mean()
            scalar_dict["lidar_depth_loss"] = lidar_loss.item()
            loss += optim_args.lambda_depth_lidar * lidar_loss

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
                ema_ssim = 0.4 * ssim_val.item() + 0.6 * ema_ssim

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
