"""
KITTI-360 Gaussian Splatting – Training Script
===============================================

Self-contained training entry point.  All library code lives under
``lib/`` (datasets, models, utils, config).

Usage
-----
    cd MyGaussianSplatting/Gaussians
    python train.py --cfg_file configs/kitti360_drive_0000.yaml

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
from random import randint

import torch
from argparse import Namespace
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════
#  Path setup – make local ``lib`` importable
# ═══════════════════════════════════════════════════════════════════════════
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

# ── Local imports ─────────────────────────────────────────────────────────
from lib.utils.loss_utils import l1_loss, psnr, ssim                    # noqa: E402
from lib.utils.img_utils import save_img_torch, visualize_depth_numpy   # noqa: E402
from lib.models.street_gaussian_renderer import StreetGaussianRenderer  # noqa: E402
from lib.models.street_gaussian_model import StreetGaussianModel        # noqa: E402
from lib.utils.general_utils import safe_state                          # noqa: E402
from lib.utils.camera_utils import Camera                               # noqa: E402
from lib.utils.cfg_utils import save_cfg                                # noqa: E402
from lib.models.scene import Scene                                      # noqa: E402
from lib.datasets.dataset import Dataset                                # noqa: E402
from lib.config import cfg                                              # noqa: E402
from lib.utils.system_utils import searchForMaxIteration                # noqa: E402

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


# ═══════════════════════════════════════════════════════════════════════════
#  Output & Logging Helpers
# ═══════════════════════════════════════════════════════════════════════════

def prepare_output_and_logger():
    """Create the four output directories & return a TensorBoard writer."""
    print(f"Output folder: {cfg.model_path}")
    os.makedirs(cfg.model_path, exist_ok=True)
    os.makedirs(cfg.trained_model_dir, exist_ok=True)
    os.makedirs(cfg.record_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.model_path, "log_images"), exist_ok=True)

    if not cfg.resume:
        for d in (cfg.record_dir, cfg.trained_model_dir):
            p = Path(d)
            if p.exists():
                for f in p.iterdir():
                    f.unlink(missing_ok=True)

    # cfg_args for the SIBR real-time viewer
    with open(os.path.join(cfg.model_path, "cfg_args"), "w") as fp:
        fp.write(str(Namespace(
            sh_degree=cfg.model.gaussian.sh_degree,
            white_background=cfg.data.white_background,
            source_path=cfg.source_path,
            model_path=cfg.model_path,
        )))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(cfg.record_dir)
    else:
        print("TensorBoard not available – training will not be logged.")
    return tb_writer


def training_report(
    tb_writer,
    iteration: int,
    scalar_stats: dict,
    tensor_stats: dict,
    testing_iterations: list,
    scene: Scene,
    renderer: StreetGaussianRenderer,
):
    """Write scalars & histograms to TensorBoard; run test-set eval."""
    if tb_writer:
        try:
            for k, v in scalar_stats.items():
                tb_writer.add_scalar(f"train/{k}", v, iteration)
            for k, v in tensor_stats.items():
                tb_writer.add_histogram(f"train/{k}", v, iteration)
        except Exception:
            print("WARNING: Failed to write to TensorBoard")

    if iteration not in testing_iterations:
        return

    torch.cuda.empty_cache()
    val_cfgs = (
        {"name": "test/test_view",
         "cameras": scene.getTestCameras()},
        {"name": "test/train_view",
         "cameras": [
             scene.getTrainCameras()[i % len(scene.getTrainCameras())]
             for i in range(5, 30, 5)
         ]},
    )
    for vc in val_cfgs:
        cams = vc["cameras"]
        if not cams:
            continue
        l1_tot, psnr_tot = 0.0, 0.0
        for vi, vp in enumerate(cams):
            img = torch.clamp(
                renderer.render(vp, scene.gaussians)["rgb"], 0.0, 1.0
            )
            gt = torch.clamp(vp.original_image.cuda(), 0.0, 1.0)

            if tb_writer and vi < 5:
                tag = f"{vc['name']}_{vp.image_name}"
                tb_writer.add_images(f"{tag}/render", img[None],
                                     global_step=iteration)
                if iteration == testing_iterations[0]:
                    tb_writer.add_images(f"{tag}/ground_truth", gt[None],
                                         global_step=iteration)

            mask = (vp.original_mask.cuda().bool()
                    if hasattr(vp, "original_mask")
                    else torch.ones_like(gt[0]).bool())
            l1_tot  += l1_loss(img, gt, mask).mean().double()
            psnr_tot += psnr(img, gt, mask).mean().double()

        n = len(cams)
        print(f"\n[ITER {iteration}] {vc['name']}: "
              f"L1 {l1_tot / n:.5f}  PSNR {psnr_tot / n:.4f}")
        if tb_writer:
            tb_writer.add_scalar(f"{vc['name']}/l1_loss",  l1_tot / n,  iteration)
            tb_writer.add_scalar(f"{vc['name']}/psnr",     psnr_tot / n, iteration)

    if tb_writer:
        tb_writer.add_histogram(
            "test/opacity_histogram", scene.gaussians.get_opacity, iteration)
        tb_writer.add_scalar(
            "test/points_total", scene.gaussians.get_xyz.shape[0], iteration)
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════
#  Training Loop
# ═══════════════════════════════════════════════════════════════════════════

def training():
    """Main training loop for KITTI-360 Gaussian Splatting.

    Outputs produced:
        * Checkpoints      -> ``trained_model_dir/iteration_<N>.pth``
        * PLY snapshots    -> ``point_cloud_dir/iteration_<N>/``
        * Log images       -> ``model_path/log_images/<iter>.jpg``
        * TensorBoard logs -> ``record_dir/``
    """
    training_args = cfg.train
    optim_args    = cfg.optim
    data_args     = cfg.data

    start_iter = 0
    tb_writer  = prepare_output_and_logger()

    # ── Data & model ──────────────────────────────────────────────────
    dataset   = Dataset()
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
    progress = tqdm(range(start_iter, training_args.iterations))
    start_iter += 1

    viewpoint_stack = None

    for iteration in range(start_iter, training_args.iterations + 1):

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Increase SH band every 1 000 iterations
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # ── Sample a training camera ──────────────────────────────────
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        cam: Camera = viewpoint_stack.pop(
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
        depth = render_pkg["depth"]          # [1, H, W]
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

        # ── LiDAR depth loss ──────────────────────────────────────────
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

        loss.backward()
        iter_end.record()

        # ── Save log images (every 1 000 iterations) ─────────────────
        if iteration % 1000 == 0:
            with torch.no_grad():
                depth_np = depth.detach().cpu().numpy().squeeze(0)
                depth_rgb, _ = visualize_depth_numpy(depth_np)
                depth_rgb = (
                    torch.from_numpy(depth_rgb[..., [2, 1, 0]] / 255.0)
                    .permute(2, 0, 1).float().cuda()
                )
                row0 = torch.cat([gt_image, image, depth_rgb], dim=2)

                acc3 = acc.repeat(3, 1, 1)
                img_obj  = torch.zeros_like(image)
                acc_obj  = torch.zeros_like(acc)
                acc_obj3 = acc_obj.repeat(3, 1, 1)

                row1 = torch.cat([acc3, img_obj, acc_obj3], dim=2)
                vis_img = torch.clamp(torch.cat([row0, row1], dim=1), 0.0, 1.0)

            log_dir = os.path.join(cfg.model_path, "log_images")
            os.makedirs(log_dir, exist_ok=True)
            save_img_torch(vis_img, os.path.join(log_dir, f"{iteration}.jpg"))

        # ── Book-keeping (no grad) ───────────────────────────────────
        with torch.no_grad():
            tensor_dict: dict = {}

            # Progress bar
            if iteration % 10 == 0:
                ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
                ema_psnr = (
                    0.4 * psnr(image, gt_image, mask).mean().float()
                    + 0.6 * ema_psnr
                )
                progress.set_postfix({
                    "Exp":  f"{cfg.task}-{cfg.exp_name}",
                    "Loss": f"{ema_loss:.7f}",
                    "PSNR": f"{ema_psnr:.4f}",
                })
            progress.update(1)

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

            # Reset opacity periodically
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
                scene, renderer,
            )

            # ── Optimiser step ────────────────────────────────────────
            if iteration < training_args.iterations:
                gaussians.update_optimizer()

            # ── Save checkpoint ───────────────────────────────────────
            if iteration in training_args.checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                sd = gaussians.save_state_dict(
                    is_final=(iteration == training_args.iterations)
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

    # Deterministic & safe RNG state
    safe_state(cfg.train.quiet)

    torch.autograd.set_detect_anomaly(cfg.train.detect_anomaly)
    training()

    print("\nTraining complete.")
