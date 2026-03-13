"""
Training-related utilities for GoPro 360 Gaussian Splatting.

Provides output-directory setup, TensorBoard writer creation,
CSV metric logging, and the ``training_report`` function used
by the main training loop.
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from argparse import Namespace

import torch

from lib.config import cfg
from lib.utils.loss_utils import l1_loss, psnr, ssim
from lib.utils.img_utils import save_img_torch, visualize_depth_numpy
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.models.scene import Scene

try:
    import lpips as _lpips_module
    _LPIPS_FN = _lpips_module.LPIPS(net='vgg').cuda().eval()
    LPIPS_AVAILABLE = True
except Exception:
    _LPIPS_FN = None
    LPIPS_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


# ═══════════════════════════════════════════════════════════════════════════
#  CSV Metric Logger
# ═══════════════════════════════════════════════════════════════════════════

class MetricCSVLogger:
    """Append-only CSV logger for training losses and evaluation metrics.

    Creates two CSV files under ``cfg.model_path``:
      - ``train_metrics.csv``  – per-iteration training scalars
      - ``eval_metrics.csv``   – per-evaluation-iteration test/train-view metrics

    These files can be loaded by ``visualize_metrics.py`` or any plotting tool.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._train_path = os.path.join(model_path, "train_metrics.csv")
        self._eval_path = os.path.join(model_path, "eval_metrics.csv")
        self._train_file = None
        self._eval_file = None
        self._train_writer = None
        self._eval_writer = None
        self._train_header_written = False
        self._eval_header_written = False

    # -- Training scalars ---------------------------------------------------

    def log_train(self, iteration: int, scalar_dict: dict, extras: dict | None = None):
        """Append one row of training scalars."""
        row = {"iteration": iteration, **scalar_dict}
        if extras:
            row.update(extras)

        if self._train_writer is None:
            self._train_file = open(self._train_path, "a", newline="")
            self._train_writer = csv.DictWriter(
                self._train_file, fieldnames=list(row.keys()), extrasaction="ignore"
            )
            if os.path.getsize(self._train_path) == 0:
                self._train_writer.writeheader()
                self._train_header_written = True
            else:
                self._train_header_written = True
        self._train_writer.writerow(row)
        self._train_file.flush()

    # -- Evaluation scalars -------------------------------------------------

    def log_eval(self, iteration: int, split: str, l1: float, psnr_val: float,
                 ssim_val: float, n_points: int, lpips_val: float = 0.0):
        """Append one row of evaluation metrics."""
        row = {
            "iteration": iteration,
            "split": split,
            "l1_loss": l1,
            "psnr": psnr_val,
            "ssim": ssim_val,
            "lpips": lpips_val,
            "n_points": n_points,
        }
        if self._eval_writer is None:
            self._eval_file = open(self._eval_path, "a", newline="")
            fieldnames = list(row.keys())
            self._eval_writer = csv.DictWriter(
                self._eval_file, fieldnames=fieldnames, extrasaction="ignore"
            )
            if os.path.getsize(self._eval_path) == 0:
                self._eval_writer.writeheader()
        self._eval_writer.writerow(row)
        self._eval_file.flush()

    def close(self):
        if self._train_file:
            self._train_file.close()
        if self._eval_file:
            self._eval_file.close()


# ═══════════════════════════════════════════════════════════════════════════
#  Output & Logger
# ═══════════════════════════════════════════════════════════════════════════

def prepare_output_and_logger():
    """Create the output directories and return a TensorBoard ``SummaryWriter``.

    Directories created
    -------------------
    * ``cfg.model_path``
    * ``cfg.trained_model_dir``
    * ``cfg.record_dir``
    * ``{cfg.model_path}/log_images``
    """
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


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluation / Reporting
# ═══════════════════════════════════════════════════════════════════════════

def training_report(
    tb_writer,
    iteration: int,
    scalar_stats: dict,
    tensor_stats: dict,
    testing_iterations: list,
    scene: Scene,
    renderer: StreetGaussianRenderer,
    csv_logger: MetricCSVLogger | None = None,
):
    """Write scalars / histograms to TensorBoard and run test-set eval.

    Parameters
    ----------
    tb_writer : SummaryWriter | None
    iteration : int
    scalar_stats : dict
        Scalar losses to log under ``train/``.
    tensor_stats : dict
        Tensor histograms to log under ``train/``.
    testing_iterations : list[int]
        Iterations at which to run full test-set evaluation.
    scene : Scene
    renderer : StreetGaussianRenderer
    csv_logger : MetricCSVLogger | None
        Optional CSV logger for offline plotting.
    """
    # ── Write training scalars every call ─────────────────────────────
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

    # ── Full test-set evaluation ──────────────────────────────────────
    torch.cuda.empty_cache()
    n_points = scene.gaussians.get_xyz.shape[0]

    val_cfgs = (
        {"name": "test/test_view",  "split": "test",
         "cameras": scene.getTestCameras()},
        {"name": "test/train_view", "split": "train",
         "cameras": [
             scene.getTrainCameras()[i % len(scene.getTrainCameras())]
             for i in range(5, 30, 5)
         ]},
    )
    for vc in val_cfgs:
        cams = vc["cameras"]
        if not cams:
            continue
        l1_tot, psnr_tot, ssim_tot, lpips_tot = 0.0, 0.0, 0.0, 0.0
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

            mask = (vp.guidance["mask"]
                    if "mask" in vp.guidance
                    else torch.ones_like(gt[0]).bool())
            mask = mask.cuda(non_blocking=True) if not mask.is_cuda else mask

            # ── Save test images to disk ──────────────────────────────
            save_dir = os.path.join(
                cfg.model_path, "test_images",
                f"iteration_{iteration}", vc["split"],
            )
            os.makedirs(save_dir, exist_ok=True)
            name = vp.image_name
            save_img_torch(img, os.path.join(save_dir, f"{name}_render.png"))
            save_img_torch(gt,  os.path.join(save_dir, f"{name}_gt.png"))
            # mask is [H,W] bool → expand to 3-channel float for saving
            mask_vis = mask.float().unsqueeze(0).expand(3, -1, -1)
            save_img_torch(mask_vis, os.path.join(save_dir, f"{name}_mask.png"))

            l1_tot   += l1_loss(img, gt, mask).mean().double()
            psnr_tot += psnr(img, gt, mask).mean().double()
            ssim_tot += ssim(img, gt, mask=mask).mean().double()

            # LPIPS (perceptual distance)
            if LPIPS_AVAILABLE and _LPIPS_FN is not None:
                with torch.no_grad():
                    # lpips expects images in [-1, 1]
                    lpips_val = _LPIPS_FN(
                        img.unsqueeze(0) * 2.0 - 1.0,
                        gt.unsqueeze(0) * 2.0 - 1.0,
                    ).item()
                    lpips_tot += lpips_val

        n = len(cams)
        avg_l1   = (l1_tot / n).item()
        avg_psnr = (psnr_tot / n).item()
        avg_ssim = (ssim_tot / n).item()
        avg_lpips = lpips_tot / n if n > 0 else 0.0

        print(f"\n[ITER {iteration}] {vc['name']}: "
              f"L1 {avg_l1:.5f}  PSNR {avg_psnr:.4f}  SSIM {avg_ssim:.4f}  "
              f"LPIPS {avg_lpips:.4f}")
        if tb_writer:
            tb_writer.add_scalar(f"{vc['name']}/l1_loss", avg_l1,   iteration)
            tb_writer.add_scalar(f"{vc['name']}/psnr",    avg_psnr, iteration)
            tb_writer.add_scalar(f"{vc['name']}/ssim",    avg_ssim, iteration)
            tb_writer.add_scalar(f"{vc['name']}/lpips",   avg_lpips, iteration)

        if csv_logger:
            csv_logger.log_eval(iteration, vc["split"], avg_l1, avg_psnr,
                                avg_ssim, n_points, avg_lpips)

    if tb_writer:
        tb_writer.add_histogram(
            "test/opacity_histogram", scene.gaussians.get_opacity, iteration)
        tb_writer.add_scalar(
            "test/points_total", n_points, iteration)
    torch.cuda.empty_cache()


def save_log_images(
    iteration: int,
    gt_image: torch.Tensor,
    image: torch.Tensor,
    depth: torch.Tensor,
    acc: torch.Tensor,
):
    """Save a 2-row visualisation grid to ``{model_path}/log_images/``.

    Row 0: GT | Render | Depth-colourmap
    Row 1: Acc | zeros  | zeros
    """
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
        acc_obj3 = torch.zeros_like(acc).repeat(3, 1, 1)

        row1 = torch.cat([acc3, img_obj, acc_obj3], dim=2)
        vis_img = torch.clamp(torch.cat([row0, row1], dim=1), 0.0, 1.0)

    log_dir = os.path.join(cfg.model_path, "log_images")
    os.makedirs(log_dir, exist_ok=True)
    save_img_torch(vis_img, os.path.join(log_dir, f"{iteration}.jpg"))
