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
from PIL import Image
import numpy as np
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.sky_model import SkySHModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch.modules.lpips import LPIPS

_ID_RE = re.compile(r"^(\d+)")

def _image_id(cam):
    """Extract the leading numeric id from cam.image_name (e.g. '0008_front' -> 8)."""
    m = _ID_RE.match(cam.image_name)
    return int(m.group(1)) if m else None


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
             sky_sh_degree, lambda_sky_opacity, sky_lr_init, sky_lr_final):

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

            if tb_writer and iteration % 10 == 0:
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

            if iteration % 10 == 0:
                seen = depth_branch_taken + depth_branch_skipped
                dpct = (100.0 * depth_branch_taken / seen) if seen else 0.0
                progress_bar.set_postfix({
                    "Loss":  f"{ema_loss_for_log:.5f}",
                    "PSNR":  f"{ema_psnr_for_log:.2f}",
                    "SSIM":  f"{ema_ssim_for_log:.4f}",
                    "DepL":  f"{ema_Ll1depth_for_log:.5f}",
                    "SkyO":  f"{ema_sky_opa_for_log:.5f}",
                    "Hit%":  f"{dpct:.1f}",
                    "G":     f"{scene.gaussians.get_xyz.shape[0]/1000:.0f}k",
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
                seen = depth_branch_taken + depth_branch_skipped
                dpct = (100.0 * depth_branch_taken / seen) if seen else 0.0
                print(f"[depth-reg] iterations: total={seen}  depth_branch_taken={depth_branch_taken} "
                      f"({dpct:.1f}%)  depth_branch_skipped={depth_branch_skipped}")

            # Periodic eval + image logging
            training_report(tb_writer, iteration, train_cams, test_cams, scene,
                            sky_model, pipe, SPARSE_ADAM_AVAILABLE,
                            lpips_model, dataset.train_test_exp, testing_iterations)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians + sky".format(iteration))
                scene.save(iteration)
                sky_model.save(os.path.join(scene.model_path, f"sky_iter_{iteration}.pth"))

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

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

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                sky_model.save(os.path.join(scene.model_path, f"sky_iter_{iteration}.pth"))

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
                    lpips_model, train_test_exp, testing_iterations):
    if iteration not in testing_iterations:
        return

    torch.cuda.empty_cache()

    n_train_sample = min(5, len(train_cams))
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
        for viewpoint in cams:
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
    parser.add_argument("--sky_sh_degree", type=int, default=3,
                        help="SH degree for the environment sky model (0..3).")
    parser.add_argument("--lambda_sky_opacity", type=float, default=0.05,
                        help="Weight of the sky-mask opacity-suppression loss "
                             "(pushes α → 0 where GT inv-depth == 0).")
    parser.add_argument("--sky_lr_init", type=float, default=1e-2)
    parser.add_argument("--sky_lr_final", type=float, default=1e-4)

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
    )

    print("\nTraining complete.")
