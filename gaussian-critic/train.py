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

import os
import re
import csv
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.sky_model import SkyModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
import math
import torch.nn.functional as F
import torchvision
from arguments import ModelParams, PipelineParams, OptimizationParams
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

try:
    from lpipsPyTorch.modules.lpips import LPIPS as LPIPSCriterion
    LPIPS_AVAILABLE = True
except Exception:
    LPIPS_AVAILABLE = False

def _write_csv(csv_path, row_dict):
    """Append one metrics row to a CSV file, writing the header on first write."""
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def _frame_num(image_name):
    """Return the leading integer from an image name (e.g. '0008_front' -> 8)."""
    m = re.match(r'^(\d+)', image_name)
    return int(m.group(1)) if m else -1


def depth_to_vis(depth_map):
    """Normalise a depth/inv-depth tensor (1, H, W) to an RGB visualisation (3, H, W).

    Uses full min-max normalisation so that GT depths remain visible even after
    the COLMAP scale/offset alignment (which can produce negative values for far
    pixels that would otherwise be masked out by a > 0 filter).
    """
    d = depth_map.squeeze(0).float()
    lo, hi = d.min(), d.max()
    if hi > lo:
        d_norm = (d - lo) / (hi - lo + 1e-8)
    else:
        d_norm = torch.zeros_like(d)
    return d_norm.unsqueeze(0).expand(3, -1, -1).clamp(0, 1).contiguous()


def save_log_images(iteration, log_cameras, gaussians, pipe, sky_model,
                    dataset, model_path, SPARSE_ADAM_AVAILABLE):
    log_dir = os.path.join(model_path, "log_images", f"iter_{iteration:06d}")
    os.makedirs(log_dir, exist_ok=True)

    for viewpoint_cam in log_cameras:
        with torch.no_grad():
            H, W = viewpoint_cam.image_height, viewpoint_cam.image_width

            # Render with black background
            bg_black = torch.zeros(3, device="cuda")
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg_black,
                                use_trained_exp=dataset.train_test_exp,
                                separate_sh=SPARSE_ADAM_AVAILABLE)
            render_rgb   = render_pkg["render"].clamp(0, 1)   # (3, H, W)
            render_depth = render_pkg["depth"]                # (1, H, W) or (H, W)
            if render_depth.dim() == 2:
                render_depth = render_depth.unsqueeze(0)

            # ---- sky compositing ----
            if sky_model is not None:
                render_white = render(viewpoint_cam, gaussians, pipe,
                                      torch.ones(3, device="cuda"),
                                      use_trained_exp=dataset.train_test_exp,
                                      separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                unoccupied = (render_white - render_rgb).clamp(0, 1).mean(dim=0, keepdim=True)
                fx = W / (2.0 * math.tan(viewpoint_cam.FoVx * 0.5))
                fy = H / (2.0 * math.tan(viewpoint_cam.FoVy * 0.5))
                ys = torch.arange(H, device="cuda", dtype=torch.float32)
                xs = torch.arange(W, device="cuda", dtype=torch.float32)
                grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
                dirs_cam = torch.stack([
                    (grid_x - W * 0.5 + 0.5) / fx,
                    (grid_y - H * 0.5 + 0.5) / fy,
                    torch.ones_like(grid_x)
                ], dim=-1).view(-1, 3)
                R_c2w = viewpoint_cam.world_view_transform[:3, :3]
                dirs_world = F.normalize(dirs_cam @ R_c2w.T, dim=-1)
                sky_colors = sky_model(dirs_world)
                sky_image  = sky_colors.view(H, W, 3).permute(2, 0, 1).clamp(0, 1)
                composited = (render_rgb + unoccupied * sky_image).clamp(0, 1)
                sky_vis    = sky_image
            else:
                render_white = render(viewpoint_cam, gaussians, pipe,
                                      torch.ones(3, device="cuda"),
                                      use_trained_exp=dataset.train_test_exp,
                                      separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                composited = render_rgb
                sky_vis    = torch.zeros(3, H, W, device="cuda")

            # ---- opacity: fraction of pixel covered by gaussians ----
            opacity_vis = (1.0 - (render_white - render_rgb).clamp(0, 1).mean(dim=0, keepdim=True)
                           ).expand(3, -1, -1).clamp(0, 1).contiguous()

            # ---- GT panels ----
            gt_rgb = viewpoint_cam.original_image.clamp(0, 1)   # (3, H, W)
            if viewpoint_cam.invdepthmap is not None:
                gt_depth_vis = depth_to_vis(viewpoint_cam.invdepthmap)
            else:
                gt_depth_vis = torch.zeros(3, H, W, device="cuda")
            # 3DGS renders metric depth, near = samll value -> dark = far.
            render_depth_vis = depth_to_vis(1.0 / render_depth.clamp(min=1e-6))
            if viewpoint_cam.alpha_mask is not None:
                gt_alpha_vis = viewpoint_cam.alpha_mask.clamp(0, 1).expand(3, -1, -1).contiguous()
            else:
                gt_alpha_vis = torch.zeros(3, H, W, device="cuda")

            # ---- 3 × 3 grid (row-major, nrow=3) ----
            # Row 1: GT RGB        | GT Depth       | GT Alpha mask
            # Row 2: Rendered RGB  | Rendered Depth | Rendered Opacity
            # Row 3: Composited    | SH Sky         | (padded)
            panels = [gt_rgb,     gt_depth_vis,     gt_alpha_vis,
                      render_rgb, render_depth_vis,  opacity_vis,
                      composited, sky_vis]
            grid = torchvision.utils.make_grid(panels, nrow=3, padding=4, pad_value=1.0)
            save_path = os.path.join(log_dir, f"{viewpoint_cam.image_name}.png")
            torchvision.utils.save_image(grid, save_path)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, sky_sh_degree=0, max_frame=-1):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    # Optionally restrict to frames <= max_frame
    def _in_frame_range(cam):
        return max_frame < 0 or _frame_num(cam.image_name) <= max_frame

    train_cameras = [cam for cam in scene.getTrainCameras() if _in_frame_range(cam)]

    # Cameras whose frame number is divisible by 8 are used for log images
    log_cameras = [cam for cam in train_cameras
                   if _frame_num(cam.image_name) > 0 and _frame_num(cam.image_name) % 8 == 0]

    # Sky model: learnable SH environment for background pixels
    if sky_sh_degree > 0:
        sky_model = SkyModel(sky_sh_degree).cuda()
        sky_optimizer = torch.optim.Adam(sky_model.parameters(), lr=opt.feature_lr)
    else:
        sky_model = None
        sky_optimizer = None

    if checkpoint:
        ckpt = torch.load(checkpoint)
        if len(ckpt) == 3:
            model_params, sky_coeffs, first_iter = ckpt
            if sky_model is not None and sky_coeffs is not None:
                sky_model.sh_coeffs.data.copy_(sky_coeffs)
        else:
            model_params, first_iter = ckpt
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = train_cameras.copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    ema_ssim_for_log = 0.0

    lpips_criterion = LPIPSCriterion('vgg').cuda() if LPIPS_AVAILABLE else None

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
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 2000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = train_cameras.copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.zeros(3, device="cuda") if sky_model is not None else (
            torch.rand((3), device="cuda") if opt.random_background else background)

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Sky compositing: replace unoccupied background with learnable SH sky
        if sky_model is not None:
            with torch.no_grad():
                render_white = render(viewpoint_cam, gaussians, pipe, torch.ones(3, device="cuda"),
                                      use_trained_exp=dataset.train_test_exp,
                                      separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
            # unoccupied fraction per pixel: render_white - render_black = (1 - alpha_gauss)
            unoccupied = (render_white - image).clamp(0, 1).mean(dim=0, keepdim=True).detach()
            # Per-pixel world-space ray directions from camera intrinsics
            H, W = viewpoint_cam.image_height, viewpoint_cam.image_width
            fx = W / (2.0 * math.tan(viewpoint_cam.FoVx * 0.5))
            fy = H / (2.0 * math.tan(viewpoint_cam.FoVy * 0.5))
            ys = torch.arange(H, device="cuda", dtype=torch.float32)
            xs = torch.arange(W, device="cuda", dtype=torch.float32)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
            dirs_cam = torch.stack([
                (grid_x - W * 0.5 + 0.5) / fx,
                (grid_y - H * 0.5 + 0.5) / fy,
                torch.ones_like(grid_x)
            ], dim=-1).view(-1, 3)
            # C2W rotation: upper-left 3x3 of world_view_transform (= world_to_cam transposed)
            R_c2w = viewpoint_cam.world_view_transform[:3, :3]
            dirs_world = F.normalize(dirs_cam @ R_c2w.T, dim=-1)  # (H*W, 3)
            sky_colors = sky_model(dirs_world)                     # (H*W, 3)
            sky_image = sky_colors.view(H, W, 3).permute(2, 0, 1) # (3, H, W)
            image = image + unoccupied * sky_image

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log  = 0.4 * loss.item()       + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth       + 0.6 * ema_Ll1depth_for_log
            ema_ssim_for_log  = 0.4 * ssim_value.item() + 0.6 * ema_ssim_for_log

            if iteration % 100 == 0:
                n_gauss = gaussians.get_xyz.shape[0]
                progress_bar.set_postfix({
                    "Loss":   f"{ema_loss_for_log:.6f}",
                    "DepthL": f"{ema_Ll1depth_for_log:.6f}",
                    "SSIM":   f"{ema_ssim_for_log:.4f}",
                    "#GS":    f"{n_gauss/1000:.1f}k",
                    "SH":     gaussians.active_sh_degree,
                })
                progress_bar.update(100)
            if iteration == opt.iterations:
                progress_bar.close()

            # Per-iteration TensorBoard scalars
            if tb_writer:
                tb_writer.add_scalar('train/l1_loss',         Ll1.item(),             iteration)
                tb_writer.add_scalar('train/total_loss',      loss.item(),            iteration)
                tb_writer.add_scalar('train/ssim',            ssim_value.item(),      iteration)
                tb_writer.add_scalar('train/depth_l1_loss',   ema_Ll1depth_for_log,   iteration)
                tb_writer.add_scalar('train/depth_l1_weight', depth_l1_weight(iteration), iteration)
                tb_writer.add_scalar('scene/num_gaussians',   gaussians.get_xyz.shape[0], iteration)
                tb_writer.add_scalar('scene/active_sh_degree', gaussians.active_sh_degree, iteration)
                for pg in gaussians.optimizer.param_groups:
                    tb_writer.add_scalar(f'lr/{pg["name"]}', pg['lr'], iteration)
                if sky_model is not None:
                    tb_writer.add_scalar('sky/sh_coeff_norm', sky_model.sh_coeffs.norm().item(), iteration)

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render,
                            (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
                            dataset.train_test_exp, sky_model=sky_model, lpips_criterion=lpips_criterion)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Log diagnostic images every 1000 iterations
            if iteration % 1000 == 0 and log_cameras:
                save_log_images(iteration, log_cameras, gaussians, pipe, sky_model,
                                dataset, scene.model_path, SPARSE_ADAM_AVAILABLE)

            # Quick train-subset metrics every 1000 iterations
            if iteration % 1000 == 0 and log_cameras and tb_writer:
                psnr_sum = ssim_sum = lpips_sum = 0.0
                n = len(log_cameras)
                for viewpoint in log_cameras:
                    img = torch.clamp(render(viewpoint, gaussians, pipe, background,
                                             use_trained_exp=dataset.train_test_exp,
                                             separate_sh=SPARSE_ADAM_AVAILABLE)["render"], 0.0, 1.0)
                    gt = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    psnr_sum += psnr(img, gt).mean().double()
                    if FUSED_SSIM_AVAILABLE:
                        ssim_sum += fused_ssim(img.unsqueeze(0), gt.unsqueeze(0)).double()
                    else:
                        ssim_sum += ssim(img, gt).mean().double()
                    if lpips_criterion is not None:
                        lpips_sum += lpips_criterion(img.unsqueeze(0), gt.unsqueeze(0)).item()
                tb_writer.add_scalar('train_quick/psnr', float(psnr_sum / n), iteration)
                tb_writer.add_scalar('train_quick/ssim', float(ssim_sum / n), iteration)
                if lpips_criterion is not None:
                    tb_writer.add_scalar('train_quick/lpips', float(lpips_sum / n), iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if sky_optimizer is not None:
                    sky_optimizer.step()
                    sky_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                sky_coeffs_ckpt = sky_model.sh_coeffs.data if sky_model is not None else None
                torch.save((gaussians.capture(), sky_coeffs_ckpt, iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations,
                    scene: Scene, renderFunc, renderArgs, train_test_exp,
                    sky_model=None, lpips_criterion=None):
    if tb_writer:
        tb_writer.add_scalar('train/iter_time_ms', elapsed, iteration)

    # Full evaluation at designated iterations
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        csv_path = os.path.join(scene.model_path, "metrics.csv")
        validation_configs = (
            {'name': 'test',  'cameras': scene.getTestCameras()},
            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                                          for idx in range(5, 30, 5)]}
        )

        for config in validation_configs:
            if not config['cameras']:
                continue
            l1_sum = psnr_sum = ssim_sum = lpips_sum = 0.0
            n = len(config['cameras'])

            for idx, viewpoint in enumerate(config['cameras']):
                image    = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                if train_test_exp:
                    image    = image[...,    image.shape[-1] // 2:]
                    gt_image = gt_image[..., gt_image.shape[-1] // 2:]

                if tb_writer and idx < 5:
                    tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/render",
                                         image[None], global_step=iteration)
                    if iteration == testing_iterations[0]:
                        tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/ground_truth",
                                             gt_image[None], global_step=iteration)

                l1_sum   += l1_loss(image, gt_image).mean().double()
                psnr_sum += psnr(image, gt_image).mean().double()
                ssim_sum += ssim(image, gt_image).mean().double()
                if lpips_criterion is not None:
                    lpips_sum += lpips_criterion(image.unsqueeze(0), gt_image.unsqueeze(0)).item()

            l1_avg    = float(l1_sum   / n)
            psnr_avg  = float(psnr_sum / n)
            ssim_avg  = float(ssim_sum / n)
            lpips_avg = float(lpips_sum / n) if lpips_criterion is not None else float('nan')

            print("\n[ITER {:6d}] Evaluating {:5s}: "
                  "L1 {:.5f} | PSNR {:.3f} dB | SSIM {:.4f} | LPIPS {:.4f}".format(
                  iteration, config['name'], l1_avg, psnr_avg, ssim_avg, lpips_avg))

            if tb_writer:
                tb_writer.add_scalar(f"{config['name']}/l1",    l1_avg,   iteration)
                tb_writer.add_scalar(f"{config['name']}/psnr",  psnr_avg, iteration)
                tb_writer.add_scalar(f"{config['name']}/ssim",  ssim_avg, iteration)
                if lpips_criterion is not None:
                    tb_writer.add_scalar(f"{config['name']}/lpips", lpips_avg, iteration)

            _write_csv(csv_path, {
                'iteration': iteration,
                'split':     config['name'],
                'l1':        l1_avg,
                'psnr':      psnr_avg,
                'ssim':      ssim_avg,
                'lpips':     lpips_avg,
                'num_gaussians': scene.gaussians.get_xyz.shape[0],
            })

        if tb_writer:
            tb_writer.add_histogram('scene/opacity_histogram', scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('scene/num_gaussians', scene.gaussians.get_xyz.shape[0], iteration)
            if sky_model is not None:
                tb_writer.add_scalar('sky/sh_coeff_norm', sky_model.sh_coeffs.norm().item(), iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--sky_sh_degree", type=int, default=0,
                        help="SH degree for learnable sky model (0 = disabled, 3 = recommended)")
    parser.add_argument("--max_frame", type=int, default=30,
                        help="Only use images with frame number <= max_frame (-1 = all frames)")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.sky_sh_degree, args.max_frame)

    # All done
    print("\nTraining complete.")
