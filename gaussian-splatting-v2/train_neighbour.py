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
from PIL import Image
import numpy as np
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    # ── Custom train/test split: image_id % 8 == 0 → test ───────────────
    # image_id is parsed from the numeric prefix of image_name (e.g.
    # "0008_front" -> 8). Pulls cameras from both train and test buckets
    # of Scene so behavior is independent of --eval / --llffhold.
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

    # ── LPIPS network (instantiated once, reused for eval) ──────────────
    lpips_model = LPIPS(net_type='vgg').to("cuda").eval()
    for p in lpips_model.parameters():
        p.requires_grad_(False)

    viewpoint_stack = train_cams.copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    # Depth-supervision reliability breakdown (one-time, set at Camera ctor).
    n_total = len(train_cams)
    n_with_depth = sum(1 for c in train_cams if getattr(c, "invdepthmap", None) is not None)
    n_reliable = sum(1 for c in train_cams if getattr(c, "depth_reliable", False))
    pct = (100.0 * n_reliable / n_total) if n_total else 0.0
    print(f"[depth-reg] train viewpoints: total={n_total}  with_depth_loaded={n_with_depth}  "
          f"depth_reliable={n_reliable} ({pct:.1f}%)  "
          f"depth_l1_weight(init→final)={opt.depth_l1_weight_init}→{opt.depth_l1_weight_final}")

    # Runtime counters: confirm how often the depth-loss branch actually fires.
    depth_branch_taken = 0
    depth_branch_skipped = 0

    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
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
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera (from the custom train split)
        if not viewpoint_stack:
            viewpoint_stack = train_cams.copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

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
            invDepth = 1.0 / render_pkg["depth"].clamp(min=1e-6)
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
            depth_branch_taken += 1
        else:
            Ll1depth = 0
            depth_branch_skipped += 1

        loss.backward()

        iter_end.record()
        iter_end.synchronize()  # ensure the GPU has reached this point before querying elapsed time

        with torch.no_grad():
            elapsed_ms = iter_start.elapsed_time(iter_end)
            psnr_live = psnr(image, gt_image).mean().item()
            ssim_live = float(ssim_value.item())

            # Progress-bar EMAs
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            ema_psnr_for_log = 0.4 * psnr_live + 0.6 * ema_psnr_for_log
            ema_ssim_for_log = 0.4 * ssim_live + 0.6 * ema_ssim_for_log

            # Per-iter scalar logging (cheap; throttled to every 10 iters).
            if tb_writer and iteration % 10 == 0:
                seen = depth_branch_taken + depth_branch_skipped
                hit_pct = 100.0 * depth_branch_taken / max(seen, 1)
                gauss_n = scene.gaussians.get_xyz.shape[0]
                tb_writer.add_scalar('train/l1', Ll1.item(), iteration)
                tb_writer.add_scalar('train/ssim', ssim_live, iteration)
                tb_writer.add_scalar('train/psnr', psnr_live, iteration)
                tb_writer.add_scalar('train/total_loss', loss.item(), iteration)
                tb_writer.add_scalar('train/depth_l1', Ll1depth, iteration)
                tb_writer.add_scalar('train/iter_time_ms', elapsed_ms, iteration)
                tb_writer.add_scalar('depth/weight', depth_l1_weight(iteration), iteration)
                tb_writer.add_scalar('depth/hit_pct', hit_pct, iteration)
                tb_writer.add_scalar('gaussians/count', gauss_n, iteration)
                tb_writer.add_scalar('gaussians/sh_degree', scene.gaussians.active_sh_degree, iteration)

            if iteration % 10 == 0:
                seen = depth_branch_taken + depth_branch_skipped
                dpct = (100.0 * depth_branch_taken / seen) if seen else 0.0
                progress_bar.set_postfix({
                    "Loss":  f"{ema_loss_for_log:.5f}",
                    "PSNR":  f"{ema_psnr_for_log:.2f}",
                    "SSIM":  f"{ema_ssim_for_log:.4f}",
                    "DepL":  f"{ema_Ll1depth_for_log:.5f}",
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
            training_report(tb_writer, iteration, train_cams, test_cams, scene, render,
                            (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
                            lpips_model, dataset.train_test_exp, testing_iterations)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

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
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

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

def _depth_to_gray(d):
    """Normalize a depth-like map to a 3-channel grayscale [0,1] tensor for logging.

    Accepts (H,W), (1,H,W) or (C,H,W) input. Uses 2/98-percentile of strictly
    positive pixels to suppress outliers; non-positive / non-finite pixels
    render black.
    """
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
        d_norm = d_norm * valid.float()  # keep sky / invalid pixels black
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


def _make_log_grid(gt_rgb, gt_depth, invmono_depth, pred_rgb, pred_depth, pred_opacity):
    """Assemble a (3, 2H, 3W) composite:
        row 1: GT_RGB     | GT_DEPTH (aligned mono)   | INVMONO_DEPTH (raw mono)
        row 2: PRED_RGB   | PRED_DEPTH                | PRED_OPACITY
    """
    gt_rgb = _to_3ch(gt_rgb)
    row1 = torch.cat([gt_rgb,            _to_3ch(gt_depth),   _to_3ch(invmono_depth)], dim=-1)
    row2 = torch.cat([_to_3ch(pred_rgb), _to_3ch(pred_depth), _to_3ch(pred_opacity)],  dim=-1)
    return torch.cat([row1, row2], dim=-2)


def training_report(tb_writer, iteration, train_cams, test_cams, scene: Scene,
                    renderFunc, renderArgs, lpips_model, train_test_exp,
                    testing_iterations):
    """Run eval pass + composite image logging on the testing_iterations cadence.

    Logs to TensorBoard:
      - {train, test} / {l1, psnr, ssim, lpips} scalars (mean over views)
      - Per-view 3x2 composite grids:
            row 1: GT_RGB | GT_DEPTH | <blank>
            row 2: PRED_RGB | PRED_DEPTH | PRED_OPACITY
      - Opacity histogram + total point count.
    """
    if iteration not in testing_iterations:
        return

    torch.cuda.empty_cache()

    # Train sample: ~5 evenly-spaced views (cheap sanity check during training).
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

    # Args needed to launch an opacity-only render (white SH overridden, black bg).
    # renderArgs is (pipe, background, scaling_modifier, separate_sh, override_color, use_trained_exp).
    pipe_arg, _bg_arg, scaling_mod, separate_sh_arg, _override, _trained_exp = renderArgs
    black_bg = torch.zeros(3, device="cuda")

    for config in validation_configs:
        cams = config['cameras']
        if not cams:
            continue

        l1_sum = psnr_sum = ssim_sum = lpips_sum = 0.0
        n = len(cams)
        for idx, viewpoint in enumerate(cams):
            render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
            rendered = torch.clamp(render_pkg["render"], 0.0, 1.0)
            pred_depth_raw = 1.0 / render_pkg["depth"].clamp(min=1e-6)

            gt = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            if train_test_exp:
                rendered       = rendered[..., rendered.shape[-1] // 2:]
                gt             = gt[..., gt.shape[-1] // 2:]
                pred_depth_raw = pred_depth_raw[..., pred_depth_raw.shape[-1] // 2:]

            l1_sum    += torch.abs(rendered - gt).mean().item()
            psnr_sum  += psnr(rendered, gt).mean().item()
            ssim_sum  += ssim(rendered, gt).item()
            lpips_sum += lpips_model(rendered.unsqueeze(0), gt.unsqueeze(0)).item()

            # GT depth = aligned mono (scale*x + offset, the actual supervision signal).
            if viewpoint.invdepthmap is not None:
                gt_depth_t = viewpoint.invdepthmap
                if train_test_exp:
                    gt_depth_t = gt_depth_t[..., gt_depth_t.shape[-1] // 2:]
            else:
                gt_depth_t = torch.zeros_like(rendered[:1])  # shape already matches

            # invmono depth = raw mono signal straight off disk (pre-alignment).
            invmono_t = getattr(viewpoint, "invmonodepth_raw", None)
            if invmono_t is not None:
                if train_test_exp:
                    invmono_t = invmono_t[..., invmono_t.shape[-1] // 2:]
            else:
                invmono_t = torch.zeros_like(rendered[:1])

            # Opacity render: override color = ones, background = black → accumulated alpha.
            N = scene.gaussians.get_xyz.shape[0]
            white = torch.ones((N, 3), device="cuda")
            opacity_pkg = renderFunc(
                viewpoint, scene.gaussians,
                pipe_arg, black_bg, scaling_mod, separate_sh_arg, white, False,
            )
            pred_opacity_raw = opacity_pkg["render"]
            if train_test_exp:
                pred_opacity_raw = pred_opacity_raw[..., pred_opacity_raw.shape[-1] // 2:]
            pred_opacity = pred_opacity_raw.mean(0, keepdim=True).clamp(0.0, 1.0)

            grid = _make_log_grid(
                gt_rgb        = gt,
                gt_depth      = _depth_to_gray(gt_depth_t),
                invmono_depth = _depth_to_gray(invmono_t),
                pred_rgb      = rendered,
                pred_depth    = _depth_to_gray(pred_depth_raw),
                pred_opacity  = pred_opacity,
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
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=None,
                        help="Iterations at which to run full eval (PSNR/SSIM/LPIPS + image logging). "
                             "Defaults to every 1000 iters up to --iterations.")
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.test_iterations is None:
        args.test_iterations = list(range(1000, args.iterations + 1, 1000))
    if args.iterations not in args.test_iterations:
        args.test_iterations.append(args.iterations)
    args.test_iterations = sorted(set(args.test_iterations))
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
