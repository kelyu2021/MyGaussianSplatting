"""
Config system for GoPro 360 Gaussian Splatting.

Copied from Gaussians/lib/config/config.py and adapted:
  - Default --cfg_file points to configs/gopro360.yaml
  - workspace = os.getcwd() (run from gopro360/)
"""

from .yacs import CfgNode as CN
import argparse
import os
import numpy as np

from lib.utils.cfg_utils import make_cfg

cfg = CN()

cfg.workspace = os.getcwd()               # ← Windows-friendly
cfg.loaded_iter = -1
cfg.ip = '127.0.0.1'
cfg.port = 6009
cfg.data_device = 'cuda'
cfg.mode = 'train'
cfg.task = 'hello'
cfg.exp_name = 'test'
cfg.gpus = [0]
cfg.debug = False
cfg.resume = True
cfg.to_cuda = False

cfg.source_path = ''
cfg.model_path = ''
cfg.record_dir = None
cfg.resolution = -1
cfg.resolution_scales = [1]

# ── Eval ──────────────────────────────────────────────────────────────────
cfg.eval = CN()
cfg.eval.skip_train = False
cfg.eval.skip_test = False
cfg.eval.eval_train = False
cfg.eval.eval_test = True
cfg.eval.quiet = False

# ── Train ─────────────────────────────────────────────────────────────────
cfg.train = CN()
cfg.train.debug_from = -1
cfg.train.detect_anomaly = False
cfg.train.test_iterations = [7000, 30000]
cfg.train.save_iterations = [7000, 30000]
cfg.train.iterations = 30000
cfg.train.epochs = 0
cfg.train.quiet = False
cfg.train.checkpoint_iterations = [30000]
cfg.train.start_checkpoint = None
cfg.train.importance_sampling = False

# ── Optim ─────────────────────────────────────────────────────────────────
cfg.optim = CN()
cfg.optim.position_lr_init = 0.00016
cfg.optim.position_lr_final = 0.0000016
cfg.optim.position_lr_delay_mult = 0.01
cfg.optim.position_lr_max_steps = 30000
cfg.optim.feature_lr = 0.0025
cfg.optim.opacity_lr = 0.05
cfg.optim.scaling_lr = 0.005
cfg.optim.rotation_lr = 0.001
cfg.optim.semantic_lr = 0.001           # ← added default

cfg.optim.percent_dense = 0.01
cfg.optim.densification_interval = 100
cfg.optim.opacity_reset_interval = 3000
cfg.optim.densify_from_iter = 500
cfg.optim.densify_until_iter = 15000
cfg.optim.densify_grad_threshold = 0.0002
cfg.optim.densify_grad_threshold_bkgd = 0.0006
cfg.optim.densify_grad_abs_bkgd = False
cfg.optim.densify_grad_abs_obj = False
cfg.optim.max_screen_size = 20
cfg.optim.min_opacity = 0.005
cfg.optim.percent_big_ws = 0.1

cfg.optim.lambda_l1 = 1.0
cfg.optim.lambda_dssim = 0.2
cfg.optim.lambda_sky = 0.0
cfg.optim.lambda_sky_scale = []
cfg.optim.lambda_semantic = 0.0
cfg.optim.lambda_reg = 0.0
cfg.optim.lambda_depth_lidar = 0.0
cfg.optim.lambda_depth_mono = 0.0
cfg.optim.lambda_normal_mono = 0.0
cfg.optim.lambda_color_correction = 0.0
cfg.optim.lambda_pose_correction = 0.0
cfg.optim.lambda_scale_flatten = 0.0
cfg.optim.lambda_opacity_sparse = 0.0

# ── Model ─────────────────────────────────────────────────────────────────
cfg.model = CN()
cfg.model.gaussian = CN()
cfg.model.gaussian.sh_degree = 3
cfg.model.gaussian.fourier_dim = 1
cfg.model.gaussian.fourier_scale = 1.0
cfg.model.gaussian.flip_prob = 0.0
cfg.model.gaussian.semantic_mode = 'logits'

cfg.model.nsg = CN()
cfg.model.nsg.include_bkgd = True
cfg.model.nsg.include_obj = False          # ← default off for KITTI-360
cfg.model.nsg.include_sky = False
cfg.model.nsg.opt_track = False

cfg.model.sky = CN()
cfg.model.sky.resolution = 1024
cfg.model.sky.white_background = True

cfg.model.use_color_correction = False
cfg.model.color_correction = CN()
cfg.model.color_correction.mode = 'image'
cfg.model.color_correction.use_mlp = False
cfg.model.color_correction.use_sky = False

cfg.model.use_pose_correction = False
cfg.model.pose_correction = CN()
cfg.model.pose_correction.mode = 'image'

# ── Data ──────────────────────────────────────────────────────────────────
cfg.data = CN()
cfg.data.white_background = False
cfg.data.use_colmap_pose = False
cfg.data.filter_colmap = False
cfg.data.box_scale = 1.0
cfg.data.split_test = -1
cfg.data.split_train = 1
cfg.data.shuffle = True
cfg.data.eval = True
cfg.data.type = 'KITTI360'
cfg.data.images = 'images'
cfg.data.selected_frames = [250, 490]
cfg.data.cameras = [0, 1]
cfg.data.extent = 15
cfg.data.drive = '2013_05_28_drive_0000_sync'
cfg.data.point_cloud_path = ''
cfg.data.use_semantic = False
cfg.data.use_mono_depth = False
cfg.data.use_mono_normal = False
cfg.data.use_colmap = False

# ── Render ────────────────────────────────────────────────────────────────
cfg.render = CN()
cfg.render.convert_SHs_python = False
cfg.render.compute_cov3D_python = False
cfg.render.debug = False
cfg.render.scaling_modifier = 1.0
cfg.render.fps = 24
cfg.render.render_normal = False
cfg.render.save_video = True
cfg.render.save_image = True
cfg.render.coord = 'world'
cfg.render.concat_cameras = []

cfg.viewer = CN()
cfg.viewer.frame_id = 0

# ── CLI ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", "--config", default="configs/gopro360.yaml", type=str)
parser.add_argument("--mode", type=str, default="")
parser.add_argument('--det', type=str, default='')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

args = parser.parse_args()
# Remap: make_cfg expects args.config
args.config = args.cfg_file
cfg = make_cfg(cfg, args)
