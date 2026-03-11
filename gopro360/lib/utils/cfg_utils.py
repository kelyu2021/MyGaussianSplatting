"""
Config parsing & saving utilities.

Adapted from street_gaussians:
  • ``os.environ['PWD']``  →  ``os.getcwd()``  (Windows compatible)
  • ``os.system('mkdir -p …')``  →  ``os.makedirs(…, exist_ok=True)``
  • Removed ``ipdb`` breakpoint
"""

import os
import numpy as np
from lib.config import yacs


def parse_cfg(cfg, args):
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')

    # assign the GPUs
    if -1 not in cfg.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(str(g) for g in cfg.gpus)

    if cfg.debug:
        os.environ["PYTHONBREAKPOINT"] = "pdbr.set_trace"

    cur_workspace = os.getcwd()   # ← Windows-friendly

    # ── model directory ──────────────────────────────────────────────────
    if cfg.model_path == '':
        cfg.model_path = os.path.join('output', cfg.task, cfg.exp_name)

    if not os.path.isabs(cfg.model_path):
        cfg.model_path = os.path.join(cfg.workspace, cfg.model_path)
        cfg.model_path = os.path.normpath(cfg.model_path)

    if not os.path.exists(cfg.model_path):
        relative_path = os.path.relpath(cfg.model_path, cfg.workspace)
        cfg.model_path = os.path.join(cur_workspace, relative_path)

    if os.path.exists(cfg.model_path) and cfg.mode == 'train':
        print('Model path already exists, this would override original model')
        print(f"model_path: {cfg.model_path}")

    cfg.trained_model_dir = os.path.join(cfg.model_path, 'trained_model')
    cfg.point_cloud_dir = os.path.join(cfg.model_path, 'point_cloud')

    # ── data directory ───────────────────────────────────────────────────
    if not os.path.isabs(cfg.source_path):
        cfg.source_path = os.path.join(cfg.workspace, cfg.source_path)
        cfg.source_path = os.path.normpath(cfg.source_path)

    if not os.path.exists(cfg.source_path):
        relative_path = os.path.relpath(cfg.source_path, cfg.workspace)
        cfg.source_path = os.path.join(cur_workspace, relative_path)
        if not os.path.exists(cfg.source_path):
            raise FileNotFoundError(
                f"source_path not found: {cfg.source_path}\n"
                f"  workspace={cfg.workspace}  cwd={cur_workspace}"
            )

    # ── log directory ────────────────────────────────────────────────────
    if cfg.record_dir is None:
        cfg.record_dir = os.path.join('output', 'record', cfg.task, cfg.exp_name)

    if not os.path.isabs(cfg.record_dir):
        cfg.record_dir = os.path.join(cfg.workspace, cfg.record_dir)
        cfg.record_dir = os.path.normpath(cfg.record_dir)

    if not os.path.exists(cfg.record_dir):
        relative_path = os.path.relpath(cfg.record_dir, cfg.workspace)
        cfg.record_dir = os.path.join(cur_workspace, relative_path)


def make_cfg(cfg, args):
    def merge_cfg(cfg_file, cfg):
        with open(cfg_file, 'r') as f:
            current_cfg = yacs.load_cfg(f)
        if 'parent_cfg' in current_cfg.keys():
            cfg = merge_cfg(current_cfg.parent_cfg, cfg)
            cfg.merge_from_other_cfg(current_cfg)
        else:
            cfg.merge_from_other_cfg(current_cfg)
        print(cfg_file)
        return cfg

    cfg_ = merge_cfg(args.config, cfg)
    try:
        index = args.opts.index('other_opts')
        cfg_.merge_from_list(args.opts[:index])
    except Exception:
        cfg_.merge_from_list(args.opts)

    # Apply --mode from CLI if provided
    if getattr(args, 'mode', '') != '':
        cfg_.mode = args.mode

    parse_cfg(cfg_, args)
    return cfg_


def save_cfg(cfg, model_dir, epoch=0):
    from contextlib import redirect_stdout
    os.makedirs(model_dir, exist_ok=True)            # ← Windows-friendly
    cfg_dir = os.path.join(model_dir, 'configs')
    os.makedirs(cfg_dir, exist_ok=True)               # ← Windows-friendly

    cfg_path = os.path.join(cfg_dir, f'config_{epoch:06d}.yaml')
    with open(cfg_path, 'w') as f:
        with redirect_stdout(f):
            print(cfg.dump())

    print(f'Save input config to {cfg_path}')
