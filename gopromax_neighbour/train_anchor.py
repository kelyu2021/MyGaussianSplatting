"""
GoPro Max Neighbour – Gaussian Splatting Training with Sky Mask
===============================================================

Self-contained training script: reads COLMAP data, initialises Gaussians
from a dense point cloud, and trains with sky-mask supervision.

Data layout (relative to gopromax_neighbour/)::

    data/colmap_pointcloud_dense/
        sparse/          <- cameras.bin, images.bin, points3D.bin
        images/          <- undistorted cubemap faces
        fused.ply        <- dense point cloud
    data/cubemap_faces/          <- original cubemap images
    data/cubemap_faces_mass13k/  <- sky masks (0=sky/masked, 255=valid)
    data/cubemap_faces_sam_moving/ <- moving object masks (255=moving, 0=static)

Usage::

    cd MyGaussianSplatting/gopromax_neighbour
    python train.py --config configs/gopromax_neighbour.yaml
"""

from __future__ import annotations

import os
import sys
import math
import copy
import json
import struct
import argparse
import csv
from pathlib import Path
from random import shuffle, seed as set_seed
from collections import namedtuple, OrderedDict

import yaml
import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# ── CUDA extensions (installed from submodules) ──────────────────────────
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

try:
    import lpips as _lpips_module
    LPIPS_FOUND = True
except ImportError:
    LPIPS_FOUND = False

try:
    from simple_knn._C import distCUDA2
except ImportError:
    def distCUDA2(points):
        """Fallback: squared distance to nearest neighbour via scipy."""
        from scipy.spatial import KDTree
        pts_np = points.detach().cpu().float().numpy()
        tree = KDTree(pts_np)
        dists, _ = tree.query(pts_np, k=2)
        return torch.tensor(
            dists[:, 1] ** 2, dtype=torch.float32, device=points.device)

try:
    from plyfile import PlyData, PlyElement
except ImportError:
    PlyData = PlyElement = None

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ═══════════════════════════════════════════════════════════════════════════
#  §1  Math / SH Utilities
# ═══════════════════════════════════════════════════════════════════════════

C0 = 0.28209479177387814


def inverse_sigmoid(x):
    return torch.log(x / (1.0 - x))


def RGB2SH(rgb):
    """Convert RGB [0,1] → 0-th order SH coefficient."""
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5


def build_rotation(q):
    """Quaternion (w,x,y,z) → 3×3 rotation matrix.  Batched: (N,4)→(N,3,3)."""
    norm = torch.sqrt((q * q).sum(dim=-1, keepdim=True)).clamp(min=1e-8)
    q = q / norm
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.zeros((q.shape[0], 3, 3), device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    """Build 4×4 world-to-view matrix.

    R: (3,3) camera-to-world rotation (R_w2c.T)
    t: (3,)  world-to-camera translation
    """
    Rt = np.zeros((4, 4), dtype=np.float32)
    Rt[:3, :3] = R.T
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrixK(znear, zfar, K, H, W):
    """Build 4×4 projection matrix from intrinsic K (torch tensor)."""
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    P = torch.zeros(4, 4, dtype=torch.float32)
    z_sign = 1.0
    P[0, 0] = 2.0 * fx / W
    P[1, 1] = 2.0 * fy / H
    P[0, 2] = -(2.0 * cx / W - 1.0)
    P[1, 2] = -(2.0 * cy / H - 1.0)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=0.01, max_steps=1_000_000,
):
    """Exponential learning-rate schedule (returns a callable)."""
    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * math.sin(
                0.5 * math.pi * min(step / lr_delay_steps, 1.0))
        else:
            delay_rate = 1.0
        t = min(step / max(max_steps, 1), 1.0)
        log_lerp = math.exp(
            math.log(max(lr_init, 1e-10)) * (1 - t) +
            math.log(max(lr_final, 1e-10)) * t)
        return delay_rate * log_lerp
    return helper


# ═══════════════════════════════════════════════════════════════════════════
#  §2  COLMAP Binary Readers
# ═══════════════════════════════════════════════════════════════════════════

ColmapCamera = namedtuple(
    "ColmapCamera", ["id", "model", "width", "height", "params"])
ColmapImage = namedtuple(
    "ColmapImage", ["id", "qvec", "tvec", "camera_id", "name",
                     "xys", "point3D_ids"])

CAMERA_MODEL_NUM_PARAMS = {
    0: 3,   # SIMPLE_PINHOLE: f, cx, cy
    1: 4,   # PINHOLE: fx, fy, cx, cy
    2: 4,   # SIMPLE_RADIAL: f, cx, cy, k
    3: 5,   # RADIAL: f, cx, cy, k1, k2
    4: 8,   # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
}


def read_cameras_binary(path: str) -> dict:
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            cam_id, model_id = struct.unpack("<ii", f.read(8))
            width, height = struct.unpack("<QQ", f.read(16))
            num_params = CAMERA_MODEL_NUM_PARAMS.get(model_id, 0)
            params = np.array(struct.unpack(
                f"<{num_params}d", f.read(8 * num_params)))
            cameras[cam_id] = ColmapCamera(
                id=cam_id, model=model_id,
                width=int(width), height=int(height), params=params)
    return cameras


def read_images_binary(path: str) -> dict:
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            data = struct.unpack("<idddddddi", f.read(64))
            image_id = data[0]
            qvec = np.array(data[1:5])
            tvec = np.array(data[5:8])
            camera_id = data[8]
            # Read null-terminated name
            name_chars = []
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name_chars.append(ch.decode("utf-8"))
            name = "".join(name_chars)
            num_pts = struct.unpack("<Q", f.read(8))[0]
            xys = np.zeros((num_pts, 2))
            point3D_ids = np.zeros(num_pts, dtype=np.int64)
            for j in range(num_pts):
                xy_id = struct.unpack("<ddq", f.read(24))
                xys[j] = xy_id[:2]
                point3D_ids[j] = xy_id[2]
            images[image_id] = ColmapImage(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def qvec2rotmat(qvec):
    """Convert COLMAP quaternion (w,x,y,z) → 3×3 rotation matrix."""
    w, x, y, z = qvec
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y    ],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x    ],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y],
    ])
    return R


def get_intrinsics(cam: ColmapCamera) -> dict:
    """Extract fx, fy, cx, cy from a COLMAP camera."""
    p = cam.params
    model = cam.model
    if model == 0:      # SIMPLE_PINHOLE
        return {"fx": p[0], "fy": p[0], "cx": p[1], "cy": p[2]}
    elif model == 1:    # PINHOLE
        return {"fx": p[0], "fy": p[1], "cx": p[2], "cy": p[3]}
    elif model == 2:    # SIMPLE_RADIAL
        return {"fx": p[0], "fy": p[0], "cx": p[1], "cy": p[2]}
    elif model == 3:    # RADIAL
        return {"fx": p[0], "fy": p[0], "cx": p[1], "cy": p[2]}
    elif model == 4:    # OPENCV
        return {"fx": p[0], "fy": p[1], "cx": p[2], "cy": p[3]}
    else:
        raise ValueError(f"Unknown COLMAP camera model: {model}")


# ═══════════════════════════════════════════════════════════════════════════
#  §3  PLY Point Cloud Reader
# ═══════════════════════════════════════════════════════════════════════════

BasicPointCloud = namedtuple(
    "BasicPointCloud", ["points", "colors", "normals"])


def read_ply(path: str) -> BasicPointCloud:
    """Read a PLY (binary little-endian) with xyz + rgb [+ normals]."""
    plydata = PlyData.read(path)
    vertex = plydata["vertex"]
    x = np.asarray(vertex["x"], dtype=np.float32)
    y = np.asarray(vertex["y"], dtype=np.float32)
    z = np.asarray(vertex["z"], dtype=np.float32)
    points = np.stack([x, y, z], axis=1)

    try:
        r = np.asarray(vertex["red"], dtype=np.float32)
        g = np.asarray(vertex["green"], dtype=np.float32)
        b = np.asarray(vertex["blue"], dtype=np.float32)
        colors = np.stack([r, g, b], axis=1) / 255.0
    except ValueError:
        colors = np.zeros_like(points)

    try:
        nx = np.asarray(vertex["nx"], dtype=np.float32)
        ny = np.asarray(vertex["ny"], dtype=np.float32)
        nz = np.asarray(vertex["nz"], dtype=np.float32)
        normals = np.stack([nx, ny, nz], axis=1)
    except ValueError:
        normals = np.zeros_like(points)

    return BasicPointCloud(points=points, colors=colors, normals=normals)


def store_ply(path: str, xyz: np.ndarray, rgb: np.ndarray):
    """Write a minimal PLY with xyz + rgb."""
    if PlyElement is None:
        print(f"WARNING: plyfile not installed, skipping PLY save: {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nrm = np.zeros_like(xyz)
    rgb_u8 = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements[:] = list(map(
        tuple, np.concatenate([xyz, nrm, rgb_u8], axis=1)))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(path)


# ═══════════════════════════════════════════════════════════════════════════
#  §4  Camera
# ═══════════════════════════════════════════════════════════════════════════

CameraInfo = namedtuple(
    "CameraInfo",
    ["uid", "R", "T", "FovY", "FovX", "K",
     "image", "image_path", "image_name",
     "width", "height", "metadata", "guidance"])


class Camera(nn.Module):
    """Torch-compatible camera with intrinsics, extrinsics, and guidance."""

    def __init__(
        self, uid, R, T, FoVx, FoVy, K,
        image, image_name,
        metadata=None, guidance=None,
    ):
        super().__init__()
        self.id = uid
        self.R = R                              # (3,3) c2w rotation = R_w2c.T
        self.T = T                              # (3,)  w2c translation
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.meta = metadata or {}
        self.guidance = guidance or {}

        self.original_image = image.clamp(0.0, 1.0)
        self.image_height = self.original_image.shape[1]
        self.image_width = self.original_image.shape[2]

        self.zfar = 1000.0
        self.znear = 0.001

        # Intrinsic
        if isinstance(K, np.ndarray):
            self.K = torch.from_numpy(K).float().cuda()
        else:
            self.K = K.float().cuda()

        # World-to-view
        self.world_view_transform = torch.tensor(
            getWorld2View2(R, T)).transpose(0, 1).cuda().float()

        # Projection from K
        self.projection_matrix = getProjectionMatrixK(
            self.znear, self.zfar, self.K,
            self.image_height, self.image_width,
        ).transpose(0, 1).cuda().float()

        # Combined
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            ).squeeze(0))

        self.camera_center = self.world_view_transform.inverse()[3, :3]


def _pil_to_torch(pil_img, resolution=None, mode=Image.BILINEAR):
    """PIL Image → (C, H, W) torch float in [0, 1]."""
    if resolution is not None:
        pil_img = pil_img.resize(resolution, mode)
    arr = torch.from_numpy(np.array(pil_img)).float() / 255.0
    if arr.ndim == 3:
        return arr.permute(2, 0, 1)
    return arr.unsqueeze(0)          # grayscale → (1, H, W)


def load_camera(cam_info: CameraInfo, resolution_scale: float = 1.0):
    """Build a Camera object from CameraInfo."""
    orig_w, orig_h = cam_info.width, cam_info.height
    scale = min(1.0, 1600 / orig_w)
    scale = scale / resolution_scale
    resolution = (int(orig_w * scale), int(orig_h * scale))

    K = copy.deepcopy(cam_info.K)
    K[:2] *= scale

    image = _pil_to_torch(cam_info.image, resolution, Image.BILINEAR)[:3]

    guidance = {}
    for k, v in cam_info.guidance.items():
        if k in ("mask", "sky_mask", "acc_mask", "moving_mask"):
            guidance[k] = _pil_to_torch(v, resolution, Image.NEAREST).bool()
        elif k in ("lidar_depth", "mono_depth"):
            t = torch.from_numpy(v).float()
            if resolution is not None:
                t = F.interpolate(
                    t.unsqueeze(0).unsqueeze(0),
                    size=(resolution[1], resolution[0]),
                    mode="nearest")[0, 0]
            guidance[k] = t
        else:
            guidance[k] = v

    return Camera(
        uid=cam_info.uid, R=cam_info.R, T=cam_info.T,
        FoVx=cam_info.FovX, FoVy=cam_info.FovY, K=K,
        image=image, image_name=cam_info.image_name,
        metadata=cam_info.metadata, guidance=guidance)


# ═══════════════════════════════════════════════════════════════════════════
#  §5  Gaussian Model
# ═══════════════════════════════════════════════════════════════════════════

class GaussianModel:
    """3-D Gaussian Splatting model with adaptive density control."""

    def __init__(self, sh_degree: int = 3):
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0

        self._xyz: nn.Parameter = None
        self._features_dc: nn.Parameter = None
        self._features_rest: nn.Parameter = None
        self._scaling: nn.Parameter = None
        self._rotation: nn.Parameter = None
        self._opacity: nn.Parameter = None

        self.optimizer: torch.optim.Adam = None
        self.spatial_lr_scale: float = 1.0
        self.percent_dense: float = 0.01
        self.xyz_gradient_accum = None
        self.denom = None
        self.max_radii2D = None

        self._xyz_scheduler_func = None

    # ── Properties ────────────────────────────────────────────────────

    @property
    def num_points(self) -> int:
        return 0 if self._xyz is None else self._xyz.shape[0]

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_rotation(self):
        return F.normalize(self._rotation, dim=-1)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    @property
    def get_features(self):
        return torch.cat([self._features_dc, self._features_rest], dim=1)

    # ── Initialisation ────────────────────────────────────────────────

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        """Initialise Gaussians from a point cloud."""
        self.spatial_lr_scale = spatial_lr_scale

        pts = torch.tensor(pcd.points, dtype=torch.float32, device="cuda")
        rgb = torch.tensor(pcd.colors, dtype=torch.float32, device="cuda")

        print(f"[GaussianModel] Initialising {pts.shape[0]:,} Gaussians "
              f"(spatial_lr_scale={spatial_lr_scale:.2f})")

        # SH features from RGB
        fused_color = RGB2SH(rgb)               # (N, 3)
        features_dc = fused_color.unsqueeze(1)   # (N, 1, 3)
        num_sh_rest = (self.max_sh_degree + 1) ** 2 - 1
        features_rest = torch.zeros(
            (pts.shape[0], num_sh_rest, 3), device="cuda")

        # Scales from nearest-neighbour distances
        dist2 = distCUDA2(pts)
        scales = torch.log(
            torch.sqrt(dist2).clamp(min=1e-7)
        ).unsqueeze(-1).repeat(1, 3)

        # Identity quaternion rotations
        rots = torch.zeros((pts.shape[0], 4), device="cuda")
        rots[:, 0] = 1.0

        # Opacity initialised to 0.1 (pre-sigmoid)
        opacities = inverse_sigmoid(
            0.1 * torch.ones((pts.shape[0], 1), device="cuda"))

        self._xyz = nn.Parameter(pts)
        self._features_dc = nn.Parameter(features_dc)
        self._features_rest = nn.Parameter(features_rest)
        self._scaling = nn.Parameter(scales)
        self._rotation = nn.Parameter(rots)
        self._opacity = nn.Parameter(opacities)

    # ── Optimiser Setup ───────────────────────────────────────────────

    def training_setup(self, optim_cfg: dict):
        """Create Adam optimiser with per-parameter learning rates."""
        self.active_sh_degree = 0
        self.percent_dense = optim_cfg.get("percent_dense", 0.01)

        lr_pos = optim_cfg["position_lr_init"] * self.spatial_lr_scale

        param_groups = [
            {"params": [self._xyz],           "lr": lr_pos,                          "name": "xyz"},
            {"params": [self._features_dc],   "lr": optim_cfg["feature_lr"],         "name": "f_dc"},
            {"params": [self._features_rest], "lr": optim_cfg["feature_lr"] / 20.0,  "name": "f_rest"},
            {"params": [self._opacity],       "lr": optim_cfg["opacity_lr"],         "name": "opacity"},
            {"params": [self._scaling],       "lr": optim_cfg["scaling_lr"],         "name": "scaling"},
            {"params": [self._rotation],      "lr": optim_cfg["rotation_lr"],        "name": "rotation"},
        ]
        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

        self._xyz_scheduler_func = get_expon_lr_func(
            lr_init=optim_cfg["position_lr_init"] * self.spatial_lr_scale,
            lr_final=optim_cfg["position_lr_final"] * self.spatial_lr_scale,
            max_steps=optim_cfg["position_lr_max_steps"],
        )

        # Densification bookkeeping
        N = self._xyz.shape[0]
        self.xyz_gradient_accum = torch.zeros((N, 1), device="cuda")
        self.denom = torch.zeros((N, 1), device="cuda")
        self.max_radii2D = torch.zeros(N, device="cuda")

    def update_learning_rate(self, step: int):
        for pg in self.optimizer.param_groups:
            if pg["name"] == "xyz":
                pg["lr"] = self._xyz_scheduler_func(step)
                return pg["lr"]

    def update_optimizer(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def oneup_sh_degree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # ── Adaptive Density Control ──────────────────────────────────────

    def set_max_radii2D(self, radii, visibility_filter):
        self.max_radii2D[visibility_filter] = torch.max(
            self.max_radii2D[visibility_filter],
            radii[visibility_filter].float())

    def add_densification_stats(self, viewspace_points, visibility_filter):
        self.xyz_gradient_accum[visibility_filter] += torch.norm(
            viewspace_points.grad[visibility_filter, :2],
            dim=-1, keepdim=True)
        self.denom[visibility_filter] += 1

    def densify_and_prune(
        self, max_grad, min_opacity, max_screen_size, scene_extent,
        percent_big_ws=0.1,
    ):
        grads = self.xyz_gradient_accum / self.denom.clamp(min=1)
        grads[grads.isnan()] = 0.0

        n_before = self.num_points
        self._densify_and_clone(grads, max_grad, scene_extent)
        self._densify_and_split(grads, max_grad, scene_extent)

        # Prune
        prune_mask = (self.get_opacity < min_opacity).squeeze(-1)
        if max_screen_size > 0:
            big_screen = self.max_radii2D > max_screen_size
            prune_mask = prune_mask | big_screen
        if percent_big_ws > 0:
            big_world = (
                self.get_scaling.max(dim=1).values >
                percent_big_ws * scene_extent)
            prune_mask = prune_mask | big_world

        self._prune_points(prune_mask)

        # Reset densification stats
        self.xyz_gradient_accum = torch.zeros(
            (self.num_points, 1), device="cuda")
        self.denom = torch.zeros(
            (self.num_points, 1), device="cuda")
        self.max_radii2D = torch.zeros(self.num_points, device="cuda")

        n_after = self.num_points
        torch.cuda.empty_cache()
        return n_before, n_after

    def _densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected = (
            (grads.squeeze(-1) >= grad_threshold) &
            (self.get_scaling.max(dim=1).values <=
             self.percent_dense * scene_extent))
        if not selected.any():
            return
        new_tensors = {
            "xyz":      self._xyz[selected],
            "f_dc":     self._features_dc[selected],
            "f_rest":   self._features_rest[selected],
            "opacity":  self._opacity[selected],
            "scaling":  self._scaling[selected],
            "rotation": self._rotation[selected],
        }
        self._cat_tensors_to_optimizer(new_tensors)

    def _densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init = self.num_points

        padded_grad = torch.zeros(n_init, device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze(-1)

        selected = (
            (padded_grad >= grad_threshold) &
            (self.get_scaling.max(dim=1).values >
             self.percent_dense * scene_extent))
        if not selected.any():
            return

        stds = self.get_scaling[selected].repeat(N, 1)
        means = torch.zeros((stds.shape[0], 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(
            self._rotation[selected]).repeat(N, 1, 1)
        new_xyz = (
            torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) +
            self.get_xyz[selected].repeat(N, 1))
        new_scaling = torch.log(
            self.get_scaling[selected].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected].repeat(N, 1)
        new_f_dc = self._features_dc[selected].repeat(N, 1, 1)
        new_f_rest = self._features_rest[selected].repeat(N, 1, 1)
        new_opacity = self._opacity[selected].repeat(N, 1)

        new_tensors = {
            "xyz": new_xyz, "f_dc": new_f_dc, "f_rest": new_f_rest,
            "opacity": new_opacity, "scaling": new_scaling,
            "rotation": new_rotation,
        }
        self._cat_tensors_to_optimizer(new_tensors)

        # Remove the originals that were split
        prune = torch.cat([
            selected,
            torch.zeros(N * selected.sum(), device="cuda", dtype=torch.bool),
        ])
        self._prune_points(prune)

    def _cat_tensors_to_optimizer(self, tensors_dict):
        for group in self.optimizer.param_groups:
            name = group["name"]
            if name not in tensors_dict:
                continue
            ext = tensors_dict[name]
            stored = self.optimizer.state.get(group["params"][0], None)
            if stored is not None:
                stored["exp_avg"] = torch.cat(
                    [stored["exp_avg"], torch.zeros_like(ext)], dim=0)
                stored["exp_avg_sq"] = torch.cat(
                    [stored["exp_avg_sq"], torch.zeros_like(ext)], dim=0)
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat([group["params"][0], ext], dim=0)
                    .requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat([group["params"][0], ext], dim=0)
                    .requires_grad_(True))

        # Re-bind references
        self._rebind_params()
        self._pad_aux_tensors()

    def _pad_aux_tensors(self):
        """Extend auxiliary tensors to match current num_points."""
        N = self.num_points
        for attr, extra_dim in [("xyz_gradient_accum", 1), ("denom", 1), ("max_radii2D", None)]:
            old = getattr(self, attr, None)
            if old is None:
                continue
            if old.shape[0] < N:
                pad_n = N - old.shape[0]
                if extra_dim is not None:
                    pad = torch.zeros((pad_n, old.shape[1]), device=old.device)
                else:
                    pad = torch.zeros(pad_n, device=old.device)
                setattr(self, attr, torch.cat([old, pad], dim=0))
            elif old.shape[0] > N:
                setattr(self, attr, old[:N])

    def _prune_points(self, mask):
        valid = ~mask
        for group in self.optimizer.param_groups:
            stored = self.optimizer.state.get(group["params"][0], None)
            if stored is not None:
                stored["exp_avg"] = stored["exp_avg"][valid]
                stored["exp_avg_sq"] = stored["exp_avg_sq"][valid]
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    group["params"][0][valid].requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][valid].requires_grad_(True))

        self._rebind_params()

        # Trim auxiliary tensors
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid]
        self.denom = self.denom[valid]
        self.max_radii2D = self.max_radii2D[valid]

    def _rebind_params(self):
        for group in self.optimizer.param_groups:
            n = group["name"]
            p = group["params"][0]
            if n == "xyz":      self._xyz = p
            elif n == "f_dc":   self._features_dc = p
            elif n == "f_rest": self._features_rest = p
            elif n == "opacity": self._opacity = p
            elif n == "scaling": self._scaling = p
            elif n == "rotation": self._rotation = p

    # ── Opacity Reset ─────────────────────────────────────────────────

    def reset_opacity(self):
        new_opacity = inverse_sigmoid(
            torch.min(self.get_opacity,
                      torch.ones_like(self.get_opacity) * 0.01))
        # Replace in-place while updating optimizer state
        for group in self.optimizer.param_groups:
            if group["name"] == "opacity":
                stored = self.optimizer.state.get(group["params"][0], None)
                if stored is not None:
                    stored["exp_avg"].zero_()
                    stored["exp_avg_sq"].zero_()
                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(
                        new_opacity.requires_grad_(True))
                    self.optimizer.state[group["params"][0]] = stored
                else:
                    group["params"][0] = nn.Parameter(
                        new_opacity.requires_grad_(True))
                self._opacity = group["params"][0]
                break

    # ── Save / Load ───────────────────────────────────────────────────

    def save_ply(self, path: str):
        if PlyElement is None:
            print(f"WARNING: plyfile not installed, skipping: {path}")
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (self._features_dc.detach()
                .transpose(1, 2).flatten(start_dim=1)
                .contiguous().cpu().numpy())
        f_rest = (self._features_rest.detach()
                  .transpose(1, 2).flatten(start_dim=1)
                  .contiguous().cpu().numpy())
        opacities = self._opacity.detach().cpu().numpy()
        scales = self._scaling.detach().cpu().numpy()
        rotations = self._rotation.detach().cpu().numpy()

        dtype_list = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
        for i in range(f_dc.shape[1]):
            dtype_list.append((f"f_dc_{i}", "f4"))
        for i in range(f_rest.shape[1]):
            dtype_list.append((f"f_rest_{i}", "f4"))
        dtype_list.append(("opacity", "f4"))
        for i in range(scales.shape[1]):
            dtype_list.append((f"scale_{i}", "f4"))
        for i in range(rotations.shape[1]):
            dtype_list.append((f"rot_{i}", "f4"))

        all_attrs = np.concatenate(
            [xyz, normals, f_dc, f_rest, opacities, scales, rotations],
            axis=1)
        elements = np.empty(xyz.shape[0], dtype=dtype_list)
        elements[:] = list(map(tuple, all_attrs))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)
        print(f"  Saved PLY: {path}  ({xyz.shape[0]:,} Gaussians)")

    def save_state_dict(self, is_final=False):
        sd = {
            "active_sh_degree": self.active_sh_degree,
            "_xyz": self._xyz.detach(),
            "_features_dc": self._features_dc.detach(),
            "_features_rest": self._features_rest.detach(),
            "_opacity": self._opacity.detach(),
            "_scaling": self._scaling.detach(),
            "_rotation": self._rotation.detach(),
        }
        if not is_final and self.optimizer is not None:
            sd["optimizer_state"] = self.optimizer.state_dict()
            sd["xyz_gradient_accum"] = self.xyz_gradient_accum
            sd["denom"] = self.denom
            sd["max_radii2D"] = self.max_radii2D
        return sd

    def load_state_dict(self, sd):
        self.active_sh_degree = sd.get(
            "active_sh_degree", self.max_sh_degree)
        self._xyz = nn.Parameter(sd["_xyz"].cuda())
        self._features_dc = nn.Parameter(sd["_features_dc"].cuda())
        self._features_rest = nn.Parameter(sd["_features_rest"].cuda())
        self._opacity = nn.Parameter(sd["_opacity"].cuda())
        self._scaling = nn.Parameter(sd["_scaling"].cuda())
        self._rotation = nn.Parameter(sd["_rotation"].cuda())

    def _sync_auxiliary_tensors(self):
        """Resize aux tensors to current point count after checkpoint load."""
        N = self.num_points
        device = self._xyz.device
        for attr, shape in [
            ("xyz_gradient_accum", (N, 1)),
            ("denom",              (N, 1)),
            ("max_radii2D",        (N,)),
        ]:
            old = getattr(self, attr, None)
            if old is None or old.shape[0] != N:
                setattr(self, attr, torch.zeros(shape, device=device))


# ═══════════════════════════════════════════════════════════════════════════
#  §6  Rendering
# ═══════════════════════════════════════════════════════════════════════════

def render(camera: Camera, gaussians: GaussianModel, bg_color: torch.Tensor,
           scaling_modifier: float = 1.0):
    """Render Gaussians into a camera view.

    Returns dict with keys: rgb, depth, acc, viewspace_points,
    visibility_filter, radii.
    """
    N = gaussians.num_points

    screenspace_points = torch.zeros(
        (N, 3), dtype=torch.float32, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    raster_settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
        tanfovx=math.tan(camera.FoVx * 0.5),
        tanfovy=math.tan(camera.FoVy * 0.5),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=gaussians.active_sh_degree,
        campos=camera.camera_center,
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = gaussians.get_xyz
    opacity = gaussians.get_opacity
    scales = gaussians.get_scaling
    rotations = gaussians.get_rotation
    shs = gaussians.get_features

    rendered_color, radii, rendered_depth, rendered_acc, _ = rasterizer(
        means3D=means3D,
        means2D=screenspace_points,
        opacities=opacity,
        shs=shs,
        colors_precomp=None,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
        semantics=None,
    )

    visibility_filter = radii > 0

    return {
        "rgb": rendered_color,
        "depth": rendered_depth,
        "acc": rendered_acc,
        "viewspace_points": screenspace_points,
        "visibility_filter": visibility_filter,
        "radii": radii,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  §7  Loss Functions
# ═══════════════════════════════════════════════════════════════════════════

def l1_loss(rendered, gt, mask=None):
    loss = torch.abs(rendered - gt)
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum().clamp(min=1)
    return loss.mean()


def _gaussian_window(window_size: int, channel: int):
    """Create a 2-D Gaussian window for SSIM."""
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(1) @ g.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0)         # (1, 1, H, W)
    return window.expand(channel, 1, -1, -1).contiguous()


def ssim(img1, img2, window_size=11, mask=None):
    """Structural Similarity Index (differentiable).

    img1, img2: (C, H, W)   mask: (1, H, W) bool or None
    """
    channel = img1.shape[0]
    window = _gaussian_window(window_size, channel).to(img1.device)
    pad = window_size // 2

    mu1 = F.conv2d(
        img1.unsqueeze(0), window, padding=pad, groups=channel).squeeze(0)
    mu2 = F.conv2d(
        img2.unsqueeze(0), window, padding=pad, groups=channel).squeeze(0)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(
            (img1 * img1).unsqueeze(0), window, padding=pad, groups=channel
        ).squeeze(0) - mu1_sq)
    sigma2_sq = (
        F.conv2d(
            (img2 * img2).unsqueeze(0), window, padding=pad, groups=channel
        ).squeeze(0) - mu2_sq)
    sigma12 = (
        F.conv2d(
            (img1 * img2).unsqueeze(0), window, padding=pad, groups=channel
        ).squeeze(0) - mu1_mu2)

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = (
        (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    ) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if mask is not None:
        ssim_map = ssim_map * mask
        return ssim_map.sum() / (mask.sum() * channel).clamp(min=1)
    return ssim_map.mean()


def psnr(rendered, gt, mask=None):
    mse = ((rendered - gt) ** 2)
    if mask is not None:
        mse = (mse * mask).sum() / mask.sum().clamp(min=1)
    else:
        mse = mse.mean()
    return 10.0 * torch.log10(1.0 / mse.clamp(min=1e-10))


def depth_mono_loss(rendered_depth, mono_depth, acc, mask=None):
    """Robust scale-shift-invariant depth loss for monocular depth.

    Adapted from gopro360's LiDAR depth loss.  Since monocular depth
    has unknown absolute scale, we first align it to the rendered depth
    via least-squares scale+shift, then compute robust L1.

    Parameters
    ----------
    rendered_depth : (1, H, W)  rendered depth from the rasterizer
    mono_depth     : (H, W)     monocular relative depth (sky == 0)
    acc            : (1, H, W)  accumulated alpha from rasterizer
    mask           : (1, H, W)  optional; True = valid pixel
    """
    # Expected per-pixel depth (normalised by alpha), same as gopro360
    rd = (rendered_depth / (acc + 1e-10)).squeeze(0)   # (H, W)
    md = mono_depth                                     # (H, W)

    valid = md > 0
    if mask is not None:
        valid = valid & mask.squeeze(0)

    if valid.sum() < 10:
        return rendered_depth.new_tensor(0.0)

    r = rd[valid]
    m = md[valid]

    # Closed-form least-squares: align mono → rendered  (detached)
    with torch.no_grad():
        m_mean = m.mean()
        r_mean = r.mean()
        m_c = m - m_mean
        r_c = r - r_mean
        scale = (m_c * r_c).sum() / (m_c * m_c).sum().clamp(min=1e-8)
        shift = r_mean - scale * m_mean

    aligned_m = scale * m + shift          # fully detached target

    d_err = torch.abs(r - aligned_m)
    # Robust: keep bottom 95 % of errors (discard outliers like gopro360)
    d_err, _ = torch.topk(d_err, int(0.95 * d_err.numel()), largest=False)
    return d_err.mean()


# ═══════════════════════════════════════════════════════════════════════════
#  §8  Dataset Reader  (COLMAP + cubemap images + sky masks)
# ═══════════════════════════════════════════════════════════════════════════

FACE_TO_CAM_ID = {"front": 0, "right": 1, "back": 2, "left": 3}


def _parse_image_name(filename: str):
    """Parse '0001_front.jpg' → (frame_name='0001', face='front')."""
    stem = Path(filename).stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in FACE_TO_CAM_ID:
        return parts[0], parts[1]
    return stem, "unknown"


SceneInfo = namedtuple(
    "SceneInfo",
    ["point_cloud", "train_cameras", "test_cameras",
     "scene_center", "scene_radius", "ply_path"])


def read_scene(
    source_path: str,
    point_cloud_path: str,
    images_dir: str,
    mask_dir: str,
    moving_mask_dir: str = "",
    depth_dir: str = "",
    split_test: int = 8,
    workspace: str = "",
    max_frames: int = 0,
) -> SceneInfo:
    """Read COLMAP model, images, masks, and point cloud.

    Parameters
    ----------
    source_path   : path to COLMAP dense output (contains sparse/, images/)
    point_cloud_path : explicit path to fused.ply
    images_dir    : path to cubemap face images
    mask_dir      : path to sky mask images
    moving_mask_dir : path to moving object mask images (255=moving, 0=static)
    depth_dir     : path to monocular depth maps (*_depth_raw.npy, sky=0)
    split_test    : every Nth frame → test set
    workspace     : project root for resolving relative paths
    max_frames    : if >0, only load the first N frames (for quick testing)
    """
    src = Path(source_path)
    ws = Path(workspace) if workspace else Path.cwd()

    # ── Resolve paths ─────────────────────────────────────────────────
    def _resolve(p):
        p = Path(p)
        if p.is_absolute():
            return p
        return (ws / p).resolve()

    sparse_dir = _resolve(src / "sparse")
    images_path = _resolve(images_dir)
    mask_path = _resolve(mask_dir) if mask_dir else None
    moving_mask_path = _resolve(moving_mask_dir) if moving_mask_dir else None
    depth_path = _resolve(depth_dir) if depth_dir else None
    pcd_path = _resolve(point_cloud_path)

    assert (sparse_dir / "cameras.bin").exists(), \
        f"cameras.bin not found in {sparse_dir}"
    assert (sparse_dir / "images.bin").exists(), \
        f"images.bin not found in {sparse_dir}"

    # ── Read COLMAP model ─────────────────────────────────────────────
    cameras_bin = read_cameras_binary(str(sparse_dir / "cameras.bin"))
    images_bin = read_images_binary(str(sparse_dir / "images.bin"))
    print(f"[Scene] COLMAP cameras: {len(cameras_bin)}, "
          f"images: {len(images_bin)}")

    # ── Group images by frame ─────────────────────────────────────────
    frame_groups: OrderedDict[str, list] = OrderedDict()
    sorted_images = sorted(images_bin.values(), key=lambda x: x.name)

    for colmap_img in sorted_images:
        frame_name, face_name = _parse_image_name(colmap_img.name)
        frame_groups.setdefault(frame_name, []).append(
            (colmap_img, face_name))

    unique_frames = list(frame_groups.keys())
    if max_frames > 0:
        unique_frames = unique_frames[:max_frames]
        print(f"[Scene] Limiting to first {max_frames} frames for quick testing")
    print(f"[Scene] Unique frames: {len(unique_frames)}, "
          f"faces/frame: {[len(v) for v in list(frame_groups.values())[:3]]}...")

    # ── Build CameraInfo lists ────────────────────────────────────────
    train_cams: list[CameraInfo] = []
    test_cams: list[CameraInfo] = []
    uid = 0

    for frame_idx, frame_name in enumerate(unique_frames):
        is_test = (split_test > 0) and (frame_idx % split_test == 0)

        for colmap_img, face_name in frame_groups[frame_name]:
            # Find image file
            img_file = images_path / colmap_img.name
            if not img_file.exists():
                print(f"  WARNING: image not found: {img_file}, skipping")
                continue

            cam = cameras_bin[colmap_img.camera_id]
            intr = get_intrinsics(cam)
            width, height = cam.width, cam.height

            K = np.array([
                [intr["fx"], 0,          intr["cx"]],
                [0,          intr["fy"], intr["cy"]],
                [0,          0,          1],
            ], dtype=np.float64)
            FovX = float(2.0 * np.arctan(width  / (2.0 * intr["fx"])))
            FovY = float(2.0 * np.arctan(height / (2.0 * intr["fy"])))

            R_w2c = qvec2rotmat(colmap_img.qvec)
            t_w2c = colmap_img.tvec

            # Camera-to-world
            c2w = np.eye(4)
            c2w[:3, :3] = R_w2c.T
            c2w[:3, 3] = -R_w2c.T @ t_w2c

            # 3DGS convention
            R = R_w2c.T        # stored as c2w rotation
            T = t_w2c          # stored as w2c translation

            cam_id = FACE_TO_CAM_ID.get(face_name, 0)
            image = Image.open(str(img_file))

            # ── Load sky mask ─────────────────────────────────────────
            guidance: dict = {}
            if mask_path is not None:
                m_file = mask_path / colmap_img.name
                if not m_file.exists():
                    # Try same stem with different extension
                    stem = Path(colmap_img.name).stem
                    for ext in (".jpg", ".png", ".jpeg"):
                        candidate = mask_path / (stem + ext)
                        if candidate.exists():
                            m_file = candidate
                            break
                if m_file.exists():
                    guidance["mask"] = Image.open(str(m_file)).convert("L")

            # ── Load monocular depth ──────────────────────────────────
            if depth_path is not None:
                stem = Path(colmap_img.name).stem
                npy_file = depth_path / f"{stem}_depth_raw.npy"
                if npy_file.exists():
                    guidance["mono_depth"] = np.load(str(npy_file)).astype(np.float32)

            # ── Load moving object mask ───────────────────────────────
            if moving_mask_path is not None:
                mm_file = moving_mask_path / colmap_img.name
                if not mm_file.exists():
                    stem = Path(colmap_img.name).stem
                    for ext in (".jpg", ".png", ".jpeg"):
                        candidate = moving_mask_path / (stem + ext)
                        if candidate.exists():
                            mm_file = candidate
                            break
                if mm_file.exists():
                    guidance["moving_mask"] = Image.open(
                        str(mm_file)).convert("L")

            cam_info = CameraInfo(
                uid=uid, R=R, T=T,
                FovY=FovY, FovX=FovX, K=K,
                image=image,
                image_path=str(img_file),
                image_name=f"{frame_name}_{face_name}",
                width=width, height=height,
                metadata={
                    "frame_name": frame_name,
                    "face": face_name,
                    "frame_idx": frame_idx,
                    "ego_pose": c2w,
                    "cam": cam_id,
                },
                guidance=guidance,
            )
            uid += 1
            (test_cams if is_test else train_cams).append(cam_info)

    print(f"[Scene] Train cameras: {len(train_cams)}, "
          f"Test cameras: {len(test_cams)}")
    if len(train_cams) == 0:
        sys.exit("ERROR: No training cameras found!")

    # ── Load point cloud ──────────────────────────────────────────────
    assert pcd_path.exists(), f"Point cloud not found: {pcd_path}"
    pcd = read_ply(str(pcd_path))
    print(f"[Scene] Point cloud: {pcd.points.shape[0]:,} points "
          f"from {pcd_path}")

    # ── Outlier filtering & scene extent ──────────────────────────────
    centre = pcd.points.mean(axis=0)
    dists = np.linalg.norm(pcd.points - centre, axis=1)
    radius_pct = float(np.percentile(dists, 99))

    keep = dists < 5.0 * radius_pct
    n_removed = int((~keep).sum())
    if n_removed > 0:
        print(f"[Scene] Removed {n_removed:,} outlier points")
        pcd = BasicPointCloud(
            points=pcd.points[keep],
            colors=pcd.colors[keep],
            normals=pcd.normals[keep])

    print(f"[Scene] Scene radius: {radius_pct:.2f}")

    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cams,
        test_cameras=test_cams,
        scene_center=centre,
        scene_radius=radius_pct,
        ply_path=str(pcd_path))


# ═══════════════════════════════════════════════════════════════════════════
#  §9  Config
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_CFG = {
    "task": "gopromax_neighbour",
    "exp_name": "default",
    "output_root": "output",
    "gpus": [0],
    "source_path": "data/colmap_pointcloud_dense",

    "data": {
        "white_background": False,
        "split_test": 8,
        "point_cloud_path": "data/colmap_pointcloud_dense/fused.ply",
        "mask_dir": "data/cubemap_faces_mass13k",
        "moving_mask_dir": "data/cubemap_faces_sam_moving",
        "images": "data/cubemap_faces",
    },

    "model": {
        "sh_degree": 3,
    },

    "train": {
        "epochs": 180,
        "test_epochs": [30, 60, 90, 120, 150, 180],
        "save_epochs": [180],
        "checkpoint_epochs": [60, 120, 180],
    },

    "optim": {
        "position_lr_init": 0.00016,
        "position_lr_final": 1.6e-06,
        "position_lr_max_epochs": 135,
        "feature_lr": 0.0025,
        "opacity_lr": 0.05,
        "scaling_lr": 0.005,
        "rotation_lr": 0.001,
        "percent_dense": 0.01,
        "densification_interval": 100,
        "densify_from_epoch": 3,
        "densify_until_epoch": 90,
        "densify_grad_threshold": 0.0002,
        "min_opacity": 0.005,
        "max_screen_size": 20,
        "percent_big_ws": 0.1,
        "opacity_reset_epoch_interval": 18,
        "prune_epoch_interval": 30,
        "prune_min_opacity": 0.003,
        "lambda_l1": 1.0,
        "lambda_dssim": 0.2,
        "lambda_sky_acc": 0.01,
        "lambda_sh_reg": 0.001,
        "lambda_opacity_entropy": 0.0,
        "lambda_depth": 0.0,
        # COLMAP anchor (constrain splats to dense point cloud)
        "lambda_xyz_anchor": 0.005,
        "anchor_radius_scale": 0.05,        # × scene_radius
        "floater_radius_scale": 0.10,       # × scene_radius
        "floater_prune_epoch_interval": 10,
        "anchor_nn_refresh_interval": 50,   # iters between KDTree queries
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = _deep_merge(DEFAULT_CFG, user_cfg)
    return cfg


# ═══════════════════════════════════════════════════════════════════════════
#  §10  Output & Logging Helpers
# ═══════════════════════════════════════════════════════════════════════════

def prepare_output(cfg: dict, workspace: str):
    """Create output directories and return a TensorBoard writer (or None)."""
    task = cfg["task"]
    exp = cfg["exp_name"]
    output_root = cfg.get("output_root", "output")
    if not os.path.isabs(output_root):
        output_root = os.path.join(workspace, output_root)

    model_path = os.path.join(output_root, task, exp)
    os.makedirs(model_path, exist_ok=True)
    trained_model_dir = os.path.join(model_path, "trained_model")
    os.makedirs(trained_model_dir, exist_ok=True)
    point_cloud_dir = os.path.join(model_path, "point_cloud")
    os.makedirs(point_cloud_dir, exist_ok=True)
    log_images_dir = os.path.join(model_path, "log_images")
    os.makedirs(log_images_dir, exist_ok=True)

    record_dir = os.path.join(output_root, "record", task, exp)
    os.makedirs(record_dir, exist_ok=True)

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(record_dir)

    # Save config
    cfg_path = os.path.join(model_path, "config.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Save cfg_args (needed by render.py / viewers)
    with open(os.path.join(model_path, "cfg_args"), "w") as fp:
        fp.write(str(argparse.Namespace(
            sh_degree=cfg["model"]["sh_degree"],
            white_background=cfg["data"].get("white_background", False),
            source_path=cfg["source_path"],
            model_path=model_path,
        )))

    # CSV metric files for visualize_metrics.py
    train_csv_path = os.path.join(model_path, "train_metrics.csv")
    eval_csv_path = os.path.join(model_path, "eval_metrics.csv")

    return {
        "model_path": model_path,
        "trained_model_dir": trained_model_dir,
        "point_cloud_dir": point_cloud_dir,
        "log_images_dir": log_images_dir,
        "record_dir": record_dir,
        "tb_writer": tb_writer,
        "train_csv_path": train_csv_path,
        "eval_csv_path": eval_csv_path,
    }


def save_log_images(log_dir: str, epoch: int, gt, rendered, depth, acc,
                    gt_depth=None, gt_acc=None):
    """Save a 2x3 visualisation grid.

    Row 1: ground truth RGB | ground truth opacity | ground truth depth
    Row 2: predicted RGB    | predicted opacity    | predicted depth
    """
    import torchvision
    gt_np = gt.detach().cpu()
    rn_np = rendered.detach().cpu()
    h, w = gt_np.shape[-2], gt_np.shape[-1]

    def _prep_depth(t):
        """Squeeze a depth tensor to (1, H, W) float on CPU (or None)."""
        if t is None:
            return None
        t = t.detach().cpu().float() if hasattr(t, 'detach') else torch.as_tensor(t).float()
        t = t.squeeze()
        if t.ndim == 2:
            t = t.unsqueeze(0)
        return t

    def _depth_to_3ch(t, v_min=None, v_max=None, invert=False):
        """Normalise a depth tensor (any shape) to (3, H, W) in [0,1].

        Pixels that are non-finite or non-positive (e.g. sky == 0 or
        negative DA2 outputs) are treated as invalid and rendered black,
        while valid pixels are min/max normalised on a log1p scale.

        ``invert=False``: large value -> light (use this for inverse-depth /
            disparity inputs like Depth-Anything mono depth, where larger = near).
        ``invert=True``:  large value -> dark  (use this for true metric depth,
            where larger = far).

        If ``v_min``/``v_max`` (in log1p space) are provided they are used
        instead of per-image min/max so multiple depth maps share a scale.
        """
        if t is None:
            return torch.zeros(3, h, w)

        valid = torch.isfinite(t) & (t > 0)
        out = torch.zeros_like(t)
        if valid.any():
            v = torch.log1p(t[valid])
            lo = v.min() if v_min is None else v_min
            hi = v.max() if v_max is None else v_max
            denom = (hi - lo).clamp(min=1e-6)
            n = ((v - lo) / denom).clamp(0.0, 1.0)
            out[valid] = (1.0 - n) if invert else n
        return out.expand(3, -1, -1).contiguous()

    def _mask_to_3ch(t):
        """Normalise a mask/opacity tensor (any shape) to (3, H, W) in [0,1]."""
        if t is None:
            return torch.zeros(3, h, w)
        t = t.detach().cpu().float() if hasattr(t, 'detach') else torch.as_tensor(t).float()
        t = t.squeeze()
        if t.ndim == 2:
            t = t.unsqueeze(0)
        t = (t - t.min()) / (t.max() - t.min() + 1e-6)
        return t.expand(3, -1, -1).contiguous()

    gt_d_t   = _prep_depth(gt_depth)
    pred_d_t = _prep_depth(depth)

    # GT mono depth (Depth-Anything *_depth_raw.npy) is inverse-depth /
    # disparity-like: larger value = closer, so we do NOT invert.
    # Predicted depth is true metric depth: larger value = farther, so we
    # invert it. Both panels therefore use the convention light = near,
    # dark = far. Because the two have different units (disparity vs.
    # metric depth), they are normalised independently.
    gd = _depth_to_3ch(gt_d_t,   invert=False)
    ga = _mask_to_3ch(gt_acc)
    d  = _depth_to_3ch(pred_d_t, invert=True)
    a  = _mask_to_3ch(acc)

    # 2 rows x 3 cols: [gt_rgb, gt_opacity, gt_depth] / [pred_rgb, pred_opacity, pred_depth]
    grid = torchvision.utils.make_grid(
        [gt_np, ga, gd, rn_np, a, d], nrow=3, padding=2, normalize=False)
    path = os.path.join(log_dir, f"epoch_{epoch:04d}.png")
    torchvision.utils.save_image(grid, path)


@torch.no_grad()
def evaluate(
    test_cameras, gaussians, bg_color, epoch, tb_writer=None,
    eval_csv_path: str | None = None, split: str = "test",
    n_points: int | None = None,
    save_dir: str | None = None,
):
    """Run evaluation on test cameras and return mean metrics.

    If *save_dir* is given, rendered images are saved to
    ``{save_dir}/{split}/epoch_{epoch}/``.
    """
    if not test_cameras:
        return {}

    import cv2
    import torchvision

    l1_list, psnr_list, ssim_list, lpips_list = [], [], [], []

    # Prepare image output directory
    img_dir = None
    if save_dir is not None:
        img_dir = os.path.join(save_dir, split, f"epoch_{epoch}")
        os.makedirs(img_dir, exist_ok=True)

    lpips_fn = None
    if LPIPS_FOUND:
        lpips_fn = _lpips_module.LPIPS(net="vgg").cuda().eval()

    for cam in test_cameras:
        gt = cam.original_image.cuda(non_blocking=True)
        mask = cam.guidance.get("mask")
        if mask is not None:
            mask = mask.cuda(non_blocking=True)

        pkg = render(cam, gaussians, bg_color)
        image = pkg["rgb"].clamp(0.0, 1.0)

        l1_list.append(l1_loss(image, gt, mask).item())
        psnr_list.append(psnr(image, gt, mask).item())
        ssim_list.append(ssim(image, gt, mask=mask).item())

        if lpips_fn is not None:
            lpips_val = lpips_fn(
                image.unsqueeze(0), gt.unsqueeze(0)).item()
            lpips_list.append(lpips_val)

        # ── Save 2×3 grid: [GT RGB | GT mask | GT depth] / [Pred RGB | Pred alpha | Pred depth] ──
        if img_dir is not None:
            name = cam.image_name
            H, W = int(cam.image_height), int(cam.image_width)

            def _to_rgb_uint8(t: torch.Tensor) -> np.ndarray:
                """(C,H,W) or (H,W) tensor in [0,1] → (H,W,3) uint8."""
                arr = t.detach().cpu().numpy()
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                else:                          # (C,H,W)
                    arr = np.transpose(arr, (1, 2, 0))
                    if arr.shape[-1] == 1:
                        arr = np.repeat(arr, 3, axis=-1)
                    elif arr.shape[-1] > 3:
                        arr = arr[..., :3]
                return np.clip(arr * 255.0, 0, 255).astype(np.uint8)

            def _depth_to_grey(depth_np: np.ndarray,
                               valid: np.ndarray | None = None,
                               d_min: float | None = None,
                               d_max: float | None = None,
                               invert: bool = True) -> np.ndarray:
                """Normalize depth to greyscale RGB. Convention: light = near,
                dark = far, invalid pixels black.

                ``invert=True``  -> input is true metric depth (large = far),
                                    so we map (1 - normalised) to grey.
                ``invert=False`` -> input is inverse-depth / disparity
                                    (large = near, e.g. Depth-Anything mono
                                    *_depth_raw.npy), so we map directly.

                If ``d_min``/``d_max`` are provided, they are used instead of
                per-image min/max so multiple depth maps share a scale."""
                d = depth_np.astype(np.float32)
                out = np.zeros(d.shape, dtype=np.uint8)
                if valid is None:
                    valid = np.isfinite(d) & (d > 0)
                if valid.any():
                    if d_min is None or d_max is None:
                        d_valid = d[valid]
                        d_min = float(d_valid.min())
                        d_max = float(d_valid.max())
                    if d_max - d_min > 1e-6:
                        n = (d - d_min) / (d_max - d_min)
                        scaled = ((1.0 - n) if invert else n) * 255.0
                    else:
                        scaled = np.full_like(d, 255.0)
                    scaled = np.clip(scaled, 0, 255).astype(np.uint8)
                    out[valid] = scaled[valid]
                return np.stack([out] * 3, axis=-1)

            # ── GT panels ────────────────────────────────────────────
            gt_rgb_img = _to_rgb_uint8(gt[:3])

            gt_mask = cam.guidance.get("mask")
            if gt_mask is not None:
                m = gt_mask.detach().float().cpu().numpy().squeeze()
                gt_mask_img = np.stack(
                    [(np.clip(m, 0, 1) * 255).astype(np.uint8)] * 3, axis=-1)
            else:
                gt_mask_img = np.zeros((H, W, 3), dtype=np.uint8)

            mono_depth = cam.guidance.get("mono_depth")
            if mono_depth is not None:
                md = mono_depth.detach().cpu().numpy().squeeze() \
                    if torch.is_tensor(mono_depth) else np.asarray(mono_depth).squeeze()
            else:
                md = None

            # ── Predicted panels ─────────────────────────────────────
            pred_rgb_img = _to_rgb_uint8(image)

            acc_np = pkg["acc"].detach().permute(1, 2, 0).cpu().numpy().squeeze()
            pred_alpha_img = np.stack(
                [(np.clip(acc_np, 0, 1) * 255).astype(np.uint8)] * 3, axis=-1)

            depth_np = pkg["depth"].detach().permute(1, 2, 0).cpu().numpy().squeeze()
            pred_valid = acc_np > 1e-3

            # GT mono depth (Depth-Anything *_depth_raw.npy) is inverse-depth
            # (large value = near), predicted depth is metric depth (large
            # value = far). They have different units, so each is normalised
            # independently. Both panels share the convention
            # light = near, dark = far.
            if md is not None:
                gt_depth_img = _depth_to_grey(md, invert=False)
            else:
                gt_depth_img = np.full((H, W, 3), 255, dtype=np.uint8)

            pred_depth_img = _depth_to_grey(
                depth_np, valid=pred_valid, invert=True)

            # ── Compose 2×3 grid ────────────────────────────────────
            row1 = np.concatenate([gt_rgb_img,   gt_mask_img,   gt_depth_img], axis=1)
            row2 = np.concatenate([pred_rgb_img, pred_alpha_img, pred_depth_img], axis=1)
            grid = np.concatenate([row1, row2], axis=0)
            Image.fromarray(grid).save(os.path.join(img_dir, f"{name}.png"))

    metrics = {
        "l1_loss": np.mean(l1_list),
        "psnr": np.mean(psnr_list),
        "ssim": np.mean(ssim_list),
    }
    if lpips_list:
        metrics["lpips"] = np.mean(lpips_list)

    if n_points is not None:
        metrics["n_points"] = n_points

    lpips_str = (f"  LPIPS={metrics['lpips']:.4f}"
                 if "lpips" in metrics else "")
    print(f"  [EVAL {split} epoch {epoch}] "
          f"L1={metrics['l1_loss']:.4f}  "
          f"PSNR={metrics['psnr']:.2f}  "
          f"SSIM={metrics['ssim']:.4f}"
          f"{lpips_str}")

    if tb_writer is not None:
        for k, v in metrics.items():
            tb_writer.add_scalar(f"eval_{split}/{k}", v, epoch)

    # ── Append to eval CSV ────────────────────────────────────────
    if eval_csv_path is not None:
        file_exists = os.path.isfile(eval_csv_path)
        row = {"split": split, "epoch": epoch}
        row.update(metrics)
        fieldnames = ["split", "epoch", "l1_loss", "psnr", "ssim", "lpips", "n_points"]
        with open(eval_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
#  §11  Training Loop
# ═══════════════════════════════════════════════════════════════════════════

def training(cfg: dict):
    """Main training entry point."""
    workspace = os.getcwd()
    train_cfg = cfg["train"]
    optim_cfg = cfg["optim"]
    data_cfg = cfg["data"]

    # ── Output directories & logging ──────────────────────────────────
    dirs = prepare_output(cfg, workspace)
    tb_writer = dirs["tb_writer"]

    # ── GPU ───────────────────────────────────────────────────────────
    gpus = cfg.get("gpus", [0])
    if gpus and gpus[0] >= 0:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpus[0]))

    white_bg = data_cfg.get("white_background", False)
    bg_color = torch.tensor(
        [1, 1, 1] if white_bg else [0, 0, 0],
        dtype=torch.float32, device="cuda")

    # ── Dataset ───────────────────────────────────────────────────────
    scene_info = read_scene(
        source_path=cfg["source_path"],
        point_cloud_path=data_cfg["point_cloud_path"],
        images_dir=data_cfg["images"],
        mask_dir=data_cfg.get("mask_dir", ""),
        moving_mask_dir=data_cfg.get("moving_mask_dir", ""),
        depth_dir=data_cfg.get("depth_dir", ""),
        split_test=data_cfg.get("split_test", 8),
        workspace=workspace,
        max_frames=data_cfg.get("max_frames", 0),
    )

    # Save input PLY
    input_ply = os.path.join(dirs["model_path"], "input.ply")
    store_ply(input_ply,
              scene_info.point_cloud.points,
              scene_info.point_cloud.colors)

    # Save cameras.json
    def _fov2focal(fov, pixels):
        return pixels / (2.0 * math.tan(fov / 2.0))

    def _camera_info_to_json(cid, cam_info):
        Rt = np.eye(4)
        Rt[:3, :3] = cam_info.R.T
        Rt[:3, 3] = cam_info.T
        W2C = np.linalg.inv(Rt)
        pos = W2C[:3, 3]
        rot = W2C[:3, :3]
        return {
            'id': cid,
            'img_name': cam_info.image_name,
            'width': cam_info.width,
            'height': cam_info.height,
            'position': pos.tolist(),
            'rotation': [x.tolist() for x in rot],
            'fy': _fov2focal(cam_info.FovY, cam_info.height),
            'fx': _fov2focal(cam_info.FovX, cam_info.width),
        }

    json_cams = []
    all_cam_infos = list(scene_info.test_cameras) + list(scene_info.train_cameras)
    for cid, ci in enumerate(all_cam_infos):
        json_cams.append(_camera_info_to_json(cid, ci))
    cam_json_path = os.path.join(dirs["model_path"], "cameras.json")
    with open(cam_json_path, "w") as fp:
        json.dump(json_cams, fp)
    print(f"Saved cameras.json ({len(json_cams)} cameras) to {cam_json_path}")

    # Build Camera objects
    print("Loading training cameras …")
    train_cameras = [
        load_camera(ci) for ci in tqdm(scene_info.train_cameras)]
    print("Loading test cameras …")
    test_cameras = [
        load_camera(ci) for ci in tqdm(scene_info.test_cameras)]

    # ── Gaussian model ────────────────────────────────────────────────
    gaussians = GaussianModel(sh_degree=cfg["model"]["sh_degree"])
    gaussians.create_from_pcd(
        scene_info.point_cloud, scene_info.scene_radius)

    cams_per_epoch = len(train_cameras)
    num_epochs = train_cfg["epochs"]

    # Position LR schedule: convert epoch count → iteration count
    optim_cfg["position_lr_max_steps"] = (
        optim_cfg["position_lr_max_epochs"] * cams_per_epoch)

    gaussians.training_setup(optim_cfg)

    # ── Schedule parameters ───────────────────────────────────────────
    test_epochs_set = set(train_cfg.get("test_epochs", []))
    save_epochs_set = set(train_cfg.get("save_epochs", []))
    checkpoint_epochs_set = set(train_cfg.get("checkpoint_epochs", []))

    densify_from_epoch = optim_cfg["densify_from_epoch"]
    densify_until_epoch = optim_cfg["densify_until_epoch"]
    densification_interval = optim_cfg["densification_interval"]
    opacity_reset_interval = optim_cfg["opacity_reset_epoch_interval"]
    prune_interval = optim_cfg["prune_epoch_interval"]
    scene_extent = scene_info.scene_radius

    # ── COLMAP anchor (constrain splats to dense point cloud) ──────────
    lambda_xyz_anchor = optim_cfg.get("lambda_xyz_anchor", 0.005)
    anchor_radius = (
        optim_cfg.get("anchor_radius_scale", 0.05) * scene_extent)
    floater_radius = (
        optim_cfg.get("floater_radius_scale", 0.10) * scene_extent)
    floater_prune_interval = optim_cfg.get(
        "floater_prune_epoch_interval", 10)
    nn_refresh_interval = optim_cfg.get("anchor_nn_refresh_interval", 50)

    colmap_pts_np = np.ascontiguousarray(
        scene_info.point_cloud.points.astype(np.float32))
    colmap_kdtree = cKDTree(colmap_pts_np)
    colmap_pts_t = torch.as_tensor(colmap_pts_np, device="cuda")
    cached_nn_idx: "torch.Tensor | None" = None
    cached_nn_step = -10**9

    print(f"  anchor:      λ={lambda_xyz_anchor}  r={anchor_radius:.3f}  "
          f"floater_r={floater_radius:.3f}  "
          f"prune every {floater_prune_interval} ep")

    print(f"Training: {num_epochs} epochs × {cams_per_epoch} cameras/epoch")
    print(f"  densify:     epoch {densify_from_epoch}–{densify_until_epoch}")
    print(f"  opacity reset every {opacity_reset_interval} epochs")
    print(f"  prune every  {prune_interval} epochs (after densify)")
    print(f"  test at      epochs {sorted(test_epochs_set)}")
    print(f"  checkpoints  epochs {sorted(checkpoint_epochs_set)}")
    print(f"  save PLY     epochs {sorted(save_epochs_set)}")

    # ── Resume from checkpoint ────────────────────────────────────────
    start_epoch = 0
    step = 0
    ckpt_files = sorted(Path(dirs["trained_model_dir"]).glob("epoch_*.pth"))
    if ckpt_files:
        latest = ckpt_files[-1]
        state = torch.load(str(latest), weights_only=False)
        start_epoch = state.get("epoch", 0)
        step = state.get("step", start_epoch * cams_per_epoch)
        gaussians.load_state_dict(state)
        gaussians.training_setup(optim_cfg)
        if "optimizer_state" in state:
            gaussians.optimizer.load_state_dict(state["optimizer_state"])
        gaussians._sync_auxiliary_tensors()
        print(f"Resumed from {latest}  (epoch {start_epoch})")

    # ── CSV metric files ──────────────────────────────────────────────
    train_csv_path = dirs["train_csv_path"]
    eval_csv_path = dirs["eval_csv_path"]

    # Initialise train CSV header (only if file doesn't exist yet)
    if not os.path.isfile(train_csv_path):
        with open(train_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss", "l1_loss", "psnr", "ssim",
                             "ema_loss", "ema_psnr", "ema_ssim", "n_points"])

    # ── Training metrics ──────────────────────────────────────────────
    ema_loss = 0.0
    ema_psnr = 0.0
    ema_ssim = 0.0

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    import time as _time
    epoch_wall_start = _time.time()

    progress = tqdm(
        range(start_epoch, num_epochs),
        initial=start_epoch, total=num_epochs,
        desc="Epochs", unit="ep")

    # ══════════════════════════════════════════════════════════════════
    #  EPOCH LOOP
    # ══════════════════════════════════════════════════════════════════
    for epoch in range(start_epoch + 1, num_epochs + 1):
        epoch_wall_start = _time.time()

        viewpoint_stack = list(train_cameras)
        shuffle(viewpoint_stack)

        # ── Per-epoch accumulators for CSV logging ────────────────────
        epoch_loss_sum = 0.0
        epoch_l1_sum = 0.0
        epoch_psnr_sum = 0.0
        epoch_ssim_sum = 0.0
        epoch_count = 0

        # ── Increase SH degree every 3 epochs ────────────────────────
        if epoch % 3 == 0:
            gaussians.oneup_sh_degree()

        # ── Camera loop ───────────────────────────────────────────────
        for cam_idx, cam in enumerate(viewpoint_stack):
            step += 1

            iter_start.record()
            gaussians.update_learning_rate(step)

            gt_image = cam.original_image
            gt_image = (gt_image.cuda(non_blocking=True)
                        if not gt_image.is_cuda else gt_image)

            # ── Sky mask ──────────────────────────────────────────────
            sky_mask = None
            if "mask" in cam.guidance:
                sky_mask = cam.guidance["mask"]
                sky_mask = (sky_mask.cuda(non_blocking=True)
                            if not sky_mask.is_cuda else sky_mask)

            # ── Moving object mask ────────────────────────────────────
            moving_mask = None
            if "moving_mask" in cam.guidance:
                moving_mask = cam.guidance["moving_mask"]
                moving_mask = (moving_mask.cuda(non_blocking=True)
                               if not moving_mask.is_cuda else moving_mask)

            # ── Combined mask (exclude sky + moving objects) ──────────
            mask = sky_mask
            if mask is not None and moving_mask is not None:
                mask = mask & (~moving_mask)
            elif moving_mask is not None:
                mask = ~moving_mask

            # ── Render ────────────────────────────────────────────────
            render_pkg = render(cam, gaussians, bg_color)
            image = render_pkg["rgb"]
            acc = render_pkg["acc"]
            depth = render_pkg["depth"]
            viewspace_pts = render_pkg["viewspace_points"]
            visibility = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]

            scalar_dict: dict = {}

            # ── RGB loss (L1 + D-SSIM) ───────────────────────────────
            lambda_l1 = optim_cfg.get("lambda_l1", 1.0)
            lambda_dssim = optim_cfg.get("lambda_dssim", 0.2)

            Ll1 = l1_loss(image, gt_image, mask)
            scalar_dict["l1_loss"] = Ll1.item()

            loss = (
                (1.0 - lambda_dssim) * lambda_l1 * Ll1 +
                lambda_dssim * (1.0 - ssim(image, gt_image, mask=mask)))

            # ── SH regularisation ─────────────────────────────────────
            lambda_sh = optim_cfg.get("lambda_sh_reg", 1e-3)
            if lambda_sh > 0:
                sh_rest = gaussians._features_rest
                sh_reg = lambda_sh * (sh_rest ** 2).mean()
                scalar_dict["sh_reg_loss"] = sh_reg.item()
                loss += sh_reg

            # ── Sky opacity penalty ───────────────────────────────────
            #     Push acc → 0 in sky regions (where sky_mask == False)
            lambda_sky_acc = optim_cfg.get("lambda_sky_acc", 0.01)
            if lambda_sky_acc > 0 and sky_mask is not None:
                sky_region = 1.0 - sky_mask.float()     # 1 where sky
                sky_acc_loss = lambda_sky_acc * (acc * sky_region).mean()
                scalar_dict["sky_acc_loss"] = sky_acc_loss.item()
                loss += sky_acc_loss

            # ── Opacity entropy regularisation ─────────────────────
            #     Push opacity toward 0 or 1 to prevent semi-transparent
            #     layers stacking at incorrect depths.
            lambda_oe = optim_cfg.get("lambda_opacity_entropy", 0.0)
            if lambda_oe > 0:
                o = gaussians.get_opacity.clamp(1e-6, 1.0 - 1e-6)
                oe_loss = lambda_oe * -(o * o.log() + (1 - o) * (1 - o).log()).mean()
                scalar_dict["opacity_entropy_loss"] = oe_loss.item()
                loss += oe_loss

            # ── Monocular depth supervision (robust L1) ────────────
            lambda_depth = optim_cfg.get("lambda_depth", 0.0)
            if lambda_depth > 0 and "mono_depth" in cam.guidance:
                mono_depth = cam.guidance["mono_depth"]
                mono_depth = (mono_depth.cuda(non_blocking=True)
                              if not mono_depth.is_cuda else mono_depth)
                d_loss = lambda_depth * depth_mono_loss(
                    depth, mono_depth, acc, mask)
                scalar_dict["depth_loss"] = d_loss.item()
                loss += d_loss

            # ── COLMAP position anchor (soft) ────────────────────────────
            if lambda_xyz_anchor > 0:
                xyz = gaussians.get_xyz
                need_refresh = (
                    cached_nn_idx is None
                    or cached_nn_idx.shape[0] != xyz.shape[0]
                    or (step - cached_nn_step) >= nn_refresh_interval)
                if need_refresh:
                    xyz_np = xyz.detach().cpu().numpy()
                    _, nn_idx = colmap_kdtree.query(xyz_np, k=1)
                    cached_nn_idx = torch.as_tensor(
                        nn_idx, device="cuda", dtype=torch.long)
                    cached_nn_step = step
                nn_pts = colmap_pts_t[cached_nn_idx]
                d2 = ((xyz - nn_pts) ** 2).sum(-1)
                inside = d2 < (anchor_radius ** 2)
                if inside.any():
                    anchor_loss = lambda_xyz_anchor * d2[inside].mean()
                    scalar_dict["xyz_anchor_loss"] = anchor_loss.item()
                    loss = loss + anchor_loss

            scalar_dict["loss"] = loss.item()

            # ── SSIM / PSNR for logging (detached) ────────────────────
            with torch.no_grad():
                ssim_val = ssim(image, gt_image, mask=mask).item()
                psnr_val = psnr(image, gt_image, mask).item()
                scalar_dict["ssim"] = ssim_val
                scalar_dict["psnr"] = psnr_val

            loss.backward()

            iter_end.record()

            # ── Book-keeping (no grad) ────────────────────────────────
            with torch.no_grad():
                if cam_idx % 10 == 0:
                    ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
                    ema_psnr = 0.4 * psnr_val + 0.6 * ema_psnr
                    ema_ssim = 0.4 * ssim_val + 0.6 * ema_ssim
                    progress.set_postfix({
                        "Loss": f"{ema_loss:.6f}",
                        "PSNR": f"{ema_psnr:.2f}",
                        "SSIM": f"{ema_ssim:.4f}",
                        "#G": f"{gaussians.num_points:,}",
                    })

                # ── TensorBoard ───────────────────────────────────────
                if tb_writer and cam_idx % 10 == 0:
                    for k, v in scalar_dict.items():
                        tb_writer.add_scalar(f"train/{k}", v, step)

                # ── Adaptive density control ──────────────────────────
                if epoch <= densify_until_epoch:
                    gaussians.set_max_radii2D(radii, visibility)
                    gaussians.add_densification_stats(
                        viewspace_pts, visibility)

                    if (epoch > densify_from_epoch and
                            cam_idx % densification_interval == 0):
                        n_before, n_after = gaussians.densify_and_prune(
                            max_grad=optim_cfg["densify_grad_threshold"],
                            min_opacity=optim_cfg["min_opacity"],
                            max_screen_size=optim_cfg["max_screen_size"],
                            scene_extent=scene_extent,
                            percent_big_ws=optim_cfg.get("percent_big_ws", 0.1),
                        )
                        if n_before != n_after:
                            scalar_dict["n_gaussians"] = n_after
                            tqdm.write(
                                f"  [E{epoch} step {cam_idx}] "
                                f"densify: {n_before:,} → {n_after:,} "
                                f"(Δ {n_after - n_before:+,})")

                # ── Optimiser step ────────────────────────────────────
                gaussians.update_optimizer()

                # ── Accumulate epoch-level averages ──────────────
                epoch_loss_sum += scalar_dict["loss"]
                epoch_l1_sum += scalar_dict["l1_loss"]
                epoch_psnr_sum += psnr_val
                epoch_ssim_sum += ssim_val
                epoch_count += 1

        # ══════════════════════════════════════════════════════════════
        #  END OF EPOCH
        # ══════════════════════════════════════════════════════════════
        with torch.no_grad():
            progress.update(1)

            # ── Epoch summary log ─────────────────────────────────────
            if epoch_count > 0:
                epoch_wall_elapsed = _time.time() - epoch_wall_start
                avg_loss = epoch_loss_sum / epoch_count
                avg_l1 = epoch_l1_sum / epoch_count
                avg_psnr = epoch_psnr_sum / epoch_count
                avg_ssim = epoch_ssim_sum / epoch_count
                cur_lr = gaussians.optimizer.param_groups[0]["lr"]
                gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
                xyz = gaussians.get_xyz
                opac = gaussians.get_opacity
                scale = gaussians.get_scaling
                tqdm.write(
                    f"\n{'─'*72}\n"
                    f"  EPOCH {epoch}/{num_epochs}  "
                    f"({epoch_wall_elapsed:.1f}s, "
                    f"{epoch_count} cams)\n"
                    f"  Loss  │ total={avg_loss:.6f}  "
                    f"L1={avg_l1:.6f}  "
                    f"PSNR={avg_psnr:.2f}  "
                    f"SSIM={avg_ssim:.4f}\n"
                    f"  EMA   │ loss={ema_loss:.6f}  "
                    f"PSNR={ema_psnr:.2f}  "
                    f"SSIM={ema_ssim:.4f}\n"
                    f"  Gauss │ N={gaussians.num_points:,}  "
                    f"xyz=[{xyz.min().item():.2f}, "
                    f"{xyz.max().item():.2f}]  "
                    f"opacity=[{opac.min().item():.3f}, "
                    f"{opac.max().item():.3f}] "
                    f"mean={opac.mean().item():.3f}  "
                    f"scale_mean={scale.mean().item():.4f}\n"
                    f"  LR={cur_lr:.2e}  "
                    f"GPU={gpu_mem:.2f} GB  "
                    f"SH={gaussians.active_sh_degree}\n"
                    f"{'─'*72}")

            # ── Write train CSV row ───────────────────────────────────
            if epoch_count > 0:
                with open(train_csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch,
                        epoch_loss_sum / epoch_count,
                        epoch_l1_sum / epoch_count,
                        epoch_psnr_sum / epoch_count,
                        epoch_ssim_sum / epoch_count,
                        ema_loss,
                        ema_psnr,
                        ema_ssim,
                        gaussians.num_points,
                    ])

            # ── Save log images ───────────────────────────────────────
            if epoch % 20 == 0:
                try:
                    save_log_images(
                        dirs["log_images_dir"], epoch,
                        gt_image, image, depth, acc,
                        gt_depth=cam.guidance.get("mono_depth"),
                        gt_acc=cam.guidance.get("mask"))
                except Exception as e:
                    tqdm.write(f"  [E{epoch}] log image save failed: {e}")

            # ── Save PLY ──────────────────────────────────────────────
            if epoch in save_epochs_set:
                ply_path = os.path.join(
                    dirs["point_cloud_dir"],
                    f"iteration_{epoch}", "point_cloud.ply")
                gaussians.save_ply(ply_path)

            # ── Opacity reset ─────────────────────────────────────────
            if (epoch <= densify_until_epoch and
                    opacity_reset_interval > 0 and
                    epoch % opacity_reset_interval == 0):
                gaussians.reset_opacity()

            # ── Post-densify pruning ──────────────────────────────────
            if (epoch > densify_until_epoch and
                    prune_interval > 0 and
                    epoch % prune_interval == 0):
                prune_min = optim_cfg.get("prune_min_opacity", 0.003)
                prune_mask = (
                    gaussians.get_opacity < prune_min).squeeze(-1)
                if prune_mask.any():
                    n_before = gaussians.num_points
                    gaussians._prune_points(prune_mask)
                    # Reset aux tensors after pruning
                    gaussians.xyz_gradient_accum = torch.zeros(
                        (gaussians.num_points, 1), device="cuda")
                    gaussians.denom = torch.zeros(
                        (gaussians.num_points, 1), device="cuda")
                    gaussians.max_radii2D = torch.zeros(
                        gaussians.num_points, device="cuda")
                    torch.cuda.empty_cache()
                    print(f"\n[EPOCH {epoch}] Pruned "
                          f"{n_before - gaussians.num_points:,} "
                          f"low-opacity → {gaussians.num_points:,}")

            # ── Floater pruning (splats far from any COLMAP point) ──────
            if (epoch > densify_until_epoch and
                    floater_prune_interval > 0 and
                    epoch % floater_prune_interval == 0 and
                    floater_radius > 0):
                xyz_np = gaussians.get_xyz.detach().cpu().numpy()
                d_nn, _ = colmap_kdtree.query(xyz_np, k=1)
                floater_mask = torch.from_numpy(
                    d_nn > floater_radius).cuda()
                if floater_mask.any():
                    n_before = gaussians.num_points
                    gaussians._prune_points(floater_mask)
                    gaussians.xyz_gradient_accum = torch.zeros(
                        (gaussians.num_points, 1), device="cuda")
                    gaussians.denom = torch.zeros(
                        (gaussians.num_points, 1), device="cuda")
                    gaussians.max_radii2D = torch.zeros(
                        gaussians.num_points, device="cuda")
                    torch.cuda.empty_cache()
                    cached_nn_idx = None
                    print(f"\n[EPOCH {epoch}] Pruned "
                          f"{n_before - gaussians.num_points:,} floaters "
                          f"(>{floater_radius:.2f}) → "
                          f"{gaussians.num_points:,}")

            # ── Evaluation ────────────────────────────────────────────
            if epoch in test_epochs_set:
                evaluate(test_cameras, gaussians, bg_color,
                         epoch, tb_writer,
                         eval_csv_path=eval_csv_path, split="test",
                         n_points=gaussians.num_points,
                         save_dir=dirs["model_path"])
                evaluate(train_cameras[:len(test_cameras)],
                         gaussians, bg_color,
                         epoch, tb_writer,
                         eval_csv_path=eval_csv_path, split="train",
                         n_points=gaussians.num_points,
                         save_dir=dirs["model_path"])

            # ── Save checkpoint ───────────────────────────────────────
            is_final = (epoch == num_epochs)
            if epoch in checkpoint_epochs_set or is_final:
                sd = gaussians.save_state_dict(is_final=is_final)
                sd["epoch"] = epoch
                sd["step"] = step
                ckpt_path = os.path.join(
                    dirs["trained_model_dir"], f"epoch_{epoch}.pth")
                torch.save(sd, ckpt_path)
                print(f"\n[EPOCH {epoch}] Checkpoint → {ckpt_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  §12  Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GoPro Max Neighbour – Gaussian Splatting Training")
    parser.add_argument(
        "--config", default="configs/gopromax_neighbour.yaml",
        help="Path to YAML config file.")
    parser.add_argument(
        "--output-dir", default=None,
        help="Base directory for training outputs. Defaults to the config value or 'output'.")
    parser.add_argument(
        "--max_frames", type=int, default=None,
        help="Only load the first N frames for quick testing. Overrides config value.")
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="GPU device index to use (sets CUDA_VISIBLE_DEVICES).")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cfg = load_config(args.config)
    if args.output_dir is not None:
        cfg["output_root"] = args.output_dir
    if args.max_frames is not None:
        cfg.setdefault("data", {})["max_frames"] = args.max_frames

    print(f"Task: {cfg['task']}  Exp: {cfg['exp_name']}")

    # Reproducibility
    set_seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True

    torch.autograd.set_detect_anomaly(False)

    training(cfg)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
