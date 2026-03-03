"""
StreetGaussianModel – scene-level model (background + optional sky).

Simplified from street_gaussians: all actor/object code removed.
Only background, sky-cubemap, colour-correction and pose-correction remain.
"""

import torch
import torch.nn as nn
import numpy as np
import os
from lib.config import cfg
from lib.utils.general_utils import (
    build_scaling_rotation,
    strip_symmetric,
    startswith_any,
    matrix_to_quaternion,
)
from lib.utils.graphics_utils import BasicPointCloud
from lib.utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from lib.models.gaussian_model import GaussianModel
from lib.models.gaussian_model_bkgd import GaussianModelBkgd
from lib.utils.camera_utils import Camera
from lib.utils.sh_utils import eval_sh
from lib.models.sky_cubemap import SkyCubeMap
from lib.models.color_correction import ColorCorrection
from lib.models.camera_pose import PoseCorrection


class StreetGaussianModel(nn.Module):
    def __init__(self, metadata):
        super().__init__()
        self.metadata = metadata

        self.max_sh_degree = cfg.model.gaussian.sh_degree
        self.active_sh_degree = self.max_sh_degree

        # background
        self.include_background = cfg.model.nsg.get('include_bkgd', True)

        # sky (cube-map based)
        self.include_sky = cfg.model.nsg.get('include_sky', False)
        if self.include_sky:
            assert cfg.data.white_background is False

        # colour correction
        self.use_color_correction = cfg.model.use_color_correction
        # camera pose correction
        self.use_pose_correction = cfg.model.use_pose_correction

        self.setup_functions()

    # ------------------------------------------------------------------
    # visibility helpers (kept for renderer compatibility)
    # ------------------------------------------------------------------
    def set_visibility(self, include_list):
        self.include_list = include_list

    def get_visibility(self, model_name):
        if model_name == 'background':
            return model_name in self.include_list and self.include_background
        elif model_name == 'sky':
            return model_name in self.include_list and self.include_sky
        else:
            return False

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def setup_functions(self):
        self.model_name_id = dict()
        self.models_num = 0

        # background
        if self.include_background:
            self.background = GaussianModelBkgd(
                model_name='background',
                scene_center=self.metadata.get('scene_center', np.zeros(3)),
                scene_radius=self.metadata.get('scene_radius', 20.0),
                sphere_center=self.metadata.get('sphere_center', np.zeros(3)),
                sphere_radius=self.metadata.get('sphere_radius', 20.0),
            )
            self.model_name_id['background'] = 0
            self.models_num += 1

        # sky cube-map
        if self.include_sky:
            self.sky_cubemap = SkyCubeMap()
        else:
            self.sky_cubemap = None

        # colour correction
        if self.use_color_correction:
            self.color_correction = ColorCorrection(self.metadata)
        else:
            self.color_correction = None

        # pose correction
        if self.use_pose_correction:
            self.pose_correction = PoseCorrection(self.metadata)
        else:
            self.pose_correction = None

    # ------------------------------------------------------------------
    # create / save / load
    # ------------------------------------------------------------------
    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        for model_name in self.model_name_id.keys():
            model: GaussianModel = getattr(self, model_name)
            model.create_from_pcd(pcd, spatial_lr_scale)

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        plydata_list = []
        for model_name in self.model_name_id.keys():
            model: GaussianModel = getattr(self, model_name)
            plydata = model.make_ply()
            plydata = PlyElement.describe(plydata, f'vertex_{model_name}')
            plydata_list.append(plydata)
        PlyData(plydata_list).write(path)

    def load_ply(self, path):
        plydata_list = PlyData.read(path).elements
        for plydata in plydata_list:
            model_name = plydata.name[7:]  # strip 'vertex_'
            if model_name in self.model_name_id:
                print('Loading model', model_name)
                model: GaussianModel = getattr(self, model_name)
                model.load_ply(path=None, input_ply=plydata)
                plydata_list = PlyData.read(path).elements
        self.active_sh_degree = self.max_sh_degree

    def save_state_dict(self, is_final, exclude_list=[]):
        state_dict = dict()
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)
            state_dict[model_name] = model.state_dict(is_final)
        if self.sky_cubemap is not None:
            state_dict['sky_cubemap'] = self.sky_cubemap.save_state_dict(is_final)
        if self.color_correction is not None:
            state_dict['color_correction'] = self.color_correction.save_state_dict(is_final)
        if self.pose_correction is not None:
            state_dict['pose_correction'] = self.pose_correction.save_state_dict(is_final)
        return state_dict

    def load_state_dict(self, state_dict, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)
            model.load_state_dict(state_dict[model_name])
        if self.sky_cubemap is not None and 'sky_cubemap' in state_dict:
            self.sky_cubemap.load_state_dict(state_dict['sky_cubemap'])
        if self.color_correction is not None and 'color_correction' in state_dict:
            self.color_correction.load_state_dict(state_dict['color_correction'])
        if self.pose_correction is not None and 'pose_correction' in state_dict:
            self.pose_correction.load_state_dict(state_dict['pose_correction'])

    # ------------------------------------------------------------------
    # parse_camera – build per-frame graph (background only, no objects)
    # ------------------------------------------------------------------
    def parse_camera(self, camera: Camera):
        self.viewpoint_camera = camera
        self.background.set_background_mask(camera)

        self.frame = camera.meta.get('frame', 0)
        self.frame_idx = camera.meta.get('frame_idx', 0)

        self.num_gaussians = 0
        self.graph_gaussian_range = dict()
        idx = 0

        if self.get_visibility('background'):
            n = self.background.get_xyz.shape[0]
            self.num_gaussians += n
            self.graph_gaussian_range['background'] = [idx, idx + n]
            idx += n

    # ------------------------------------------------------------------
    # per-gaussian properties
    # ------------------------------------------------------------------
    @property
    def get_scaling(self):
        parts = []
        if self.get_visibility('background'):
            parts.append(self.background.get_scaling)
        return torch.cat(parts, dim=0) if parts else torch.empty(0, 3, device='cuda')

    @property
    def get_rotation(self):
        parts = []
        if self.get_visibility('background'):
            rot = self.background.get_rotation
            if self.use_pose_correction:
                rot = self.pose_correction.correct_gaussian_rotation(self.viewpoint_camera, rot)
            parts.append(rot)
        return torch.cat(parts, dim=0) if parts else torch.empty(0, 4, device='cuda')

    @property
    def get_xyz(self):
        parts = []
        if self.get_visibility('background'):
            xyz = self.background.get_xyz
            if self.use_pose_correction:
                xyz = self.pose_correction.correct_gaussian_xyz(self.viewpoint_camera, xyz)
            parts.append(xyz)
        return torch.cat(parts, dim=0) if parts else torch.empty(0, 3, device='cuda')

    @property
    def get_features(self):
        parts = []
        if self.get_visibility('background'):
            parts.append(self.background.get_features)
        return torch.cat(parts, dim=0) if parts else torch.empty(0, 0, 0, device='cuda')

    def get_colors(self, camera_center):
        colors = []
        if self.get_visibility('background'):
            model = self.background
            sh_dim = (model.max_sh_degree + 1) ** 2
            shs = model.get_features.transpose(1, 2).view(-1, 3, sh_dim)
            directions = model.get_xyz - camera_center
            directions = directions / torch.norm(directions, dim=1, keepdim=True)
            sh2rgb = eval_sh(model.max_sh_degree, shs, directions)
            color = torch.clamp_min(sh2rgb + 0.5, 0.0)
            colors.append(color)
        return torch.cat(colors, dim=0) if colors else torch.empty(0, 3, device='cuda')

    @property
    def get_semantic(self):
        parts = []
        if self.get_visibility('background'):
            parts.append(self.background.get_semantic)
        return torch.cat(parts, dim=0) if parts else torch.empty(0, device='cuda')

    @property
    def get_opacity(self):
        parts = []
        if self.get_visibility('background'):
            parts.append(self.background.get_opacity)
        return torch.cat(parts, dim=0) if parts else torch.empty(0, 1, device='cuda')

    def get_covariance(self, scaling_modifier=1):
        scaling = self.get_scaling
        rotation = self.get_rotation
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        return strip_symmetric(actual_covariance)

    # ------------------------------------------------------------------
    # training lifecycle
    # ------------------------------------------------------------------
    def oneupSHdegree(self, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            if model_name in exclude_list:
                continue
            getattr(self, model_name).oneupSHdegree()
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def training_setup(self, exclude_list=[]):
        self.active_sh_degree = 0
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            getattr(self, model_name).training_setup()
        if self.sky_cubemap is not None:
            self.sky_cubemap.training_setup()
        if self.color_correction is not None:
            self.color_correction.training_setup()
        if self.pose_correction is not None:
            self.pose_correction.training_setup()

    def update_learning_rate(self, iteration, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            getattr(self, model_name).update_learning_rate(iteration)
        if self.sky_cubemap is not None:
            self.sky_cubemap.update_learning_rate(iteration)
        if self.color_correction is not None:
            self.color_correction.update_learning_rate(iteration)
        if self.pose_correction is not None:
            self.pose_correction.update_learning_rate(iteration)

    def update_optimizer(self, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            getattr(self, model_name).update_optimizer()
        if self.sky_cubemap is not None:
            self.sky_cubemap.update_optimizer()
        if self.color_correction is not None:
            self.color_correction.update_optimizer()
        if self.pose_correction is not None:
            self.pose_correction.update_optimizer()

    def set_max_radii2D(self, radii, visibility_filter):
        radii = radii.float()
        for model_name in self.graph_gaussian_range.keys():
            model: GaussianModel = getattr(self, model_name)
            start, end = self.graph_gaussian_range[model_name]
            vis = visibility_filter[start:end]
            r = radii[start:end]
            model.max_radii2D[vis] = torch.max(model.max_radii2D[vis], r[vis])

    def add_densification_stats(self, viewspace_point_tensor, visibility_filter):
        grad = viewspace_point_tensor.grad
        for model_name in self.graph_gaussian_range.keys():
            model: GaussianModel = getattr(self, model_name)
            start, end = self.graph_gaussian_range[model_name]
            vis = visibility_filter[start:end]
            g = grad[start:end]
            model.xyz_gradient_accum[vis, 0:1] += torch.norm(g[vis, :2], dim=-1, keepdim=True)
            model.xyz_gradient_accum[vis, 1:2] += torch.norm(g[vis, 2:], dim=-1, keepdim=True)
            model.denom[vis] += 1

    def densify_and_prune(self, max_grad, min_opacity, prune_big_points, exclude_list=[]):
        scalars = None
        tensors = None
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)
            s, t = model.densify_and_prune(max_grad, min_opacity, prune_big_points)
            if model_name == 'background':
                scalars, tensors = s, t
        return scalars, tensors

    def reset_opacity(self, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            getattr(self, model_name).reset_opacity()
