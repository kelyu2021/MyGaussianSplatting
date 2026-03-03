"""
StreetGaussianRenderer – render composition of background + sky + colour
correction.

Simplified: object rendering returns empty (no actors).
"""

import torch
from lib.utils.sh_utils import eval_sh
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.utils.camera_utils import Camera, make_rasterizer
from lib.config import cfg


class StreetGaussianRenderer:
    def __init__(self):
        self.cfg = cfg.render

    # ------------------------------------------------------------------
    # public entry points
    # ------------------------------------------------------------------
    def render_all(
        self,
        viewpoint_camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python=None,
        compute_cov3D_python=None,
        scaling_modifier=None,
        override_color=None,
    ):
        render_composition = self.render(
            viewpoint_camera, pc, convert_SHs_python,
            compute_cov3D_python, scaling_modifier, override_color,
        )
        render_background = self.render_background(
            viewpoint_camera, pc, convert_SHs_python,
            compute_cov3D_python, scaling_modifier, override_color,
        )
        render_object = self.render_object(
            viewpoint_camera, pc, convert_SHs_python,
            compute_cov3D_python, scaling_modifier, override_color,
        )

        result = render_composition
        result['rgb_background'] = render_background['rgb']
        result['acc_background'] = render_background['acc']
        result['rgb_object'] = render_object['rgb']
        result['acc_object'] = render_object['acc']
        return result

    def render_object(
        self,
        viewpoint_camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python=None,
        compute_cov3D_python=None,
        scaling_modifier=None,
        override_color=None,
        parse_camera_again: bool = True,
    ):
        """No objects – return empty white image."""
        pc.set_visibility(include_list=[])
        if parse_camera_again:
            pc.parse_camera(viewpoint_camera)
        result = self.render_kernel(
            viewpoint_camera, pc, convert_SHs_python,
            compute_cov3D_python, scaling_modifier, override_color,
            white_background=True,
        )
        return result

    def render_background(
        self,
        viewpoint_camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python=None,
        compute_cov3D_python=None,
        scaling_modifier=None,
        override_color=None,
        parse_camera_again: bool = True,
    ):
        pc.set_visibility(include_list=['background'])
        if parse_camera_again:
            pc.parse_camera(viewpoint_camera)
        result = self.render_kernel(
            viewpoint_camera, pc, convert_SHs_python,
            compute_cov3D_python, scaling_modifier, override_color,
            white_background=True,
        )
        return result

    def render_sky(
        self,
        viewpoint_camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python=None,
        compute_cov3D_python=None,
        scaling_modifier=None,
        override_color=None,
        parse_camera_again: bool = True,
    ):
        pc.set_visibility(include_list=['sky'])
        if parse_camera_again:
            pc.parse_camera(viewpoint_camera)
        result = self.render_kernel(
            viewpoint_camera, pc, convert_SHs_python,
            compute_cov3D_python, scaling_modifier, override_color,
        )
        return result

    def render(
        self,
        viewpoint_camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python=None,
        compute_cov3D_python=None,
        scaling_modifier=None,
        override_color=None,
        exclude_list=[],
    ):
        include_list = list(set(pc.model_name_id.keys()) - set(exclude_list))

        # Step 1: render foreground (background gaussians)
        pc.set_visibility(include_list)
        pc.parse_camera(viewpoint_camera)

        result = self.render_kernel(
            viewpoint_camera, pc, convert_SHs_python,
            compute_cov3D_python, scaling_modifier, override_color,
        )

        # Step 2: composite sky
        if pc.include_sky:
            sky_color = pc.sky_cubemap(viewpoint_camera, result['acc'].detach())
            result['rgb'] = result['rgb'] + sky_color * (1 - result['acc'])

        # Step 3: colour correction
        if pc.use_color_correction:
            result['rgb'] = pc.color_correction(viewpoint_camera, result['rgb'])

        if cfg.mode != 'train':
            result['rgb'] = torch.clamp(result['rgb'], 0., 1.)

        return result

    # ------------------------------------------------------------------
    # core rasterisation kernel
    # ------------------------------------------------------------------
    def render_kernel(
        self,
        viewpoint_camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python=None,
        compute_cov3D_python=None,
        scaling_modifier=None,
        override_color=None,
        white_background=cfg.data.white_background,
    ):
        try:
            means3D = pc.get_xyz
            num_gaussians = len(means3D)
        except Exception:
            num_gaussians = 0

        if num_gaussians == 0:
            H, W = int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)
            fill = 1.0 if white_background else 0.0
            return {
                'rgb': torch.full((3, H, W), fill, device='cuda'),
                'acc': torch.zeros(1, H, W, device='cuda'),
                'semantic': torch.zeros(0, H, W, device='cuda'),
            }

        # rasteriser setup
        bg_color = [1, 1, 1] if white_background else [0, 0, 0]
        bg_color = torch.tensor(bg_color).float().cuda()
        scaling_modifier = scaling_modifier or self.cfg.scaling_modifier
        rasterizer = make_rasterizer(
            viewpoint_camera, pc.max_sh_degree, bg_color, scaling_modifier,
        )

        convert_SHs_python = convert_SHs_python or self.cfg.convert_SHs_python
        compute_cov3D_python = compute_cov3D_python or self.cfg.compute_cov3D_python

        # screen-space points for gradient tracking
        if cfg.mode == 'train':
            screenspace_points = torch.zeros(
                (num_gaussians, 3), requires_grad=True, device='cuda',
            ).float() + 0
            try:
                screenspace_points.retain_grad()
            except Exception:
                pass
        else:
            screenspace_points = None

        means2D = screenspace_points
        opacity = pc.get_opacity

        # covariance
        scales = rotations = cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

        # colours
        shs = colors_precomp = None
        if override_color is None:
            if convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(
                    -1, 3, (pc.max_sh_degree + 1) ** 2,
                )
                dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                    pc.get_features.shape[0], 1,
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                try:
                    shs = pc.get_features
                except Exception:
                    colors_precomp = pc.get_colors(viewpoint_camera.camera_center)
        else:
            colors_precomp = override_color

        # extra features (semantic, normals)
        feature_names, feature_dims, features = [], [], []
        if cfg.data.get('use_semantic', False):
            semantics = pc.get_semantic
            feature_names.append('semantic')
            feature_dims.append(semantics.shape[-1])
            features.append(semantics)

        features = torch.cat(features, dim=-1) if features else None

        # rasterise
        rendered_color, radii, rendered_depth, rendered_acc, rendered_feature = rasterizer(
            means3D=means3D,
            means2D=means2D,
            opacities=opacity,
            shs=shs,
            colors_precomp=colors_precomp,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            semantics=features,
        )

        if cfg.mode != 'train':
            rendered_color = torch.clamp(rendered_color, 0., 1.)

        # unpack extra features
        rendered_feature_dict = dict()
        if rendered_feature.shape[0] > 0:
            rendered_feature_list = torch.split(rendered_feature, feature_dims, dim=0)
            for i, name in enumerate(feature_names):
                rendered_feature_dict[name] = rendered_feature_list[i]

        if 'semantic' in rendered_feature_dict:
            sem = rendered_feature_dict['semantic']
            semantic_mode = cfg.model.gaussian.get('semantic_mode', 'logits')
            if semantic_mode == 'probabilities':
                sem = sem / (sem.sum(dim=0, keepdim=True) + 1e-8)
                sem = torch.log(sem + 1e-8)
            rendered_feature_dict['semantic'] = sem

        result = {
            'rgb': rendered_color,
            'acc': rendered_acc,
            'depth': rendered_depth,
            'viewspace_points': screenspace_points,
            'visibility_filter': radii > 0,
            'radii': radii,
        }
        result.update(rendered_feature_dict)
        return result
