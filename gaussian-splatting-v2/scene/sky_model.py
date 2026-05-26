#
# SH-based environment sky model for outdoor / unbounded scenes.
#
# The sky is represented as a single set of view-direction spherical-harmonic
# coefficients shared across the whole scene. At render time the SH is
# evaluated at each pixel's world-space ray direction to produce a sky RGB,
# which the caller composites behind the Gaussian render:
#
#     final_rgb = gaussian_rgb + (1 - alpha) * sky_rgb
#
# Sky-mask supervision (alpha -> 0 at pixels where the GT inverse-depth is 0)
# is the mechanism that frees the Gaussians from modelling the sky; the SH
# coefficients here are trained purely through the photometric loss on the
# composited image.
#

import math
import torch
import torch.nn as nn

from utils.sh_utils import eval_sh, RGB2SH
from utils.general_utils import get_expon_lr_func


class SkySHModel(nn.Module):
    """Single-environment SH sky.

    Parameters
    ----------
    sh_degree : int
        SH degree (0..3 supported here; 3 -> 16 coefficients per channel,
        48 params total). Degree 3 gives smooth low-frequency sky gradients;
        bump to 4 if you need slightly more directional structure.
    init_rgb : tuple[float, float, float]
        DC initialization in linear [0, 1] RGB. The default mid-gray (0.5)
        encodes to SH_dc = 0 so optimisation starts from a neutral sky.
    """

    def __init__(self, sh_degree: int = 3, init_rgb=(0.5, 0.5, 0.5)):
        super().__init__()
        assert 0 <= sh_degree <= 8, "SH degree must be in [0, 8]"
        self.sh_degree = sh_degree
        self.num_coeffs = (sh_degree + 1) ** 2

        coeffs = torch.zeros(3, self.num_coeffs)
        init_rgb_t = torch.tensor(init_rgb, dtype=torch.float32)
        coeffs[:, 0] = RGB2SH(init_rgb_t)
        self.sh = nn.Parameter(coeffs)

        self.optimizer = None
        self._lr_sched = None

        # Cached per-pixel camera-space ray grid, keyed by (H, W, tanFovx, tanFovy).
        # Rebuilt only when the camera geometry changes between iters.
        self._cached_key = None
        self._cached_dir_cam = None

    # ------------------------------------------------------------------
    # Optim
    # ------------------------------------------------------------------
    def training_setup(self, lr_init: float = 0.01, lr_final: float = 1e-4,
                       max_steps: int = 30000):
        self.optimizer = torch.optim.Adam(
            [{"params": [self.sh], "lr": lr_init, "name": "sky_sh"}],
            eps=1e-15,
        )
        self._lr_sched = get_expon_lr_func(
            lr_init=lr_init, lr_final=lr_final, max_steps=max_steps
        )

    def update_learning_rate(self, iteration: int):
        if self.optimizer is None or self._lr_sched is None:
            return
        lr = self._lr_sched(iteration)
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def step(self):
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _ray_dirs_camera(self, H: int, W: int, tan_fovx: float,
                        tan_fovy: float, device, dtype):
        key = (H, W, float(tan_fovx), float(tan_fovy), device, dtype)
        if self._cached_key == key and self._cached_dir_cam is not None:
            return self._cached_dir_cam

        u = (torch.arange(W, device=device, dtype=dtype) + 0.5)
        v = (torch.arange(H, device=device, dtype=dtype) + 0.5)
        x_ndc = (2.0 * u / W - 1.0) * tan_fovx              # (W,)
        y_ndc = (2.0 * v / H - 1.0) * tan_fovy              # (H,)
        grid_x, grid_y = torch.meshgrid(x_ndc, y_ndc, indexing="xy")
        # OpenCV / 3DGS camera frame: x right, y down, z forward.
        dir_cam = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1)
        dir_cam = dir_cam / dir_cam.norm(dim=-1, keepdim=True)  # (H, W, 3)
        self._cached_key = key
        self._cached_dir_cam = dir_cam
        return dir_cam

    def forward(self, camera) -> torch.Tensor:
        """Render the sky RGB for a given Camera. Returns (3, H, W) in [0, 1]."""
        H, W = int(camera.image_height), int(camera.image_width)
        tan_fovx = math.tan(camera.FoVx * 0.5)
        tan_fovy = math.tan(camera.FoVy * 0.5)

        device = self.sh.device
        dtype = self.sh.dtype
        dir_cam = self._ray_dirs_camera(H, W, tan_fovx, tan_fovy, device, dtype)

        # world_view_transform is stored as the row-vector world->view matrix
        # (point_view = point_world @ wvt). Its top-left 3x3 equals the
        # camera-to-world rotation R_c2w. Apply via dir_world = dir_cam @ R_c2w.T.
        R_c2w = camera.world_view_transform[:3, :3].to(device=device, dtype=dtype)
        dir_world = dir_cam @ R_c2w.t()                                 # (H, W, 3)

        # eval_sh expects sh shape [..., C, (deg+1)^2] and dirs [..., 3].
        sh_b = self.sh.unsqueeze(0).unsqueeze(0).expand(
            H, W, 3, self.num_coeffs
        )
        sky = eval_sh(self.sh_degree, sh_b, dir_world)                  # (H, W, 3)
        # Convention matches GaussianModel SH: RGB = eval_sh + 0.5.
        sky = (sky + 0.5).clamp(0.0, 1.0)
        return sky.permute(2, 0, 1).contiguous()                        # (3, H, W)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    def save(self, path: str):
        torch.save(
            {"sh": self.sh.detach().cpu(), "sh_degree": self.sh_degree},
            path,
        )

    def load(self, path: str, strict: bool = True):
        state = torch.load(path, map_location=self.sh.device)
        if strict:
            assert state["sh_degree"] == self.sh_degree, (
                f"sky_model degree mismatch: ckpt={state['sh_degree']} "
                f"vs model={self.sh_degree}"
            )
        with torch.no_grad():
            self.sh.copy_(state["sh"].to(device=self.sh.device, dtype=self.sh.dtype))
