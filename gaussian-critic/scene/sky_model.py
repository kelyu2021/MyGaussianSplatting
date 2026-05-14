import torch
from utils.sh_utils import eval_sh


class SkyModel(torch.nn.Module):
    """Learnable spherical harmonic environment/sky model.

    Represents the sky (infinite-distance background) as a set of SH
    coefficients optimized jointly with the Gaussian scene.  A degree-3
    model uses 16 coefficients per RGB channel (48 parameters total).

    Usage::

        sky = SkyModel(sh_degree=3).cuda()
        # dirs: (N, 3) unit world-space ray directions
        colors = sky(dirs)  # (N, 3) non-negative RGB
    """

    def __init__(self, sh_degree: int = 3):
        super().__init__()
        n_coeffs = (sh_degree + 1) ** 2
        self.sh_coeffs = torch.nn.Parameter(torch.zeros(3, n_coeffs))
        self.sh_degree = sh_degree

    def forward(self, dirs: torch.Tensor) -> torch.Tensor:
        """Evaluate sky color for a batch of directions.

        Args:
            dirs: ``(N, 3)`` unit world-space directions (float32, CUDA).

        Returns:
            ``(N, 3)`` non-negative RGB sky colors.
        """
        # Expand coeffs to (N, 3, n_coeffs) for eval_sh
        sh = self.sh_coeffs.unsqueeze(0).expand(dirs.shape[0], -1, -1)
        return torch.clamp_min(eval_sh(self.sh_degree, sh, dirs) + 0.5, 0.0)
