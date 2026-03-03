"""Thin wrapper around the ``yacs`` package.

street_gaussians vendors its own copy of yacs.  We just depend on the
pip-installable ``yacs`` package and re-export what the rest of the
code-base expects (``CfgNode`` and the module-level ``load_cfg``).
"""

from yacs.config import CfgNode  # noqa: F401
import yaml


def load_cfg(cfg_file_or_str):
    """Load a :class:`CfgNode` from an open file object or a YAML string."""
    if hasattr(cfg_file_or_str, "read"):
        d = yaml.safe_load(cfg_file_or_str)
    else:
        d = yaml.safe_load(cfg_file_or_str)
    return CfgNode(d)
