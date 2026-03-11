"""
PLY point-cloud I/O helpers.

Reads dense ``fused.ply`` files produced by COLMAP and sparse point clouds
stored as COLMAP ``points3D.bin``.
"""

from __future__ import annotations

import numpy as np

from lib.datasets.base_readers import BasicPointCloud


def read_fused_ply(path: str) -> BasicPointCloud:
    """Read a COLMAP ``fused.ply``, auto-detecting whether normals exist.

    Returns a :class:`BasicPointCloud` with ``points``, ``colors`` in
    [0, 1], and ``normals`` (zero-filled when not present in the file).
    """
    has_normals = False
    n_vertices = 0

    with open(path, "rb") as f:
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            if line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
            if "nx" in line or "normal_x" in line:
                has_normals = True
            if line == "end_header":
                header_bytes = f.tell()
                break

    if has_normals:
        dtype = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ])
    else:
        dtype = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ])

    with open(path, "rb") as f:
        f.seek(header_bytes)
        buf = np.fromfile(f, dtype=dtype, count=n_vertices)

    pts = np.column_stack([buf["x"], buf["y"], buf["z"]])
    rgb = np.column_stack([buf["red"], buf["green"], buf["blue"]]) / 255.0
    nrm = (np.column_stack([buf["nx"], buf["ny"], buf["nz"]])
           if has_normals else np.zeros_like(pts))

    print(f"[GoPro360] Loaded {len(pts):,} points from {path}")
    return BasicPointCloud(points=pts, colors=rgb, normals=nrm)


def read_sparse_points(points3D: dict) -> BasicPointCloud:
    """Convert a dict of :class:`ColmapPoint3D` to a :class:`BasicPointCloud`.

    Parameters
    ----------
    points3D : dict
        Mapping ``{point3D_id: ColmapPoint3D}`` as returned by
        :func:`~gopro360.lib.utils.colmap_utils.read_points3D_binary`.
    """
    if not points3D:
        return BasicPointCloud(
            points=np.zeros((0, 3)),
            colors=np.zeros((0, 3)),
            normals=np.zeros((0, 3)),
        )
    pts = np.array([p.xyz for p in points3D.values()])
    rgb = np.array([p.rgb for p in points3D.values()]) / 255.0
    nrm = np.zeros_like(pts)
    print(f"[GoPro360] Loaded {len(pts):,} sparse points")
    return BasicPointCloud(points=pts, colors=rgb, normals=nrm)
