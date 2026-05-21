"""
COLMAP binary model readers.

Read ``cameras.bin``, ``images.bin``, and ``points3D.bin`` from a COLMAP
sparse reconstruction and return structured namedtuples.
"""

from __future__ import annotations

import struct
import collections
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
#  Camera model definitions
# ═══════════════════════════════════════════════════════════════════════════

CAMERA_MODEL_IDS = {
    0: "SIMPLE_PINHOLE",   # f, cx, cy
    1: "PINHOLE",          # fx, fy, cx, cy
    2: "SIMPLE_RADIAL",    # f, cx, cy, k
    3: "RADIAL",           # f, cx, cy, k1, k2
    4: "OPENCV",           # fx, fy, cx, cy, k1, k2, p1, p2
}

CAMERA_MODEL_NUM_PARAMS = {
    0: 3, 1: 4, 2: 4, 3: 5, 4: 8,
}


# ═══════════════════════════════════════════════════════════════════════════
#  Namedtuples
# ═══════════════════════════════════════════════════════════════════════════

ColmapCamera = collections.namedtuple(
    "ColmapCamera", ["id", "model", "width", "height", "params"]
)
ColmapImage = collections.namedtuple(
    "ColmapImage", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
ColmapPoint3D = collections.namedtuple(
    "ColmapPoint3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)


# ═══════════════════════════════════════════════════════════════════════════
#  Low-level binary helpers
# ═══════════════════════════════════════════════════════════════════════════

def _read_next_bytes(f, num_bytes, format_char_sequence, endian="<"):
    """Read and unpack bytes from a binary file."""
    data = f.read(num_bytes)
    return struct.unpack(endian + format_char_sequence, data)


# ═══════════════════════════════════════════════════════════════════════════
#  Binary readers
# ═══════════════════════════════════════════════════════════════════════════

def read_cameras_binary(path: str) -> dict[int, ColmapCamera]:
    """Read ``cameras.bin`` → ``{camera_id: ColmapCamera}``."""
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = _read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_cameras):
            props = _read_next_bytes(f, 24, "iiQQ")
            camera_id, model_id = props[0], props[1]
            width, height = props[2], props[3]
            num_params = CAMERA_MODEL_NUM_PARAMS[model_id]
            params = np.array(_read_next_bytes(f, 8 * num_params, "d" * num_params))
            cameras[camera_id] = ColmapCamera(
                id=camera_id, model=CAMERA_MODEL_IDS[model_id],
                width=width, height=height, params=params,
            )
    return cameras


def read_images_binary(path: str) -> dict[int, ColmapImage]:
    """Read ``images.bin`` → ``{image_id: ColmapImage}``."""
    images = {}
    with open(path, "rb") as f:
        num_reg = _read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_reg):
            props = _read_next_bytes(f, 64, "idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5])
            tvec = np.array(props[5:8])
            camera_id = props[8]

            name = b""
            ch = _read_next_bytes(f, 1, "c")[0]
            while ch != b"\x00":
                name += ch
                ch = _read_next_bytes(f, 1, "c")[0]
            name = name.decode("utf-8")

            num_pts = _read_next_bytes(f, 8, "Q")[0]
            data = _read_next_bytes(f, 24 * num_pts, "ddq" * num_pts) if num_pts > 0 else ()
            xys = np.column_stack([data[0::3], data[1::3]]) if num_pts > 0 else np.zeros((0, 2))
            pt3d_ids = np.array(data[2::3]) if num_pts > 0 else np.zeros(0)

            images[image_id] = ColmapImage(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=name,
                xys=xys, point3D_ids=pt3d_ids,
            )
    return images


def read_points3D_binary(path: str) -> dict[int, ColmapPoint3D]:
    """Read ``points3D.bin`` → ``{point3D_id: ColmapPoint3D}``."""
    points3D = {}
    with open(path, "rb") as f:
        num_pts = _read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_pts):
            props = _read_next_bytes(f, 43, "QdddBBBd")
            point3D_id = props[0]
            xyz = np.array(props[1:4])
            rgb = np.array(props[4:7])
            error = props[7]
            track_len = _read_next_bytes(f, 8, "Q")[0]
            track = _read_next_bytes(f, 8 * track_len, "ii" * track_len) if track_len > 0 else ()
            image_ids = np.array(track[0::2]) if track_len > 0 else np.zeros(0)
            pt2d_idxs = np.array(track[1::2]) if track_len > 0 else np.zeros(0)
            points3D[point3D_id] = ColmapPoint3D(
                id=point3D_id, xyz=xyz, rgb=rgb, error=error,
                image_ids=image_ids, point2D_idxs=pt2d_idxs,
            )
    return points3D


# ═══════════════════════════════════════════════════════════════════════════
#  Geometry helpers
# ═══════════════════════════════════════════════════════════════════════════

def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert COLMAP quaternion ``(w, x, y, z)`` to a 3×3 rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y],
    ])


def get_intrinsics(cam: ColmapCamera) -> dict:
    """Extract ``fx, fy, cx, cy`` from a :class:`ColmapCamera`.

    Supports PINHOLE, SIMPLE_PINHOLE, SIMPLE_RADIAL, RADIAL, and OPENCV.
    """
    if cam.model == "PINHOLE":
        fx, fy, cx, cy = cam.params[:4]
    elif cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        f = cam.params[0]; cx, cy = cam.params[1], cam.params[2]
        fx = fy = f
    elif cam.model == "OPENCV":
        fx, fy, cx, cy = cam.params[:4]
    else:
        raise ValueError(f"Unsupported camera model: {cam.model}")
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
