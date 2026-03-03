"""
Fisheye → Perspective  (KITTI-360, calibrated MEI model)
========================================================
Converts a KITTI-360 fisheye image into one or more rectilinear (pinhole)
perspective images using the official MEI camera calibration.

For each output perspective view we:
1. Build a ray direction for every output pixel using a virtual pinhole camera
2. Project each ray through the MEI model to find the source fisheye pixel
3. Remap with bilinear interpolation

Usage
-----
    python fisheye2pers.py                        # default forward view
    python fisheye2pers.py --yaw 0 --pitch 0      # custom direction
    python fisheye2pers.py --cube                  # 6-face cube map
"""

import argparse
import cv2
import numpy as np
import os
import sys

# Reuse calibration loaders from the panorama script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fisheye2pano import (
    load_mei_calibration,
    load_cam_to_pose,
    mei_project,
    detect_fisheye_circle,
    FISHEYE_FOV_DEG,
    CIRCLE_MARGIN,
)


# ─── Default output parameters ────────────────────────────────────────────────
DEFAULT_PERS_WIDTH  = 800
DEFAULT_PERS_HEIGHT = 600
DEFAULT_HFOV_DEG    = 90.0   # horizontal field-of-view of virtual pinhole camera


def build_perspective_remap(calib, T_pose_cam, src_img,
                            out_w, out_h, hfov_deg,
                            yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    """
    Build ``cv2.remap`` tables that map a virtual perspective camera back to
    the fisheye source image.

    Parameters
    ----------
    calib       : dict – MEI intrinsics (from ``load_mei_calibration``)
    T_pose_cam  : 4×4 ndarray – cam→pose extrinsic
    src_img     : ndarray – source fisheye image (used for circle detection)
    out_w, out_h : int – output perspective image size
    hfov_deg    : float – horizontal FOV of the virtual camera (degrees)
    yaw_deg     : float – yaw of virtual camera in the **camera** frame
                  (0 = optical axis, positive = right)
    pitch_deg   : float – pitch (positive = up)
    roll_deg    : float – roll (positive = CW when looking along optical axis)

    Returns
    -------
    map_x, map_y : float32  ``(out_h, out_w)``
    valid         : bool     ``(out_h, out_w)``
    """
    # ── Virtual pinhole intrinsics ──
    hfov = np.deg2rad(hfov_deg)
    f = (out_w / 2.0) / np.tan(hfov / 2.0)   # focal length in pixels
    cx, cy = out_w / 2.0, out_h / 2.0

    # ── Pixel grid → normalised image plane ──
    u = np.arange(out_w, dtype=np.float64) + 0.5
    v = np.arange(out_h, dtype=np.float64) + 0.5
    uu, vv = np.meshgrid(u, v)

    # Rays in the virtual camera frame (z-forward, x-right, y-down)
    rx = (uu - cx) / f
    ry = (vv - cy) / f
    rz = np.ones_like(rx)

    # Normalise
    norm = np.sqrt(rx**2 + ry**2 + rz**2)
    rx /= norm;  ry /= norm;  rz /= norm

    # ── Rotation: virtual camera → physical camera frame ──
    # The virtual camera's optical axis starts along the physical camera's Z.
    # We rotate it by (yaw, pitch, roll) to point elsewhere.
    yaw   = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll  = np.deg2rad(roll_deg)

    # Ry (yaw around Y-down axis)
    Ry = np.array([[ np.cos(yaw), 0, np.sin(yaw)],
                   [ 0,           1, 0           ],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    # Rx (pitch around X-right axis, positive = camera looks up → nose down)
    Rx = np.array([[1, 0,             0            ],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch),  np.cos(pitch)]])
    # Rz (roll around Z-forward axis)
    Rz = np.array([[ np.cos(roll), -np.sin(roll), 0],
                   [ np.sin(roll),  np.cos(roll), 0],
                   [ 0,            0,             1]])

    R_virt = Ry @ Rx @ Rz   # virtual → physical camera

    # Apply rotation
    dx = R_virt[0, 0] * rx + R_virt[0, 1] * ry + R_virt[0, 2] * rz
    dy = R_virt[1, 0] * rx + R_virt[1, 1] * ry + R_virt[1, 2] * rz
    dz = R_virt[2, 0] * rx + R_virt[2, 1] * ry + R_virt[2, 2] * rz

    # ── Incidence angle clipping (same as panorama pipeline) ──
    xi = calib["xi"]
    if xi > 1.0:
        theta_max = np.arccos(-1.0 / xi)
    else:
        theta_max = np.deg2rad(FISHEYE_FOV_DEG / 2.0)

    theta = np.arccos(np.clip(dz, -1.0, 1.0))

    # ── MEI forward projection ──
    map_x, map_y, valid_mei = mei_project(dx, dy, dz, calib)

    # ── Circle validity ──
    circ_cx, circ_cy, circ_r = detect_fisheye_circle(src_img)
    dist = np.sqrt((map_x - circ_cx)**2 + (map_y - circ_cy)**2)
    inside_circle = dist < circ_r

    valid = valid_mei & inside_circle & (theta < theta_max)

    map_x = np.where(valid, map_x, 0).astype(np.float32)
    map_y = np.where(valid, map_y, 0).astype(np.float32)

    return map_x, map_y, valid


def fisheye_to_perspective(src_img, calib, T_pose_cam,
                           out_w=DEFAULT_PERS_WIDTH,
                           out_h=DEFAULT_PERS_HEIGHT,
                           hfov_deg=DEFAULT_HFOV_DEG,
                           yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    """
    Convert a single fisheye image to a perspective view.

    Returns
    -------
    pers_img : ndarray (out_h, out_w, 3), uint8
    """
    mx, my, valid = build_perspective_remap(
        calib, T_pose_cam, src_img,
        out_w, out_h, hfov_deg,
        yaw_deg, pitch_deg, roll_deg,
    )
    pers = cv2.remap(src_img, mx, my, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    # Zero out invalid pixels
    pers[~valid] = 0
    return pers


def fisheye_to_cubemap(src_img, calib, T_pose_cam, face_size=512):
    """
    Generate a 6-face cube map from the fisheye image.

    Returns a dict  ``{face_name: ndarray}``  and a cross layout image.
    Only faces whose rays fall within the fisheye FOV will have content;
    the rest will be black.
    """
    # (name, yaw, pitch)
    faces = [
        ("front",    0,    0),
        ("right",   90,    0),
        ("back",   180,    0),
        ("left",  -90,    0),
        ("top",      0,  -90),
        ("bottom",   0,   90),
    ]
    results = {}
    for name, yaw, pitch in faces:
        results[name] = fisheye_to_perspective(
            src_img, calib, T_pose_cam,
            face_size, face_size, 90.0,
            yaw_deg=yaw, pitch_deg=pitch,
        )

    # ── Assemble cross layout ──
    #        [top]
    # [left][front][right][back]
    #        [bottom]
    s = face_size
    cross = np.zeros((3 * s, 4 * s, 3), dtype=np.uint8)
    cross[0:s,     s:2*s]   = results["top"]
    cross[s:2*s,   0:s]     = results["left"]
    cross[s:2*s,   s:2*s]   = results["front"]
    cross[s:2*s,   2*s:3*s] = results["right"]
    cross[s:2*s,   3*s:4*s] = results["back"]
    cross[2*s:3*s, s:2*s]   = results["bottom"]

    return results, cross


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Convert KITTI-360 fisheye to perspective image(s).")
    parser.add_argument("--image", default=None,
                        help="Path to fisheye image (default: image_02/0000000000.png)")
    parser.add_argument("--cam", default="image_02", choices=["image_02", "image_03"],
                        help="Which camera calibration to use (default: image_02)")
    parser.add_argument("--width", type=int, default=DEFAULT_PERS_WIDTH,
                        help=f"Output width (default: {DEFAULT_PERS_WIDTH})")
    parser.add_argument("--height", type=int, default=DEFAULT_PERS_HEIGHT,
                        help=f"Output height (default: {DEFAULT_PERS_HEIGHT})")
    parser.add_argument("--hfov", type=float, default=DEFAULT_HFOV_DEG,
                        help=f"Horizontal FOV in degrees (default: {DEFAULT_HFOV_DEG})")
    parser.add_argument("--yaw", type=float, default=0.0,
                        help="Yaw in degrees (0=forward, 90=right)")
    parser.add_argument("--pitch", type=float, default=0.0,
                        help="Pitch in degrees (positive=up)")
    parser.add_argument("--roll", type=float, default=0.0,
                        help="Roll in degrees")
    parser.add_argument("--cube", action="store_true",
                        help="Generate a 6-face cube map instead of a single view")
    parser.add_argument("--cube-size", type=int, default=512,
                        help="Face size for cube map (default: 512)")
    parser.add_argument("--output", default=None,
                        help="Output path (default: output/perspective.png)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    calib_dir  = os.path.join(script_dir, "..", "data", "kitti360", "calibration")
    base_dir   = os.path.join(script_dir, "..", "data", "kitti360", "data_2d_raw",
                              "2013_05_28_drive_0000_sync")

    # ── Load calibration ──
    print("Loading calibration ...")
    calib = load_mei_calibration(os.path.join(calib_dir, f"{args.cam}.yaml"))
    cam2pose = load_cam_to_pose(os.path.join(calib_dir, "calib_cam_to_pose.txt"))
    T_pose_cam = cam2pose[args.cam]

    print(f"  {args.cam}: xi={calib['xi']:.3f}  "
          f"gamma=({calib['gamma1']:.1f}, {calib['gamma2']:.1f})  "
          f"pp=({calib['u0']:.1f}, {calib['v0']:.1f})")

    # ── Load image ──
    if args.image is not None:
        img_path = args.image
    else:
        img_path = os.path.join(base_dir, args.cam, "0000000000.png")

    if not os.path.isfile(img_path):
        print(f"Error: {img_path} not found")
        sys.exit(1)

    print(f"Loading {img_path} ...")
    src_img = cv2.imread(img_path)
    print(f"  Size: {src_img.shape[1]}x{src_img.shape[0]}")

    out_dir = os.path.join(script_dir, "output")
    os.makedirs(out_dir, exist_ok=True)

    if args.cube:
        # ── Cube map ──
        print(f"Generating cube map (face size {args.cube_size}) ...")
        faces, cross = fisheye_to_cubemap(
            src_img, calib, T_pose_cam, args.cube_size)

        out_path = args.output or os.path.join(out_dir, "cubemap_cross.png")
        cv2.imwrite(out_path, cross)
        print(f"Saved cross layout: {out_path}")

        # Also save individual faces
        for name, face_img in faces.items():
            fp = os.path.join(out_dir, f"cubemap_{name}.png")
            cv2.imwrite(fp, face_img)
        print(f"Saved individual faces to {out_dir}/cubemap_*.png")
    else:
        # ── Single perspective view ──
        print(f"Generating perspective view "
              f"(yaw={args.yaw}°, pitch={args.pitch}°, "
              f"hfov={args.hfov}°, {args.width}x{args.height}) ...")
        pers = fisheye_to_perspective(
            src_img, calib, T_pose_cam,
            out_w=args.width, out_h=args.height,
            hfov_deg=args.hfov,
            yaw_deg=args.yaw, pitch_deg=args.pitch, roll_deg=args.roll,
        )
        out_path = args.output or os.path.join(out_dir, "perspective.png")
        cv2.imwrite(out_path, pers)
        print(f"Saved: {out_path}  ({pers.shape[1]}x{pers.shape[0]})")


if __name__ == "__main__":
    main()
