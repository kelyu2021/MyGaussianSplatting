"""
KITTI-360 fisheye → panorama (equirectangular)
================================================
Builds an equirectangular panorama from the front stereo fisheye cameras
(image_02, image_03) using the official MEI calibration files.

Main pieces:
 - load_mei_calibration: parse KITTI-360 MEI yaml (xi + distortion)
 - load_cam_to_pose: read cam→pose matrices and return pose→cam transforms
 - build_equirect_remap: build cv2.remap tables for a given camera
 - stitch_fisheyes: render and blend the two fisheye views into a panorama

Example
-------
python fisheye2pano.py \
  --left-img ../data/kitti360/data_2d_raw/2013_05_28_drive_0000_sync/image_02/0000000000.png \
  --right-img ../data/kitti360/data_2d_raw/2013_05_28_drive_0000_sync/image_03/0000000000.png

Outputs to test/scripts/output/panorama.png by default.
"""

import argparse
import os
import re
from typing import Dict, Tuple

import cv2
import numpy as np

# Constants that control validity checks and defaults
FISHEYE_FOV_DEG = 190.0          # used when xi <= 1 (fallback FOV)
CIRCLE_MARGIN = 8                # shrink detected circle a little to avoid edges
DEFAULT_PANO_W = 2048
DEFAULT_PANO_H = 1024


# ─── Calibration loading ─────────────────────────────────────────────────────

def load_mei_calibration(yaml_path: str) -> Dict[str, float]:
	"""Parse a KITTI-360 MEI yaml file into a flat dict of floats."""
	with open(yaml_path, "r", encoding="utf-8") as f:
		text = f.read()

	def grab(key: str) -> float:
		m = re.search(rf"{key}:\s*([-+eE0-9\.]+)", text)
		if not m:
			raise ValueError(f"Key {key} missing in {yaml_path}")
		return float(m.group(1))

	calib = {
		"xi": grab("xi"),
		"k1": grab("k1"),
		"k2": grab("k2"),
		"p1": grab("p1"),
		"p2": grab("p2"),
		"gamma1": grab("gamma1"),
		"gamma2": grab("gamma2"),
		"u0": grab("u0"),
		"v0": grab("v0"),
		"width": int(grab("image_width")),
		"height": int(grab("image_height")),
	}
	return calib


def load_cam_to_pose(txt_path: str) -> Dict[str, np.ndarray]:
	"""Load cam→pose 3x4 matrices, return pose→cam 4x4 transforms."""
	result = {}
	with open(txt_path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line or line.startswith("#"):
				continue
			if ":" not in line:
				continue
			name, values = line.split(":", 1)
			vals = [float(v) for v in values.split()]
			if len(vals) != 12:
				continue

			T_cam_pose = np.eye(4, dtype=np.float64)
			T_cam_pose[:3, :4] = np.array(vals, dtype=np.float64).reshape(3, 4)

			# Invert: pose→cam = (cam→pose)^-1
			R = T_cam_pose[:3, :3]
			t = T_cam_pose[:3, 3:4]
			R_inv = R.T
			t_inv = -R_inv @ t
			T_pose_cam = np.eye(4, dtype=np.float64)
			T_pose_cam[:3, :3] = R_inv
			T_pose_cam[:3, 3] = t_inv[:, 0]

			result[name.strip()] = T_pose_cam
	return result


# ─── Geometry helpers ───────────────────────────────────────────────────────

def detect_fisheye_circle(img: np.ndarray, margin: float = CIRCLE_MARGIN,
						  scale_up: float = 1.0) -> Tuple[float, float, float]:
	"""Detect the valid fisheye circle (center x,y, radius).

	margin   : shrink (>0) or expand (<0) the usable radius
	scale_up : multiply detected radius (e.g., 1.02 to loosen mask)
	"""
	gray = img
	if img.ndim == 3:
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Downsample to make detection cheap
	scale = 0.25
	small = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
	_, mask = cv2.threshold(small, 8, 255, cv2.THRESH_BINARY)
	coords = cv2.findNonZero(mask)

	h, w = img.shape[:2]
	if coords is None or len(coords) < 10:
		# Fallback: assume centered circle
		cx, cy = w / 2.0, h / 2.0
		r = (min(cx, cy) - margin) * scale_up
		return cx, cy, r

	# Find enclosing circle in the low-res space and rescale
	(cx_s, cy_s), r_s = cv2.minEnclosingCircle(coords)
	cx = cx_s / scale
	cy = cy_s / scale
	r = (r_s / scale - margin) * scale_up
	return float(cx), float(cy), float(r)


def mei_project(x: np.ndarray, y: np.ndarray, z: np.ndarray, calib: Dict[str, float]):
	"""MEI forward projection: camera ray → pixel (u, v)."""
	xi = calib["xi"]
	k1, k2, p1, p2 = calib["k1"], calib["k2"], calib["p1"], calib["p2"]
	gamma1, gamma2, u0, v0 = calib["gamma1"], calib["gamma2"], calib["u0"], calib["v0"]

	norm = np.sqrt(x * x + y * y + z * z)
	denom = z + xi * norm
	valid = denom > 1e-8

	mx = np.where(valid, x / denom, 0.0)
	my = np.where(valid, y / denom, 0.0)

	r2 = mx * mx + my * my
	radial = 1.0 + k1 * r2 + k2 * r2 * r2
	x_d = mx * radial + 2.0 * p1 * mx * my + p2 * (r2 + 2.0 * mx * mx)
	y_d = my * radial + p1 * (r2 + 2.0 * my * my) + 2.0 * p2 * mx * my

	u = gamma1 * x_d + u0
	v = gamma2 * y_d + v0
	return u, v, valid


# ─── Remap construction ─────────────────────────────────────────────────────

def build_equirect_remap(calib: Dict[str, float], T_pose_cam: np.ndarray,
						 src_img: np.ndarray,
						 out_w: int = DEFAULT_PANO_W,
						 out_h: int = DEFAULT_PANO_H,
						 lat_min_deg: float = -90.0,
						 lat_max_deg: float = 90.0,
						 circle_margin: float = CIRCLE_MARGIN,
						 circle_scale: float = 1.0,
						 enable_circle_mask: bool = True):
	"""Build cv2.remap tables (map_x, map_y, weight) for one camera.

	lat_min_deg / lat_max_deg let you crop vertical FOV to avoid black bands
	when the fisheye cannot see the poles; e.g., use -80..80.
	"""
	lat_min = np.deg2rad(lat_min_deg)
	lat_max = np.deg2rad(lat_max_deg)
	# Equirect grid (lon: -pi..pi, lat: lat_min..lat_max)
	lon = (np.arange(out_w, dtype=np.float64) + 0.5) / out_w * (2.0 * np.pi) - np.pi
	lat = (np.arange(out_h, dtype=np.float64) + 0.5) / out_h * (lat_max - lat_min) + lat_min
	lon_grid, lat_grid = np.meshgrid(lon, lat)

	# Pose-frame unit rays
	x_p = np.cos(lat_grid) * np.cos(lon_grid)
	y_p = np.cos(lat_grid) * np.sin(lon_grid)
	z_p = np.sin(lat_grid)

	R = T_pose_cam[:3, :3]
	# Rotate into camera frame
	x_c = R[0, 0] * x_p + R[0, 1] * y_p + R[0, 2] * z_p
	y_c = R[1, 0] * x_p + R[1, 1] * y_p + R[1, 2] * z_p
	z_c = R[2, 0] * x_p + R[2, 1] * y_p + R[2, 2] * z_p

	# Project
	map_x, map_y, valid = mei_project(x_c, y_c, z_c, calib)

	# Circle validity
	if enable_circle_mask:
		cx, cy, r = detect_fisheye_circle(src_img, margin=circle_margin, scale_up=circle_scale)
		dist = np.sqrt((map_x - cx) ** 2 + (map_y - cy) ** 2)
		valid &= dist <= r

	# Incidence angle clipping to avoid grazing rays
	if calib["xi"] > 1.0:
		theta_max = np.arccos(-1.0 / calib["xi"])
	else:
		theta_max = np.deg2rad(FISHEYE_FOV_DEG / 2.0)
	theta = np.arccos(np.clip(z_c, -1.0, 1.0))
	valid &= theta < theta_max

	# Image bounds
	h_src, w_src = src_img.shape[:2]
	valid &= (map_x >= 0.0) & (map_x < w_src - 1) & (map_y >= 0.0) & (map_y < h_src - 1)

	# Blend weight: favour rays near optical axis
	weight = np.where(valid, np.clip(z_c, 0.0, 1.0) ** 2, 0.0).astype(np.float32)

	map_x = np.where(valid, map_x, 0.0).astype(np.float32)
	map_y = np.where(valid, map_y, 0.0).astype(np.float32)
	return map_x, map_y, weight


def stitch_fisheyes(left_img: np.ndarray, left_calib: Dict[str, float], T_left: np.ndarray,
					right_img: np.ndarray, right_calib: Dict[str, float], T_right: np.ndarray,
					out_w: int, out_h: int,
					lat_min_deg: float = -90.0,
					lat_max_deg: float = 90.0,
					circle_margin: float = CIRCLE_MARGIN,
					circle_scale: float = 1.0,
					enable_circle_mask: bool = True):
	"""Render and blend both fisheye images into one panorama."""
	mx_l, my_l, w_l = build_equirect_remap(left_calib, T_left, left_img, out_w, out_h,
										  lat_min_deg, lat_max_deg,
										  circle_margin=circle_margin,
										  circle_scale=circle_scale,
										  enable_circle_mask=enable_circle_mask)
	mx_r, my_r, w_r = build_equirect_remap(right_calib, T_right, right_img, out_w, out_h,
										  lat_min_deg, lat_max_deg,
										  circle_margin=circle_margin,
										  circle_scale=circle_scale,
										  enable_circle_mask=enable_circle_mask)

	pano_l = cv2.remap(left_img, mx_l, my_l, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
	pano_r = cv2.remap(right_img, mx_r, my_r, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

	w_l3 = w_l[..., None]
	w_r3 = w_r[..., None]
	denom = w_l3 + w_r3 + 1e-6

	pano = (pano_l.astype(np.float32) * w_l3 + pano_r.astype(np.float32) * w_r3) / denom
	pano = np.where((w_l3 + w_r3) > 0, pano, 0.0)
	return pano.astype(np.uint8), {"left": w_l, "right": w_r}


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
	parser = argparse.ArgumentParser(description="Fuse KITTI-360 stereo fisheyes into a panorama")
	parser.add_argument("--left-img", default=None,
						help="Path to image_02 fisheye (default: dataset sample)")
	parser.add_argument("--right-img", default=None,
						help="Path to image_03 fisheye (default: dataset sample)")
	parser.add_argument("--calib-dir", default=None,
						help="Calibration directory (default: ../data/kitti360/calibration)")
	parser.add_argument("--width", type=int, default=DEFAULT_PANO_W, help="Panorama width")
	parser.add_argument("--height", type=int, default=DEFAULT_PANO_H, help="Panorama height")
	parser.add_argument("--lat-min", type=float, default=-90.0,
						help="Minimum latitude in degrees (default -90)")
	parser.add_argument("--lat-max", type=float, default=90.0,
						help="Maximum latitude in degrees (default 90)")
	parser.add_argument("--circle-margin", type=float, default=CIRCLE_MARGIN,
						help="Shrink (>0) or expand (<0) detected circle radius (pixels)")
	parser.add_argument("--circle-scale", type=float, default=1.0,
						help="Scale factor on detected circle radius (e.g., 1.02 to loosen)")
	parser.add_argument("--no-circle-mask", action="store_true",
						help="Disable circle mask (use only incidence-angle + bounds)")
	parser.add_argument("--output", default=None, help="Output path for panorama")
	parser.add_argument("--debug-weights", action="store_true", help="Save weight maps for inspection")
	args = parser.parse_args()

	script_dir = os.path.dirname(os.path.abspath(__file__))
	calib_dir = args.calib_dir or os.path.join(script_dir, "..", "data", "kitti360", "calibration")
	data_dir = os.path.join(script_dir, "..", "data", "kitti360", "data_2d_raw",
							"2013_05_28_drive_0000_sync")

	left_path = args.left_img or os.path.join(data_dir, "image_02", "0000000000.png")
	right_path = args.right_img or os.path.join(data_dir, "image_03", "0000000000.png")

	os.makedirs(os.path.join(script_dir, "output"), exist_ok=True)
	out_path = args.output or os.path.join(script_dir, "output", "panorama.png")

	print("Loading calibration ...")
	calib_left = load_mei_calibration(os.path.join(calib_dir, "image_02.yaml"))
	calib_right = load_mei_calibration(os.path.join(calib_dir, "image_03.yaml"))
	cam2pose = load_cam_to_pose(os.path.join(calib_dir, "calib_cam_to_pose.txt"))
	T_left = cam2pose["image_02"]
	T_right = cam2pose["image_03"]

	print(f"  image_02 xi={calib_left['xi']:.3f} gamma=({calib_left['gamma1']:.1f}, {calib_left['gamma2']:.1f})")
	print(f"  image_03 xi={calib_right['xi']:.3f} gamma=({calib_right['gamma1']:.1f}, {calib_right['gamma2']:.1f})")

	print(f"Loading images ...")
	left_img = cv2.imread(left_path, cv2.IMREAD_COLOR)
	right_img = cv2.imread(right_path, cv2.IMREAD_COLOR)
	if left_img is None or right_img is None:
		raise FileNotFoundError("Could not load input images")

	print(f"Building panorama {args.width}x{args.height} with lat [{args.lat_min}, {args.lat_max}] ...")
	pano, weights = stitch_fisheyes(left_img, calib_left, T_left,
								   right_img, calib_right, T_right,
								   args.width, args.height,
								   lat_min_deg=args.lat_min,
								   lat_max_deg=args.lat_max,
								   circle_margin=args.circle_margin,
								   circle_scale=args.circle_scale,
								   enable_circle_mask=not args.no_circle_mask)

	cv2.imwrite(out_path, pano)
	print(f"Saved panorama: {out_path}")

	if args.debug_weights:
		cv2.imwrite(os.path.join(script_dir, "output", "weight_left.png"),
					(np.clip(weights["left"], 0, 1) * 255).astype(np.uint8))
		cv2.imwrite(os.path.join(script_dir, "output", "weight_right.png"),
					(np.clip(weights["right"], 0, 1) * 255).astype(np.uint8))
		print("Saved weight maps to output/weight_*.png")


if __name__ == "__main__":
	main()
