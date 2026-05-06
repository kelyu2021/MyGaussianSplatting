("""
Render Wobble – Jittered Viewpoint Video Generator
=================================================

Generates four videos by interpolating between on-path and jittered viewpoints (left/right) for a given camera.

Inputs:
	--img_name:      Camera image name (from cameras.json)
	--road_width:    Road width (float, meters)
	--splats_path:   Path to trained model checkpoint directory
	--cameras_json:  Path to cameras.json
	--fps:           Frames per second for output videos
	--steps:         Number of interpolation steps (default: 10)
	--output_dir:    Output directory for videos

Outputs:
	Four videos: on2right.mp4, right2on.mp4, on2left.mp4, left2on.mp4
""")

import argparse
import os
import json
import numpy as np
import torch
import imageio
from tqdm import tqdm

from train import GaussianModel, Camera, load_camera, render


from collections import namedtuple
from train import CameraInfo

def load_camerainfo_from_json(cameras_json, img_name):
	with open(cameras_json, 'r') as f:
		cameras = json.load(f)
	for cam in cameras:
		if cam['img_name'] != img_name:
			continue

		# Build CameraInfo from JSON.
		# In this codebase (train.py / Gaussian Splatting convention):
		#   R: camera-to-world rotation (R_c2w)
		#   T: world-to-camera translation (tvec), such that X_cam = R_w2c * X_world + T
		# where R_w2c = R_c2w.T.
		width = int(cam.get('width', 2048))
		height = int(cam.get('height', 1536))
		fx = float(cam.get('fx', 500.0))
		fy = float(cam.get('fy', 500.0))
		K = np.array(
			[[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]],
			dtype=np.float32,
		)
		R = np.array(cam['rotation'], dtype=np.float32)  # expected R_c2w

		if 'tvec' in cam:
			T = np.array(cam['tvec'], dtype=np.float32)
		elif 'position' in cam:
			# position is camera center C in world coordinates.
			# tvec = -R_w2c * C = -(R_c2w.T) * C
			C = np.array(cam['position'], dtype=np.float32)
			T = -R.T @ C
		else:
			T = np.zeros(3, dtype=np.float32)

		image = torch.zeros(3, height, width)
		return CameraInfo(
			uid=cam.get('id', 0),
			R=R,
			T=T,
			FovY=2 * np.arctan(height / (2 * fy)),
			FovX=2 * np.arctan(width / (2 * fx)),
			K=K,
			image=image,
			image_path='',
			image_name=cam['img_name'],
			width=width,
			height=height,
			metadata=cam.get('metadata', {}),
			guidance={},
		)
	raise ValueError(f"Camera with img_name '{img_name}' not found in {cameras_json}")

def compute_jittered_positions_caminfo(caminfo, road_width):
	# caminfo: CameraInfo
	# caminfo.R is camera-to-world (R_c2w). Camera center C = -R_c2w * tvec
	pos = -caminfo.R @ caminfo.T  # camera center in world coordinates
	R = caminfo.R
	right = R[:,0]
	pos_right = pos + right * road_width
	pos_left = pos - right * road_width
	return pos, pos_right, pos_left, R

def interpolate_pose(pos_a, R_a, pos_b, R_b, t):
	pos = (1-t) * pos_a + t * pos_b
	rot = (1-t) * R_a + t * R_b
	u, _, vh = np.linalg.svd(rot)
	rot_ortho = u @ vh
	return pos, rot_ortho

def make_caminfo_jittered(base_caminfo, pos, rot):
	# Returns a new CameraInfo with updated position/rotation
	# rot is camera-to-world (R_c2w). tvec = -(R_c2w.T) * C
	T = -rot.T @ pos
	return base_caminfo._replace(R=rot, T=T)

def render_video(gaussians, bg_color, caminfo_a, caminfo_b, steps, fps):
	frames = []
	pos_a = -caminfo_a.R @ caminfo_a.T
	pos_b = -caminfo_b.R @ caminfo_b.T
	# Keep rotation fixed (matches render.py for the on-path view).
	R_fixed = caminfo_a.R
	for i in tqdm(range(steps), desc="Rendering segment"):
		t = i / (steps-1)
		pos = (1 - t) * pos_a + t * pos_b
		caminfo_interp = make_caminfo_jittered(caminfo_a, pos, R_fixed)
		cam = load_camera(caminfo_interp)
		with torch.no_grad():
			result = render(cam, gaussians, bg_color)
			rgb = result['rgb'].detach().cpu().clamp(0,1).numpy().transpose(1,2,0)
			frames.append((rgb*255).astype(np.uint8))
	return frames

def main():
	parser = argparse.ArgumentParser(description="Render jittered viewpoint videos")
	parser.add_argument('--img_name', required=True)
	parser.add_argument('--road_width', type=float, required=True)
	parser.add_argument('--model_path', required=True, help='Path to trained model checkpoint (.pth)')
	parser.add_argument('--cameras_json', required=True)
	parser.add_argument('--fps', type=int, default=10)
	parser.add_argument('--steps', type=int, default=10)
	parser.add_argument('--output_dir', default='wobble_videos')
	args = parser.parse_args()

	os.makedirs(args.output_dir, exist_ok=True)


	caminfo_on = load_camerainfo_from_json(args.cameras_json, args.img_name)
	pos_on, pos_right, pos_left, R = compute_jittered_positions_caminfo(caminfo_on, args.road_width)
	caminfo_right = make_caminfo_jittered(caminfo_on, pos_right, R)
	caminfo_left = make_caminfo_jittered(caminfo_on, pos_left, R)

	# Load model directly from the provided .pth file
	gaussians = torch.load(args.model_path, map_location='cuda')
	if isinstance(gaussians, dict) and 'active_sh_degree' in gaussians:
		model = GaussianModel(sh_degree=gaussians.get('active_sh_degree', 3))
		model.load_state_dict(gaussians)
		gaussians = model
	bg_color = torch.tensor([0,0,0], dtype=torch.float32, device='cuda')

	# Render all segments and concatenate
	frames = []
	frames += render_video(gaussians, bg_color, caminfo_on, caminfo_right, args.steps, args.fps)
	frames += render_video(gaussians, bg_color, caminfo_right, caminfo_on, args.steps, args.fps)
	frames += render_video(gaussians, bg_color, caminfo_on, caminfo_left, args.steps, args.fps)
	frames += render_video(gaussians, bg_color, caminfo_left, caminfo_on, args.steps, args.fps)

	# Use image name as the video filename
	base_name = os.path.splitext(os.path.basename(args.img_name))[0]
	out_path = os.path.join(args.output_dir, f'{base_name}.mp4')
	imageio.mimwrite(out_path, frames, fps=args.fps)
	print(f"Combined video saved to {out_path}")

if __name__ == '__main__':
	main()
