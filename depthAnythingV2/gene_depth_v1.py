"""
Generate depth maps from panoramic images using Depth Anything V2.

Usage:
    python gene_depth.py --input_dir data/panoramic \
                         --output_dir data/panoramic_depth \
                         --video_out data/panoramic_depth/depth_video.mp4 \
                         --fps 5
"""

import argparse
import os
import glob

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image


def generate_depth_maps(input_dir: str, output_dir: str,
                        video_out: str, fps: float,
                        threshold: float = 1.0) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "depth-anything/Depth-Anything-V2-Large-hf"
    print(f"Loading {model_name} on {device} ...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device)
    model.eval()

    # ── Gather input images (sorted) ─────────────────────────────────
    exts = ("*.png", "*.jpg", "*.jpeg")
    paths = sorted(
        p for ext in exts for p in glob.glob(os.path.join(input_dir, ext))
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {input_dir}")
    print(f"Found {len(paths)} images")

    # ── Inference ─────────────────────────────────────────────────────
    depth_frames = []
    for i, img_path in enumerate(paths):
        pil_img = Image.open(img_path).convert("RGB")
        inputs = processor(images=pil_img, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        h, w = pil_img.size[1], pil_img.size[0]
        depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

        # Flip vertically to match original image orientation
        depth = np.flipud(depth)

        # Normalise to 0-255 for visualisation
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        # Threshold out close objects (e.g. car) – high depth_norm = close
        if threshold < 1.0:
            depth_norm[depth_norm > threshold] = 0.0

        depth_u8 = (depth_norm * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)

        # Save depth image
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(output_dir, f"{base}_depth.png")
        cv2.imwrite(out_path, depth_color)
        depth_frames.append(depth_color)

        print(f"  [{i+1}/{len(paths)}] {os.path.basename(img_path)} -> {os.path.basename(out_path)}")

    # ── Combine into video ────────────────────────────────────────────
    h_v, w_v = depth_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_out, fourcc, fps, (w_v, h_v))
    for frame in depth_frames:
        writer.write(frame)
    writer.release()
    print(f"Depth video saved to {video_out}  ({len(depth_frames)} frames, {fps} fps)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--video_out", type=str, default=None)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="Normalised depth threshold (0-1). Pixels above "
                             "this value (close objects like car) are zeroed out. "
                             "Try 0.7 to remove the car.")
    args = parser.parse_args()

    if args.video_out is None:
        args.video_out = os.path.join(args.output_dir, "depth_video.mp4")

    generate_depth_maps(args.input_dir, args.output_dir,
                        args.video_out, args.fps, args.threshold)
