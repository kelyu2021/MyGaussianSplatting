"""Extract panoramic frames from a video at a given FPS."""

import argparse
import os
import cv2


def extract_frames(video_path: str, output_dir: str, fps: float) -> None:
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = src_fps / fps

    frame_idx = 0
    saved = 0
    next_capture = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= next_capture:
            out_path = os.path.join(output_dir, f"frame_{saved + 1:04d}.png")
            cv2.imwrite(out_path, frame)
            saved += 1
            next_capture += interval
        frame_idx += 1

    cap.release()
    print(f"Saved {saved} frames to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--fps", type=float, default=5.0)
    args = parser.parse_args()

    extract_frames(args.video, args.output_dir, args.fps)
