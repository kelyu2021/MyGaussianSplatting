"""
Generate a panoramic video from per-frame mask visualisation images.

Layout follows the original 360° equirectangular order:
    left → front → right → back

Usage:
    python visualize_mask.py
"""

import os
import re
from pathlib import Path

import cv2
import numpy as np

INPUT_DIR = Path(__file__).resolve().parent / "output" / "masks_depth" / "vis"
OUTPUT_PATH = INPUT_DIR.parent / "vis" / "masks_depth_panoramic.mp4"
FPS = 10
FACE_ORDER = ["left", "front", "right", "back"]  # equirectangular L→R


def collect_frames(input_dir: Path) -> dict[int, dict[str, Path]]:
    """Return {frame_idx: {face_name: path}} sorted by frame index."""
    pattern = re.compile(r"^frame_(\d+)_(front|back|left|right)\.png$")
    frames: dict[int, dict[str, Path]] = {}
    for p in sorted(input_dir.iterdir()):
        m = pattern.match(p.name)
        if m:
            idx = int(m.group(1))
            face = m.group(2)
            frames.setdefault(idx, {})[face] = p
    return dict(sorted(frames.items()))


def main():
    frames = collect_frames(INPUT_DIR)
    if not frames:
        print(f"No frames found in {INPUT_DIR}")
        return

    # Read one image to determine size
    sample = cv2.imread(str(next(iter(next(iter(frames.values())).values()))))
    h, w = sample.shape[:2]
    strip_w = w * len(FACE_ORDER)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, FPS, (strip_w, h))

    for idx in sorted(frames):
        faces = frames[idx]
        if not all(f in faces for f in FACE_ORDER):
            print(f"  Skipping frame {idx} (missing faces)")
            continue
        panels = [cv2.imread(str(faces[f])) for f in FACE_ORDER]
        strip = np.concatenate(panels, axis=1)
        writer.write(strip)

    writer.release()
    print(f"Saved {len(frames)} frames → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
