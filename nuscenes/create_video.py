"""Generate videos from a nuScenes-style ``cams`` directory.

Expected input layout::

    <cams-dir>/
        CAM_FRONT/        kf00_*.jpg, kf01_*.jpg, ...
        CAM_FRONT_LEFT/   ...
        CAM_FRONT_RIGHT/  ...
        CAM_BACK/         ...
        CAM_BACK_LEFT/    ...
        CAM_BACK_RIGHT/   ...

Creates one MP4 per camera plus a 2x3 grid video with the camera order:

    front_left | front | front_right
    back_left  | back  | back_right
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

KF_RE = re.compile(r"^kf(?P<idx>\d+)_", re.IGNORECASE)
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def list_frames(cam_dir: Path) -> list[Path]:
    items: list[tuple[int | str, Path]] = []
    for p in cam_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        m = KF_RE.match(p.name)
        items.append((int(m.group("idx")) if m else p.name, p))
    return [p for _, p in sorted(items, key=lambda x: x[0])]


def write_video(frames: list[Path], out_path: Path, fps: int) -> None:
    if not frames:
        return
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Failed to read {frames[0]}")
    h, w = first.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    try:
        for f in frames:
            img = cv2.imread(str(f))
            if img is None:
                print(f"  skip unreadable: {f.name}")
                continue
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            writer.write(img)
    finally:
        writer.release()
    print(f"  wrote {out_path} ({len(frames)} frames, {w}x{h} @ {fps}fps)")


def write_grid_video(
    groups: dict[str, list[Path]],
    out_path: Path,
    fps: int,
    layout: list[list[str]],
    tile_width: int = 640,
) -> None:
    keys_present = {k for row in layout for k in row if k in groups}
    if not keys_present:
        return
    n_frames = min(len(groups[k]) for k in keys_present)
    if n_frames == 0:
        return

    sample = cv2.imread(str(groups[next(iter(keys_present))][0]))
    src_h, src_w = sample.shape[:2]
    tw = tile_width
    th = int(round(src_h * (tw / src_w)))

    rows = len(layout)
    cols = max(len(r) for r in layout)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (tw * cols, th * rows))
    try:
        for i in range(n_frames):
            row_imgs = []
            for row in layout:
                tiles = []
                for k in row:
                    if k in groups:
                        img = cv2.imread(str(groups[k][i]))
                        if img is None:
                            img = np.zeros((th, tw, 3), dtype=np.uint8)
                        else:
                            img = cv2.resize(img, (tw, th))
                        cv2.putText(
                            img, k, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2, cv2.LINE_AA,
                        )
                    else:
                        img = np.zeros((th, tw, 3), dtype=np.uint8)
                    tiles.append(img)
                while len(tiles) < cols:
                    tiles.append(np.zeros((th, tw, 3), dtype=np.uint8))
                row_imgs.append(np.hstack(tiles))
            writer.write(np.vstack(row_imgs))
    finally:
        writer.release()
    print(
        f"  wrote {out_path} ({n_frames} frames, {tw * cols}x{th * rows} @ {fps}fps)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cams-dir",
        type=Path,
        default=Path(__file__).parent / "outputs/round_trip_10s/1/cams",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--tile-width", type=int, default=640)
    parser.add_argument("--no-grid", action="store_true")
    args = parser.parse_args()

    cams_dir: Path = args.cams_dir.resolve()
    if not cams_dir.is_dir():
        raise SystemExit(f"cams dir not found: {cams_dir}")
    output_dir: Path = (args.output_dir or cams_dir.parent / "videos").resolve()

    cam_dirs = sorted(p for p in cams_dir.iterdir() if p.is_dir())
    groups: dict[str, list[Path]] = {}
    for d in cam_dirs:
        frames = list_frames(d)
        if frames:
            groups[d.name] = frames

    if not groups:
        raise SystemExit(f"no camera frames found under {cams_dir}")

    print(f"found {len(groups)} cameras in {cams_dir}:")
    for k, v in groups.items():
        print(f"  {k}: {len(v)} frames")

    for name, frames in groups.items():
        write_video(frames, output_dir / f"{name}.mp4", args.fps)

    if not args.no_grid and len(groups) > 1:
        layout = [
            ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
            ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"],
        ]
        write_grid_video(
            groups, output_dir / "grid.mp4", args.fps, layout, args.tile_width
        )


if __name__ == "__main__":
    main()
