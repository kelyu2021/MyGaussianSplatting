"""Generate videos from infinigen/street_view_output/train/images.

Creates one MP4 per camera direction (front/back/left/right) plus a 2x2 grid
combining all four directions. Output goes to ``street_view_output/train/videos``.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

FRAME_RE = re.compile(r"^(?P<name>.+?)_(?P<idx>\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)


def collect_frames(images_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for p in images_dir.iterdir():
        if not p.is_file():
            continue
        m = FRAME_RE.match(p.name)
        if not m:
            continue
        groups[m.group("name")].append((int(m.group("idx")), p))
    return {k: [p for _, p in sorted(v)] for k, v in sorted(groups.items())}


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
    groups: dict[str, list[Path]], out_path: Path, fps: int
) -> None:
    # Layout: single row in order left | front | right | back (when present).
    preferred = ["left", "front", "right", "back"]
    keys = [k for k in preferred if k in groups] + [
        k for k in groups if k not in preferred
    ]
    if not keys:
        return
    n_frames = min(len(groups[k]) for k in keys)
    if n_frames == 0:
        return

    sample = cv2.imread(str(groups[keys[0]][0]))
    h, w = sample.shape[:2]
    cols = len(keys)
    rows = 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w * cols, h * rows))
    try:
        for i in range(n_frames):
            tiles = []
            for k in keys:
                img = cv2.imread(str(groups[k][i]))
                if img is None or img.shape[:2] != (h, w):
                    img = (
                        np.zeros((h, w, 3), dtype=np.uint8)
                        if img is None
                        else cv2.resize(img, (w, h))
                    )
                cv2.putText(
                    img, k, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (255, 255, 255), 2, cv2.LINE_AA,
                )
                tiles.append(img)
            while len(tiles) < rows * cols:
                tiles.append(np.zeros((h, w, 3), dtype=np.uint8))
            grid = np.vstack(
                [np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)]
            )
            writer.write(grid)
    finally:
        writer.release()
    print(f"  wrote {out_path} ({n_frames} frames, {w * cols}x{h * rows} @ {fps}fps)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path(__file__).parent / "street_view_output/train/images",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--no-grid", action="store_true", help="Skip the combined 2x2 grid video."
    )
    args = parser.parse_args()

    images_dir: Path = args.images_dir.resolve()
    if not images_dir.is_dir():
        raise SystemExit(f"images dir not found: {images_dir}")
    output_dir: Path = (args.output_dir or images_dir.parent / "videos").resolve()

    groups = collect_frames(images_dir)
    if not groups:
        raise SystemExit(f"no frames matching <name>_<idx>.png in {images_dir}")

    print(f"found {len(groups)} groups in {images_dir}:")
    for k, v in groups.items():
        print(f"  {k}: {len(v)} frames")

    for name, frames in groups.items():
        write_video(frames, output_dir / f"{name}.mp4", args.fps)

    if not args.no_grid and len(groups) > 1:
        write_grid_video(groups, output_dir / "grid.mp4", args.fps)


if __name__ == "__main__":
    main()
