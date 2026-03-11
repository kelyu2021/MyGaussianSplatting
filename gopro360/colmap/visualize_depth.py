#!/usr/bin/env python3
"""
Visualize COLMAP dense depth/normal .bin files.

Usage
-----
  # Visualize a single depth map
  python visualize_depth.py output/gopro360/dense/stereo/depth_maps/frame_000050_front.png.geometric.bin

  # Visualize a single normal map
  python visualize_depth.py output/gopro360/dense/stereo/normal_maps/frame_000050_front.png.geometric.bin

  # Batch: save PNGs for all geometric depth maps (no GUI needed)
  python visualize_depth.py --batch output/gopro360/dense/stereo/depth_maps --pattern "*.geometric.bin" --save_dir output/gopro360/dense/depth_previews

  # Batch: save PNGs for all normal maps
  python visualize_depth.py --batch output/gopro360/dense/stereo/normal_maps --pattern "*.geometric.bin" --save_dir output/gopro360/dense/normal_previews
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_colmap_bin(path: str | Path) -> np.ndarray:
    """Read a COLMAP .bin depth or normal map.

    Format: ASCII header "<width>&<height>&<channels>&" followed by
            raw float32 pixel data (W * H * C floats).

    Returns
    -------
    np.ndarray
        Shape (H, W) for depth maps (1 channel) or (H, W, 3) for normal maps.
    """
    with open(path, "rb") as f:
        raw = f.read()

    # Parse ASCII header: read until we find 3 '&'-separated numbers
    # Header looks like "1024&1024&1&" (no null terminator, data follows immediately)
    header_end = 0
    ampersand_count = 0
    for i, byte in enumerate(raw):
        if byte == ord("&"):
            ampersand_count += 1
            if ampersand_count == 3:
                header_end = i + 1
                break

    header_str = raw[:header_end].decode("ascii")
    parts = header_str.split("&")
    width = int(parts[0])
    height = int(parts[1])
    channels = int(parts[2])

    data = np.frombuffer(raw[header_end:], dtype=np.float32)
    expected = width * height * channels
    if data.size != expected:
        raise ValueError(
            f"Expected {expected} floats ({width}x{height}x{channels}), got {data.size}"
        )
    arr = data.reshape((height, width, channels))
    return arr.squeeze()  # (H,W) if 1-channel, (H,W,3) if 3-channel


def visualize_single(path: Path, save_path: Path | None = None) -> None:
    """Visualize one .bin file."""
    arr = read_colmap_bin(path)
    is_normal = arr.ndim == 3 and arr.shape[2] == 3

    fig, axes = plt.subplots(1, 1, figsize=(12, 8))

    if is_normal:
        # Normal map: remap [-1,1] → [0,1] for display
        display = (arr + 1.0) / 2.0
        display = np.clip(display, 0, 1)
        axes.imshow(display)
        axes.set_title(f"Normal Map: {path.name}", fontsize=10)
    else:
        # Depth map
        valid = arr[arr > 0]
        if valid.size == 0:
            print(f"WARNING: No valid depth pixels in {path}")
            vmin, vmax = 0, 1
        else:
            # Use percentiles to exclude outliers
            vmin, vmax = np.percentile(valid, [2, 98])

        im = axes.imshow(arr, cmap="turbo", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=axes, label="Depth", shrink=0.8)
        total_px = arr.size
        valid_px = valid.size
        axes.set_title(
            f"Depth: {path.name}\n"
            f"Shape: {arr.shape}  |  Range: [{arr.min():.2f}, {arr.max():.2f}]  |  "
            f"Valid: {valid_px:,}/{total_px:,} ({100*valid_px/total_px:.1f}%)",
            fontsize=9,
        )

    axes.axis("off")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def batch_visualize(
    folder: Path, pattern: str, save_dir: Path, max_files: int = 0
) -> None:
    """Batch convert .bin files to PNG previews."""
    files = sorted(folder.glob(pattern))
    if not files:
        print(f"No files matching '{pattern}' in {folder}")
        sys.exit(1)

    if max_files > 0:
        files = files[:max_files]

    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(files)} files → {save_dir}/")

    for i, f in enumerate(files, 1):
        out_name = f.name.replace(".bin", ".png")
        save_path = save_dir / out_name
        print(f"  [{i}/{len(files)}] {f.name}", end="")
        try:
            visualize_single(f, save_path=save_path)
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone! {len(files)} previews saved to {save_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize COLMAP .bin depth/normal maps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "file", nargs="?", type=str,
        help="Path to a single .bin file to visualize",
    )
    parser.add_argument(
        "--batch", type=str,
        help="Folder containing .bin files for batch processing",
    )
    parser.add_argument(
        "--pattern", type=str, default="*.geometric.bin",
        help="Glob pattern for batch mode (default: *.geometric.bin)",
    )
    parser.add_argument(
        "--save_dir", type=str,
        help="Directory to save PNG previews (batch mode). If omitted in single mode, shows interactive plot.",
    )
    parser.add_argument(
        "--save", type=str,
        help="Save path for single file mode (e.g. depth_preview.png)",
    )
    parser.add_argument(
        "--max_files", type=int, default=0,
        help="Limit number of files in batch mode (0 = all)",
    )
    args = parser.parse_args()

    if args.batch:
        if not args.save_dir:
            print("ERROR: --save_dir is required for batch mode", file=sys.stderr)
            sys.exit(1)
        batch_visualize(
            Path(args.batch), args.pattern, Path(args.save_dir), args.max_files
        )
    elif args.file:
        p = Path(args.file)
        if not p.exists():
            print(f"ERROR: File not found: {p}", file=sys.stderr)
            sys.exit(1)
        save = Path(args.save) if args.save else None
        visualize_single(p, save_path=save)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
