"""
Convert *_depth_raw.npy (metric depth, metres) to 16-bit inverse-depth PNGs
expected by make_depth_scale.py and the 3DGS depth-regularisation pipeline.

Output filename: {stem}.png  (strips '_depth_raw' suffix)
Output encoding: uint16, where 0xFFFF = max inverse depth in the image.
                 make_depth_scale.py divides by 2^16 to recover float values.

Usage:
    python utils/convert_depth_npy_to_png.py \
        --input_dir  data/cubemap_faces_da2 \
        --output_dir data/cubemap_faces_da2_png
"""

import argparse
import os
import numpy as np
import cv2
from pathlib import Path


def convert(npy_path: Path, output_dir: Path) -> None:
    depth = np.load(npy_path).astype(np.float32)  # DA2 inverse depth (disparity)

    print(f'da2 inverse depth  min={depth.min():.4f}  max={depth.max():.4f}')

    # Valid pixels: DA2 values near zero or negative are sky/invalid and produce
    # astronomically large direct-depth values that crush the normalisation range.
    # Use the 1st percentile of positive values as a lower bound to exclude them.
    pos_mask = depth > 0
    if pos_mask.any():
        low_thresh = np.percentile(depth[pos_mask], 0.3)
    else:
        low_thresh = 0.0
    valid_mask = depth >= low_thresh

    direct_depth = np.where(valid_mask, 1.0 / depth, 0.0)

    # Normalise to [0, 65535] over valid pixels only (percentile clip for outliers).
    # make_depth_scale.py fits a per-image scale/offset to COLMAP, so absolute
    # scale does not matter — only the relative structure needs to be preserved.
    valid_vals = direct_depth[valid_mask]
    print(f'direct depth (valid)  min={valid_vals.min():.4f}  max={valid_vals.max():.4f}')
    if valid_vals.size > 0:
        d_lo, d_hi = np.percentile(valid_vals, [0.3, 99.7])
        direct_depth_clipped = np.clip(direct_depth, d_lo, d_hi)
        direct_depth_clipped[~valid_mask] = 0.0
        scale = d_hi - d_lo
        direct_depth_u16 = ((direct_depth_clipped - d_lo) / (scale + 1e-8) * 65535).astype(np.uint16)
        direct_depth_u16[~valid_mask] = 0
    else:
        direct_depth_u16 = np.zeros_like(direct_depth, dtype=np.uint16)

    # Strip '_depth_raw' suffix and change extension to .png
    stem = npy_path.stem  # e.g. '0001_front_depth_raw'
    if stem.endswith('_depth_raw'):
        stem = stem[:-len('_depth_raw')]
    out_name = stem + '.png'

    out_path = output_dir / out_name
    cv2.imwrite(str(out_path), direct_depth_u16)
    print(f"  {npy_path.name} -> {out_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True,
                        help='Directory containing *_depth_raw.npy files')
    parser.add_argument('--output_dir', required=True,
                        help='Directory to write 16-bit PNG inverse-depth files')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(input_dir.glob('*_depth_raw.npy'))
    if not npy_files:
        print(f'No *_depth_raw.npy files found in {input_dir}')
        return

    print(f'Converting {len(npy_files)} files ...')
    for f in npy_files:
        convert(f, output_dir)
    print('Done.')


if __name__ == '__main__':
    main()
