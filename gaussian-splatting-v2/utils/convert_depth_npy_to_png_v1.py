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
    depth = np.load(npy_path).astype(np.float32)  # metric depth in metres

    # Inverse depth: far = small value, near = large value
    # Guard against zero/negative depth
    inv_depth = np.where(depth > 0, 1.0 / depth, 0.0)

    # Normalise to [0, 65535] using per-image max so relative structure is preserved.
    # make_depth_scale.py computes a per-image scale/offset to align with COLMAP,
    # so the absolute scale here does not matter.
    max_val = inv_depth.max()
    if max_val > 0:
        inv_depth_u16 = (inv_depth / max_val * 65535).astype(np.uint16)
    else:
        inv_depth_u16 = np.zeros_like(inv_depth, dtype=np.uint16)

    # Strip '_depth_raw' suffix and change extension to .png
    stem = npy_path.stem  # e.g. '0001_front_depth_raw'
    if stem.endswith('_depth_raw'):
        stem = stem[:-len('_depth_raw')]
    out_name = stem + '.png'

    out_path = output_dir / out_name
    cv2.imwrite(str(out_path), inv_depth_u16)
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
