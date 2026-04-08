import struct, numpy as np
from PIL import Image
import os

depth_dir = 'data/colmap_pointcloud_dense/stereo/depth_maps'
out_dir = 'data/colmap_pointcloud_dense/stereo/depth_maps_png'
os.makedirs(out_dir, exist_ok=True)

def read_colmap_depth(path):
    with open(path, 'rb') as f:
        raw = f.read()
    # Header is text: "W&H&C&" followed by binary float32 data
    header_end = 0
    amp_count = 0
    for i, b in enumerate(raw):
        if b == ord('&'):
            amp_count += 1
            if amp_count == 3:
                header_end = i + 1
                break
    header = raw[:header_end].decode('ascii')
    parts = header.split('&')
    w, h, ch = int(parts[0]), int(parts[1]), int(parts[2])
    data = np.frombuffer(raw[header_end:], dtype=np.float32).reshape(h, w)
    return data

files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.geometric.bin')])
print(f'Converting {len(files)} geometric depth maps to PNG...')

for i, fname in enumerate(files):
    depth = read_colmap_depth(os.path.join(depth_dir, fname))
    valid = depth > 0
    if valid.any():
        dmin, dmax = depth[valid].min(), depth[valid].max()
        norm = np.zeros_like(depth, dtype=np.uint8)
        norm[valid] = (255 * (depth[valid] - dmin) / (dmax - dmin + 1e-8)).astype(np.uint8)
    else:
        norm = np.zeros_like(depth, dtype=np.uint8)
    out_name = fname.replace('.geometric.bin', '_depth.png')
    Image.fromarray(norm).save(os.path.join(out_dir, out_name))
    if (i+1) % 50 == 0 or i == 0:
        print(f'  [{i+1}/{len(files)}] {out_name}')

print(f'Done. Saved to {out_dir}')