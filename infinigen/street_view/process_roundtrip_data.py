#!/usr/bin/env python3
"""
Utility functions for processing round-trip camera data for Gaussian Splatting.
Converts camera extrinsics to common GS formats (COLMAP, NeRF, Gaussian-SLAM).
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import argparse


def load_camera_metadata(output_folder: Path) -> Dict:
    """Load all camera metadata from output folder."""
    metadata = {
        'camera_metadata': Path(output_folder) / 'camera_metadata.txt',
        'camera_gps': Path(output_folder) / 'camera_gps.txt',
        'camera_extrinsics': Path(output_folder) / 'camera_extrinsics.npz',
    }
    
    # Load extrinsics
    extrinsics_data = np.load(metadata['camera_extrinsics'])
    metadata['extrinsics'] = {k: extrinsics_data[k] for k in extrinsics_data.files}
    
    # Load GPS data
    gps_data = []
    with open(metadata['camera_gps'], 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split(',')
            if len(parts) >= 5:
                gps_data.append({
                    'viewpoint': parts[0].strip(),
                    'latitude': float(parts[1]),
                    'longitude': float(parts[2]),
                    'altitude': float(parts[3]),
                    'frame_index': int(parts[4]),
                })
    metadata['gps'] = gps_data
    
    return metadata


def export_to_colmap_format(
    output_folder: Path,
    camera_intrinsics: np.ndarray = None,
    image_width: int = 1920,
    image_height: int = 1080,
) -> Path:
    """
    Export camera data to COLMAP format (cameras.txt, images.txt, points3d.txt).
    
    Args:
        output_folder: Path to folder with camera data
        camera_intrinsics: 3x3 camera intrinsic matrix (K matrix)
        image_width: Image width in pixels
        image_height: Image height in pixels
    
    Returns:
        Path to COLMAP sparse directory
    """
    metadata = load_camera_metadata(output_folder)
    extrinsics = metadata['extrinsics']
    gps_data = metadata['gps']
    
    # Default intrinsic matrix if not provided
    if camera_intrinsics is None:
        focal_length = max(image_width, image_height) / (2 * np.tan(np.radians(25)))
        camera_intrinsics = np.array([
            [focal_length, 0, image_width / 2],
            [0, focal_length, image_height / 2],
            [0, 0, 1],
        ])
    
    colmap_folder = Path(output_folder) / 'colmap' / 'sparse'
    colmap_folder.mkdir(parents=True, exist_ok=True)
    
    # Write cameras.txt
    cameras_txt = colmap_folder / 'cameras.txt'
    with open(cameras_txt, 'w') as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        f.write('# Number of cameras: 1\n')
        f.write(f'1 PINHOLE {image_width} {image_height} {camera_intrinsics[0, 0]} '
                f'{camera_intrinsics[0, 2]} {camera_intrinsics[1, 2]}\n')
    
    # Write images.txt
    images_txt = colmap_folder / 'images.txt'
    with open(images_txt, 'w') as f:
        f.write('# Image list with two lines of data per image:\n')
        f.write('# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        
        image_id = 1
        for frame_name in sorted(extrinsics.keys()):
            # Extract pose from extrinsic matrix
            extrinsic = extrinsics[frame_name]  # World to camera
            
            # Convert to camera to world for COLMAP
            cam_to_world = np.linalg.inv(extrinsic)
            rotation = cam_to_world[:3, :3]
            translation = cam_to_world[:3, 3]
            
            # Convert rotation matrix to quaternion (w, x, y, z)
            from scipy.spatial.transform import Rotation
            quat = Rotation.from_matrix(rotation).as_quat()  # Returns [x, y, z, w]
            quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
            
            f.write(f'{image_id} {quat_wxyz[0]:.6f} {quat_wxyz[1]:.6f} '
                    f'{quat_wxyz[2]:.6f} {quat_wxyz[3]:.6f} '
                    f'{translation[0]:.6f} {translation[1]:.6f} {translation[2]:.6f} '
                    f'1 {frame_name}.png\n')
            image_id += 1
        
        f.write('\n')
    
    # Write points3d.txt (empty for now)
    points3d_txt = colmap_folder / 'points3d.txt'
    with open(points3d_txt, 'w') as f:
        f.write('# 3D point list with one line of data per point:\n')
        f.write('# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
    
    print(f"Exported COLMAP format to: {colmap_folder}")
    return colmap_folder


def export_to_nerf_format(
    output_folder: Path,
    camera_intrinsics: np.ndarray = None,
    scale_factor: float = 1.0,
) -> Path:
    """
    Export camera data to NeRF/Instant-NGP JSON format.
    
    Args:
        output_folder: Path to folder with camera data
        camera_intrinsics: 3x3 camera intrinsic matrix
        scale_factor: Scale factor for camera positions
    
    Returns:
        Path to transforms.json
    """
    metadata = load_camera_metadata(output_folder)
    extrinsics = metadata['extrinsics']
    
    # Default intrinsics
    if camera_intrinsics is None:
        camera_intrinsics = np.eye(3)
    
    frames = []
    for frame_name in sorted(extrinsics.keys()):
        extrinsic = extrinsics[frame_name]
        
        # NeRF format uses camera-to-world transformation
        cam_to_world = np.linalg.inv(extrinsic)
        
        frames.append({
            'file_path': f'./images/{frame_name}.png',
            'transform_matrix': cam_to_world.tolist(),
        })
    
    # Create transforms JSON
    transforms = {
        'fl_x': camera_intrinsics[0, 0],
        'fl_y': camera_intrinsics[1, 1],
        'k1': 0.0,
        'k2': 0.0,
        'p1': 0.0,
        'p2': 0.0,
        'cx': camera_intrinsics[0, 2],
        'cy': camera_intrinsics[1, 2],
        'w': 1920,
        'h': 1080,
        'aabb_scale': 1,
        'scale': scale_factor,
        'offset': [0, 0, 0],
        'frames': frames,
    }
    
    transforms_file = Path(output_folder) / 'transforms.json'
    with open(transforms_file, 'w') as f:
        json.dump(transforms, f, indent=2)
    
    print(f"Exported NeRF format to: {transforms_file}")
    return transforms_file


def export_to_gaussian_slam_format(
    output_folder: Path,
) -> Path:
    """
    Export camera data to Gaussian-SLAM camera trajectory format.
    Format: frame_id, timestamp, tx, ty, tz, qx, qy, qz, qw
    
    Args:
        output_folder: Path to folder with camera data
    
    Returns:
        Path to trajectory.txt
    """
    metadata = load_camera_metadata(output_folder)
    extrinsics = metadata['extrinsics']
    gps_data = metadata['gps']
    
    trajectory_file = Path(output_folder) / 'trajectory.txt'
    
    with open(trajectory_file, 'w') as f:
        f.write('# frame_id timestamp tx ty tz qx qy qz qw\n')
        
        for i, frame_name in enumerate(sorted(extrinsics.keys())):
            extrinsic = extrinsics[frame_name]
            
            # Camera to world transformation
            cam_to_world = np.linalg.inv(extrinsic)
            translation = cam_to_world[:3, 3]
            
            # Extract quaternion from rotation matrix
            from scipy.spatial.transform import Rotation
            rotation = cam_to_world[:3, :3]
            quat = Rotation.from_matrix(rotation).as_quat()  # [x, y, z, w]
            
            timestamp = i * 0.5  # 2 fps = 0.5 seconds per frame
            f.write(f'{i} {timestamp:.3f} {translation[0]:.6f} {translation[1]:.6f} '
                    f'{translation[2]:.6f} {quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} '
                    f'{quat[3]:.6f}\n')
    
    print(f"Exported Gaussian-SLAM format to: {trajectory_file}")
    return trajectory_file


def split_by_viewpoint(
    output_folder: Path,
    output_splits_folder: Path = None,
) -> Dict[str, Path]:
    """
    Split data by viewpoint (front, left, back, right) into separate folders.
    Useful for training/testing split (e.g., train on front, test on back).
    
    Args:
        output_folder: Path to folder with camera data
        output_splits_folder: Base folder for splits (default: output_folder/splits)
    
    Returns:
        Dict mapping viewpoint name to folder path
    """
    if output_splits_folder is None:
        output_splits_folder = Path(output_folder) / 'splits'
    
    metadata = load_camera_metadata(output_folder)
    extrinsics = metadata['extrinsics']
    gps_data = metadata['gps']
    
    splits = {}
    
    # Group by viewpoint
    viewpoint_groups = {}
    for frame_name in extrinsics.keys():
        viewpoint = frame_name.split('_')[0]
        if viewpoint not in viewpoint_groups:
            viewpoint_groups[viewpoint] = []
        viewpoint_groups[viewpoint].append(frame_name)
    
    # Create separate folders and export data for each viewpoint
    for viewpoint, frame_names in viewpoint_groups.items():
        viewpoint_folder = output_splits_folder / viewpoint
        viewpoint_folder.mkdir(parents=True, exist_ok=True)
        
        # Export extrinsics for this viewpoint
        viewpoint_extrinsics = {
            name: extrinsics[name] for name in frame_names
        }
        np.savez(
            viewpoint_folder / 'camera_extrinsics.npz',
            **viewpoint_extrinsics
        )
        
        # Export GPS for this viewpoint
        with open(viewpoint_folder / 'camera_gps.txt', 'w') as f:
            f.write('# Viewpoint, Latitude, Longitude, Altitude (m), Frame Index\n')
            for gps in gps_data:
                if gps['viewpoint'] == viewpoint:
                    f.write(f"{gps['viewpoint']}, {gps['latitude']:.6f}, "
                            f"{gps['longitude']:.6f}, {gps['altitude']:.2f}, "
                            f"{gps['frame_index']}\n")
        
        splits[viewpoint] = viewpoint_folder
        print(f"Split '{viewpoint}': {len(frame_names)} frames -> {viewpoint_folder}")
    
    return splits


def print_summary(output_folder: Path):
    """Print summary of generated camera data."""
    metadata = load_camera_metadata(output_folder)
    extrinsics = metadata['extrinsics']
    gps_data = metadata['gps']
    
    # Group by viewpoint
    viewpoint_counts = {}
    for frame_name in extrinsics.keys():
        viewpoint = frame_name.split('_')[0]
        viewpoint_counts[viewpoint] = viewpoint_counts.get(viewpoint, 0) + 1
    
    print("\n" + "=" * 70)
    print("ROUND-TRIP CAMERA DATA SUMMARY")
    print("=" * 70)
    print(f"Total frames: {len(extrinsics)}")
    print(f"Viewpoints: {', '.join(sorted(viewpoint_counts.keys()))}")
    print()
    
    for viewpoint in sorted(viewpoint_counts.keys()):
        gps_points = [g for g in gps_data if g['viewpoint'] == viewpoint]
        if gps_points:
            first_gps = gps_points[0]
            print(f"  {viewpoint:8} | Frames: {viewpoint_counts[viewpoint]:3} | "
                  f"GPS: ({first_gps['latitude']:.6f}, {first_gps['longitude']:.6f}) | "
                  f"Alt: {first_gps['altitude']:.1f}m")
    
    print("=" * 70)
    print("\nOutput files:")
    print(f"  - {Path(output_folder) / 'camera_metadata.txt'}")
    print(f"  - {Path(output_folder) / 'camera_gps.txt'}")
    print(f"  - {Path(output_folder) / 'camera_extrinsics.npz'}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Process round-trip camera data for Gaussian Splatting'
    )
    parser.add_argument(
        'output_folder',
        type=str,
        help='Path to output folder from data_generate.py'
    )
    parser.add_argument(
        '--export-colmap',
        action='store_true',
        help='Export to COLMAP format'
    )
    parser.add_argument(
        '--export-nerf',
        action='store_true',
        help='Export to NeRF/Instant-NGP format'
    )
    parser.add_argument(
        '--export-slam',
        action='store_true',
        help='Export to Gaussian-SLAM trajectory format'
    )
    parser.add_argument(
        '--split-viewpoints',
        action='store_true',
        help='Split data by viewpoint (train/test)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Export to all formats'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print data summary'
    )
    
    args = parser.parse_args()
    output_folder = Path(args.output_folder)
    
    if not output_folder.exists():
        print(f"Error: Output folder not found: {output_folder}")
        return
    
    # Print summary
    if args.summary or args.all:
        print_summary(output_folder)
    
    # Export to various formats
    if args.export_colmap or args.all:
        export_to_colmap_format(output_folder)
    
    if args.export_nerf or args.all:
        export_to_nerf_format(output_folder)
    
    if args.export_slam or args.all:
        export_to_gaussian_slam_format(output_folder)
    
    if args.split_viewpoints or args.all:
        split_by_viewpoint(output_folder)
    
    if not any([args.export_colmap, args.export_nerf, args.export_slam,
                args.split_viewpoints, args.summary, args.all]):
        print("Use --summary to print data overview")
        print("Use --export-colmap to export to COLMAP format")
        print("Use --export-nerf to export to NeRF format")
        print("Use --export-slam to export to Gaussian-SLAM format")
        print("Use --split-viewpoints to split data by viewpoint")
        print("Use --all to export to all formats")


if __name__ == '__main__':
    main()
