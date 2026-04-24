#!/usr/bin/env python3
"""
Street-View Round-Trip Camera Generator
=======================================

Concept
-------
A virtual car drives down ONE lane of a straight street, stopping every
~1.5 m. At each stop a 4-camera rig captures front / left / right / back.

Two datasets are produced:

* train/  : clean lane trajectory, used to train 3DGS
* verify/ : same per-stop indices but each pose is perturbed (lateral
            offset, height jitter, yaw jitter) -> off-path views to
            measure 3DGS robustness when re-rendered from never-seen poses.

Outputs (per dataset folder):
    camera_poses.json         all per-frame pose data + intrinsics + GPS
    camera_extrinsics.npz     dict[name -> 4x4 world->cam]
    camera_intrinsics.npz     K + width/height/fov
    camera_gps.txt            csv: viewpoint, frame, ts, lat, lon, alt
    camera_metadata.txt       human-readable summary
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(
    format="[%(asctime)s.%(msecs)03d] [%(levelname)s] | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    "n_stops": 20,                # number of car stops along the lane
    "fps": 2,                     # 20 stops / 2 fps = 10 s
    "lane_offset_x": 3.0,         # car centerline x-coord (right lane on +X side)
    "lane_y_start": -15.0,        # first stop
    "lane_y_end":   +15.0,        # last stop
    "camera_height": 1.7,         # above ground
    "image_width": 1920,
    "image_height": 1080,
    "fov_degrees": 60,
}

GPS_REFERENCE = {
    "latitude": 40.7128,
    "longitude": -74.0060,
    "altitude": 10.0,
}

# Perturbation ranges for the verify dataset. All zero-mean uniform.
PERTURB_RANGES = {
    "lateral_x_m": 1.5,    # +/- 1.5 m off the lane center
    "longitudinal_y_m": 0.5,
    "height_z_m": 0.3,
    "yaw_deg": 12.0,
    "pitch_deg": 4.0,
    "roll_deg": 2.0,
}


# ---------------------------------------------------------------------------
# GPS helpers
# ---------------------------------------------------------------------------

EARTH_RADIUS_M = 6_378_137.0


def meters_to_gps(dx_m: float, dy_m: float, ref: dict) -> tuple:
    dlat = (dy_m / EARTH_RADIUS_M) * (180.0 / np.pi)
    dlon = (dx_m / (EARTH_RADIUS_M * np.cos(np.radians(ref["latitude"])))) * (180.0 / np.pi)
    return ref["latitude"] + dlat, ref["longitude"] + dlon


# ---------------------------------------------------------------------------
# Camera math (Blender / OpenGL convention: cam looks down -Z)
# ---------------------------------------------------------------------------

def look_at_matrix(eye: np.ndarray, target: np.ndarray, up=None) -> np.ndarray:
    if up is None:
        up = np.array([0.0, 0.0, 1.0])
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    R = np.stack([right, cam_up, -forward], axis=1)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = eye
    return M


def euler_xyz_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Intrinsic XYZ rotation (radians). Used for jitter on top of a base pose."""
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rotation_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def intrinsic_matrix(width: int, height: int, fov_deg: float) -> np.ndarray:
    fx = 0.5 * width / np.tan(0.5 * np.radians(fov_deg))
    fy = fx
    return np.array([[fx, 0, width / 2.0], [0, fy, height / 2.0], [0, 0, 1.0]])


# ---------------------------------------------------------------------------
# Pose generation
# ---------------------------------------------------------------------------

def lane_stops(params: dict) -> np.ndarray:
    """Return (n, 3) array of car stop positions along the lane."""
    n = params["n_stops"]
    ys = np.linspace(params["lane_y_start"], params["lane_y_end"], n)
    xs = np.full_like(ys, params["lane_offset_x"])
    zs = np.full_like(ys, params["camera_height"])
    return np.stack([xs, ys, zs], axis=1)


# Rig: 4 cameras at the same point, looking in cardinal directions.
# 'forward' is the car's heading (here +Y). Each camera's local "look at"
# direction is built relative to that.
RIG_DIRECTIONS = {
    "front": np.array([0.0, +1.0, 0.0]),
    "back":  np.array([0.0, -1.0, 0.0]),
    "left":  np.array([-1.0, 0.0, 0.0]),
    "right": np.array([+1.0, 0.0, 0.0]),
}


def make_pose(eye: np.ndarray, look_dir: np.ndarray,
              extra_rot: np.ndarray = None) -> np.ndarray:
    """Build cam->world from an eye and a look direction, optionally
    applying an additional 3x3 rotation in the camera's local frame."""
    target = eye + look_dir
    M = look_at_matrix(eye, target)
    if extra_rot is not None:
        M[:3, :3] = M[:3, :3] @ extra_rot
    return M


def build_dataset_poses(stops: np.ndarray,
                        params: dict,
                        perturb: bool,
                        rng: np.random.Generator,
                        opposite_lane: bool = False) -> dict:
    """Generate per-frame poses for one dataset (train OR verify).

    When `opposite_lane=True`, the rig is moved to the opposite lane
    (mirrored across the road centerline X=0) and driven in the reverse
    direction along Y. The four cameras of the rig are renamed so that
    'front' is still the car's heading, 'back' is behind, 'left' is the
    car's left, etc.  Per-stop jitter is still applied on top when
    `perturb=True`.
    """
    fps = params["fps"]
    pr = PERTURB_RANGES

    # Build effective stops + heading for this dataset.
    if opposite_lane:
        eff_stops = stops.copy()
        eff_stops[:, 0] = -eff_stops[:, 0]      # mirror lane to -X side
        eff_stops = eff_stops[::-1]             # drive the other way (reverse Y order)
        # Car heading is now -Y; rebuild the rig in that frame so 'front'
        # always means "the way the car is driving", etc.
        rig_dirs = {
            "front": np.array([0.0, -1.0, 0.0]),
            "back":  np.array([0.0, +1.0, 0.0]),
            "left":  np.array([+1.0, 0.0, 0.0]),
            "right": np.array([-1.0, 0.0, 0.0]),
        }
    else:
        eff_stops = stops
        rig_dirs = RIG_DIRECTIONS

    trajectories = {name: [] for name in rig_dirs}

    for i, stop_pos in enumerate(eff_stops):
        # Per-stop perturbation (shared across the 4 cameras of the rig
        # so they remain a coherent rig — but the WHOLE rig is off-path).
        if perturb:
            d_pos = np.array([
                rng.uniform(-pr["lateral_x_m"],     pr["lateral_x_m"]),
                rng.uniform(-pr["longitudinal_y_m"], pr["longitudinal_y_m"]),
                rng.uniform(-pr["height_z_m"],      pr["height_z_m"]),
            ])
            d_yaw   = np.radians(rng.uniform(-pr["yaw_deg"],   pr["yaw_deg"]))
            d_pitch = np.radians(rng.uniform(-pr["pitch_deg"], pr["pitch_deg"]))
            d_roll  = np.radians(rng.uniform(-pr["roll_deg"],  pr["roll_deg"]))
            extra_rot = euler_xyz_matrix(d_roll, d_pitch, d_yaw)
        else:
            d_pos = np.zeros(3)
            extra_rot = None

        eye = stop_pos + d_pos

        for vp_name, base_dir in rig_dirs.items():
            cam2world = make_pose(eye, base_dir, extra_rot)
            world2cam = np.linalg.inv(cam2world)
            quat = rotation_to_quat_wxyz(cam2world[:3, :3])

            trajectories[vp_name].append({
                "frame_index": i,
                "timestamp": i / fps,
                "position": eye.tolist(),
                "look_direction": base_dir.tolist(),
                "quat_wxyz": quat.tolist(),
                "cam_to_world": cam2world.tolist(),
                "world_to_cam": world2cam.tolist(),
                "perturbation": {
                    "d_pos": d_pos.tolist(),
                    "d_yaw_deg":   float(np.degrees(np.arctan2(extra_rot[1, 0], extra_rot[0, 0]))) if extra_rot is not None else 0.0,
                    "d_pitch_deg": float(np.degrees(np.arcsin(-extra_rot[2, 0])))                 if extra_rot is not None else 0.0,
                    "d_roll_deg":  float(np.degrees(np.arctan2(extra_rot[2, 1], extra_rot[2, 2]))) if extra_rot is not None else 0.0,
                },
            })

    return trajectories


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def save_outputs(out: Path, trajectories: dict, params: dict, gps_ref: dict, label: str):
    out.mkdir(parents=True, exist_ok=True)

    K = intrinsic_matrix(params["image_width"], params["image_height"], params["fov_degrees"])
    np.savez(out / "camera_intrinsics.npz", K=K,
             width=params["image_width"], height=params["image_height"],
             fov_degrees=params["fov_degrees"])

    extrinsics_dict = {
        f"{vp}_{f['frame_index']:04d}": np.array(f["world_to_cam"])
        for vp, frames in trajectories.items() for f in frames
    }
    np.savez(out / "camera_extrinsics.npz", **extrinsics_dict)

    with open(out / "camera_gps.txt", "w") as f:
        f.write("# viewpoint, frame_index, timestamp_s, latitude, longitude, altitude_m\n")
        for vp, frames in trajectories.items():
            for fr in frames:
                x, y, z = fr["position"]
                lat, lon = meters_to_gps(x, y, gps_ref)
                f.write(f"{vp}, {fr['frame_index']}, {fr['timestamp']:.3f}, "
                        f"{lat:.7f}, {lon:.7f}, {gps_ref['altitude'] + z:.3f}\n")

    poses = {
        "label": label,
        "params": params,
        "gps_reference": gps_ref,
        "intrinsics": {
            "K": K.tolist(),
            "width": params["image_width"],
            "height": params["image_height"],
            "fov_degrees": params["fov_degrees"],
        },
        "viewpoints": {},
    }
    for vp, frames in trajectories.items():
        out_frames = []
        for fr in frames:
            x, y, z = fr["position"]
            lat, lon = meters_to_gps(x, y, gps_ref)
            out_frames.append({
                "frame_index": fr["frame_index"],
                "timestamp": fr["timestamp"],
                "position_xyz": fr["position"],
                "look_direction": fr["look_direction"],
                "quat_wxyz": fr["quat_wxyz"],
                "gps": {"latitude": lat, "longitude": lon,
                        "altitude": gps_ref["altitude"] + z},
                "cam_to_world": fr["cam_to_world"],
                "world_to_cam": fr["world_to_cam"],
                "perturbation": fr["perturbation"],
            })
        poses["viewpoints"][vp] = out_frames
    with open(out / "camera_poses.json", "w") as f:
        json.dump(poses, f, indent=2)

    lines = [f"Dataset: {label}", "=" * 60]
    lines.append(f"Stops along lane    : {params['n_stops']}")
    lines.append(f"FPS                 : {params['fps']}")
    lines.append(f"Duration            : {params['n_stops']/params['fps']:.1f} s")
    lines.append(f"Lane center x       : {params['lane_offset_x']} m")
    lines.append(f"Lane y range        : [{params['lane_y_start']}, {params['lane_y_end']}] m")
    lines.append(f"Camera height       : {params['camera_height']} m")
    lines.append(f"Image               : {params['image_width']}x{params['image_height']} @ "
                 f"{params['fov_degrees']} deg HFOV")
    lines.append(f"GPS ref             : lat={gps_ref['latitude']}, lon={gps_ref['longitude']}, "
                 f"alt={gps_ref['altitude']} m")
    if label == "verify":
        lines.append("")
        mode = params.get("verify_mode", "jitter")
        lines.append(f"Verify mode         : {mode}")
        if mode == "opposite_lane":
            lines.append(f"  Lane mirrored to x = {-params['lane_offset_x']} m "
                         f"(opposite side of the road).")
            lines.append("  Driven in reverse Y direction; rig front/back/left/right")
            lines.append("  swapped to match the new heading.")
        lines.append("Per-stop perturbation ranges (uniform, zero-mean):")
        for k, v in PERTURB_RANGES.items():
            lines.append(f"  {k:22s}: +/- {v}")
    lines.append("")
    lines.append("Total frames        : " + str(sum(len(v) for v in trajectories.values())))
    with open(out / "camera_metadata.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info(f"[{label}] wrote {sum(len(v) for v in trajectories.values())} frames -> {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Round-trip street-view camera generator")
    parser.add_argument("-o", "--output", type=str, default="./street_view_output")
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("--n-stops", type=int, default=DEFAULT_PARAMS["n_stops"])
    parser.add_argument("--fps", type=int, default=DEFAULT_PARAMS["fps"])
    parser.add_argument("--lane-offset", type=float, default=DEFAULT_PARAMS["lane_offset_x"])
    parser.add_argument("--y-start", type=float, default=DEFAULT_PARAMS["lane_y_start"])
    parser.add_argument("--y-end", type=float, default=DEFAULT_PARAMS["lane_y_end"])
    parser.add_argument("--lat", type=float, default=GPS_REFERENCE["latitude"])
    parser.add_argument("--lon", type=float, default=GPS_REFERENCE["longitude"])
    parser.add_argument("--alt", type=float, default=GPS_REFERENCE["altitude"])
    parser.add_argument("--verify-mode",
                        choices=["jitter", "opposite_lane"],
                        default="opposite_lane",
                        help="How verify poses differ from train. "
                             "'jitter': same lane + small 6-DoF jitter. "
                             "'opposite_lane': mirrored to the other lane, "
                             "driven in the reverse direction, plus jitter.")
    args = parser.parse_args()

    params = dict(DEFAULT_PARAMS)
    params["n_stops"] = args.n_stops
    params["fps"] = args.fps
    params["lane_offset_x"] = args.lane_offset
    params["lane_y_start"] = args.y_start
    params["lane_y_end"] = args.y_end

    gps_ref = {"latitude": args.lat, "longitude": args.lon, "altitude": args.alt}

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output root : {out_root.resolve()}")
    logger.info(f"Seed        : {args.seed}")

    stops = lane_stops(params)
    rng = np.random.default_rng(args.seed)

    # Train: clean lane trajectory.
    train_traj = build_dataset_poses(stops, params, perturb=False, rng=rng)
    save_outputs(out_root / "train", train_traj, params, gps_ref, "train")

    # Verify: same per-stop indices but each rig pose perturbed off-path.
    verify_opposite = (args.verify_mode == "opposite_lane")
    params["verify_mode"] = args.verify_mode
    logger.info(f"Verify mode : {args.verify_mode}")
    verify_traj = build_dataset_poses(stops, params, perturb=True, rng=rng,
                                      opposite_lane=verify_opposite)
    save_outputs(out_root / "verify", verify_traj, params, gps_ref, "verify")

    logger.info("Done.")


if __name__ == "__main__":
    main()
