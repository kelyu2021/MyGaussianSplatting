"""Find a ~10 s round-trip pass: two scene segments that traverse the same
stretch of road in opposite directions, each ~10 s long.

For each segment we copy the 6 surround-camera JPGs + the LIDAR_TOP point cloud
of every keyframe, plus a `manifest.csv` with timestamp / pose info.

Outputs:
  outputs/round_trip_10s/1/...   first pass
  outputs/round_trip_10s/2/...   opposite pass
  outputs/round_trip_10s/overview.png   side-by-side BEV trajectories of both
"""

import csv
import json
import os
import shutil
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

DATAROOT = "/Users/kevin/Data/nuscenes-v1.0-trainval"
VERSION = "v1.0-trainval"
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "round_trip_10s")

CAM_CHANNELS = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT",
]

# Match thresholds for the anchor keyframes.
POS_TOL  = 8.0
COS_TOL  = -0.85
MIN_DT   = 30.0
GRID     = 20.0
WINDOW_S = 10.0   # length of each trip segment (centered on the anchor)

# Stricter validation for the FULL segment (not just the anchor).
SEG_POS_TOL = 5.0   # bidirectional Hausdorff threshold (meters)
SEG_COS_TOL = -0.7  # matched headings must be roughly opposite


def yaw_from_quat(q):
    return Quaternion(q).yaw_pitch_roll[0]


def scene_has_all_cams(nusc, scene):
    sample = nusc.get("sample", scene["first_sample_token"])
    for ch in CAM_CHANNELS:
        sd = nusc.get("sample_data", sample["data"][ch])
        if not os.path.isfile(os.path.join(nusc.dataroot, sd["filename"])):
            return False
    return True


def collect_keyframes(nusc):
    log_to_loc = {log["token"]: log["location"] for log in nusc.log}
    good_scenes = [s for s in nusc.scene if scene_has_all_cams(nusc, s)]
    print(f"Scenes with all 6 cameras present on disk: "
          f"{len(good_scenes)}/{len(nusc.scene)}")

    rows = []
    for scene in good_scenes:
        loc = log_to_loc[scene["log_token"]]
        sample_token = scene["first_sample_token"]
        while sample_token:
            sample = nusc.get("sample", sample_token)
            sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            ego = nusc.get("ego_pose", sd["ego_pose_token"])
            x, y, _ = ego["translation"]
            rows.append({
                "token": sample_token,
                "scene": scene["name"],
                "scene_token": scene["token"],
                "location": loc,
                "x": x, "y": y,
                "yaw": yaw_from_quat(ego["rotation"]),
                "t": sample["timestamp"] * 1e-6,
            })
            sample_token = sample["next"]
    return rows


def _segment_poses(nusc, samples):
    """Return (N, 3) array of (x, y, yaw) for each sample in the window."""
    poses = []
    for s in samples:
        sd = nusc.get("sample_data", s["data"]["LIDAR_TOP"])
        ego = nusc.get("ego_pose", sd["ego_pose_token"])
        x, y, _ = ego["translation"]
        poses.append((x, y, yaw_from_quat(ego["rotation"])))
    return np.array(poses)


def _segments_are_round_trip(poses_a, poses_b):
    """Bidirectional check: every B-pose has an A-pose within SEG_POS_TOL,
    AND every A-pose has a B-pose within SEG_POS_TOL, AND the nearest
    headings are roughly opposite. This ensures the two segments cover the
    same physical stretch of road in opposite directions."""
    if len(poses_a) < 2 or len(poses_b) < 2:
        return False

    def _check(src, dst):
        sx, sy, syaw = src[:, 0], src[:, 1], src[:, 2]
        dx, dy, dyaw = dst[:, 0], dst[:, 1], dst[:, 2]
        for i in range(len(src)):
            d2 = (dx - sx[i]) ** 2 + (dy - sy[i]) ** 2
            k = int(np.argmin(d2))
            if d2[k] > SEG_POS_TOL ** 2:
                return False
            if np.cos(syaw[i] - dyaw[k]) > SEG_COS_TOL:
                return False
        return True

    return _check(poses_b, poses_a) and _check(poses_a, poses_b)


def find_pair(nusc, rows, scene_keyframes_cache):
    buckets = defaultdict(list)
    for i, r in enumerate(rows):
        buckets[(r["location"], int(r["x"] // GRID), int(r["y"] // GRID))].append(i)

    half = WINDOW_S / 2.0
    checked = 0
    for i, a in enumerate(rows):
        ax, ay = a["x"], a["y"]
        bx_i, by_i = int(ax // GRID), int(ay // GRID)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in buckets.get((a["location"], bx_i + dx, by_i + dy), ()):
                    if j <= i:
                        continue
                    b = rows[j]
                    if a["scene_token"] == b["scene_token"]:
                        continue
                    if abs(b["t"] - a["t"]) < MIN_DT:
                        continue
                    if (b["x"] - ax) ** 2 + (b["y"] - ay) ** 2 > POS_TOL ** 2:
                        continue
                    if np.cos(a["yaw"] - b["yaw"]) > COS_TOL:
                        continue

                    # Anchor passes -- now validate the whole 10s window.
                    samples_a = select_window(scene_keyframes_cache[a["scene_token"]],
                                              a["t"], half)
                    samples_b = select_window(scene_keyframes_cache[b["scene_token"]],
                                              b["t"], half)
                    poses_a = _segment_poses(nusc, samples_a)
                    poses_b = _segment_poses(nusc, samples_b)
                    checked += 1
                    if _segments_are_round_trip(poses_a, poses_b):
                        print(f"  validated full-segment match after "
                              f"{checked} candidate(s).")
                        return a, b, samples_a, samples_b
    print(f"  no valid pair after checking {checked} anchor candidates.")
    return None


def scene_keyframes(nusc, scene_token):
    """Ordered list of (sample_dict, t) for every keyframe in a scene."""
    scene = nusc.get("scene", scene_token)
    out = []
    tok = scene["first_sample_token"]
    while tok:
        s = nusc.get("sample", tok)
        out.append((s, s["timestamp"] * 1e-6))
        tok = s["next"]
    return out


def select_window(samples_with_t, anchor_t, half_window):
    """Pick keyframes whose timestamps lie within [anchor_t-half, anchor_t+half]."""
    return [s for s, t in samples_with_t
            if anchor_t - half_window <= t <= anchor_t + half_window]


def _camera_world_pose(nusc, sd_token):
    """Return (p_world (3,), q_world (w,x,y,z), R_cam2world (3,3)) for a camera sample_data."""
    sd = nusc.get("sample_data", sd_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    R_ew = Quaternion(ep["rotation"]).rotation_matrix
    t_ew = np.array(ep["translation"])
    R_se = Quaternion(cs["rotation"]).rotation_matrix
    t_se = np.array(cs["translation"])
    p_world = t_ew + R_ew @ t_se
    R_sw = R_ew @ R_se
    q_world = Quaternion(matrix=R_sw)
    return p_world, q_world, R_sw, sd, cs, ep


def export_segment(nusc, scene_name, samples, out_dir):
    """Copy 6 cam JPGs + LIDAR_TOP .pcd.bin + manifest.csv (with per-camera
    world poses) for each keyframe. Also writes calibrations.json with the
    intrinsics + sensor->ego extrinsics shared by all keyframes."""
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    cam_dir = os.path.join(out_dir, "cams")
    lidar_dir = os.path.join(out_dir, "lidar")
    os.makedirs(cam_dir)
    os.makedirs(lidar_dir)

    rows = []
    poses = []
    calibrations = {}  # ch -> {intrinsic, translation, rotation} in ego frame
    for k, sample in enumerate(samples):
        sd_lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego = nusc.get("ego_pose", sd_lidar["ego_pose_token"])
        t = sample["timestamp"] * 1e-6
        x, y, z = ego["translation"]
        yaw = yaw_from_quat(ego["rotation"])
        poses.append((x, y, yaw))

        # Copy LIDAR.
        src = os.path.join(nusc.dataroot, sd_lidar["filename"])
        lidar_dst = os.path.join(lidar_dir, f"kf{k:02d}_{os.path.basename(src)}")
        shutil.copy2(src, lidar_dst)

        # Per-camera: copy file + compute world pose.
        cam_paths = {}
        cam_world = {}  # ch -> (px, py, pz, qw, qx, qy, qz, ts)
        for ch in CAM_CHANNELS:
            sd_token = sample["data"][ch]
            sd = nusc.get("sample_data", sd_token)
            src = os.path.join(nusc.dataroot, sd["filename"])
            sub = os.path.join(cam_dir, ch)
            os.makedirs(sub, exist_ok=True)
            dst = os.path.join(sub, f"kf{k:02d}_{os.path.basename(src)}")
            shutil.copy2(src, dst)
            cam_paths[ch] = os.path.relpath(dst, out_dir)

            p_world, q_world, _, _, cs, _ = _camera_world_pose(nusc, sd_token)
            cam_world[ch] = (
                p_world[0], p_world[1], p_world[2],
                q_world.w, q_world.x, q_world.y, q_world.z,
                sd["timestamp"] * 1e-6,
            )
            if ch not in calibrations:
                calibrations[ch] = {
                    "image_size_wh": [sd["width"], sd["height"]],
                    "intrinsic_3x3": cs["camera_intrinsic"],
                    "translation_cam_in_ego": list(cs["translation"]),
                    "rotation_cam_in_ego_wxyz": list(cs["rotation"]),
                }

        # Build the per-keyframe row.
        row = {
            "kf": k,
            "sample_token": sample["token"],
            "timestamp_s": f"{t:.6f}",
            "ego_x": f"{x:.3f}", "ego_y": f"{y:.3f}", "ego_z": f"{z:.3f}",
            "ego_yaw_deg": f"{np.degrees(yaw):.2f}",
            "lidar": os.path.relpath(lidar_dst, out_dir),
        }
        for ch in CAM_CHANNELS:
            row[f"img_{ch}"] = cam_paths[ch]
        for ch in CAM_CHANNELS:
            px, py, pz, qw, qx, qy, qz, ts = cam_world[ch]
            row[f"{ch}_x"]  = f"{px:.3f}"
            row[f"{ch}_y"]  = f"{py:.3f}"
            row[f"{ch}_z"]  = f"{pz:.3f}"
            row[f"{ch}_qw"] = f"{qw:.6f}"
            row[f"{ch}_qx"] = f"{qx:.6f}"
            row[f"{ch}_qy"] = f"{qy:.6f}"
            row[f"{ch}_qz"] = f"{qz:.6f}"
            row[f"{ch}_ts"] = f"{ts:.6f}"
        rows.append(row)

    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(os.path.join(out_dir, "calibrations.json"), "w") as fh:
        json.dump(calibrations, fh, indent=2)

    return np.array(poses)


def render_overview(poses1, poses2, anchor_a, anchor_b, out_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(poses1[:, 0], poses1[:, 1], "-o", color="tab:blue",
            label="trip 1", markersize=3)
    ax.plot(poses2[:, 0], poses2[:, 1], "-o", color="tab:red",
            label="trip 2 (opposite)", markersize=3)
    ax.scatter([anchor_a["x"]], [anchor_a["y"]], marker="*", s=200,
               color="tab:blue", edgecolor="black", zorder=5, label="anchor 1")
    ax.scatter([anchor_b["x"]], [anchor_b["y"]], marker="*", s=200,
               color="tab:red", edgecolor="black", zorder=5, label="anchor 2")
    ax.set_aspect("equal")
    ax.set_title(f"Round-trip ~{int(WINDOW_S)} s windows  --  {anchor_a['location']}")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)

    print("\nIndexing keyframes ...")
    rows = collect_keyframes(nusc)

    print("Searching for a round-trip pair ...")
    # Cache per-scene keyframe lists once, used during full-segment validation.
    scene_kf_cache = {}
    for r in rows:
        if r["scene_token"] not in scene_kf_cache:
            scene_kf_cache[r["scene_token"]] = scene_keyframes(nusc, r["scene_token"])

    pair = find_pair(nusc, rows, scene_kf_cache)
    if pair is None:
        print("No matching pair found.")
        return
    a, b, samples_a, samples_b = pair
    print(f"\nAnchor pair:\n  A  scene={a['scene']}  yaw={np.degrees(a['yaw']):+6.1f}°"
          f"\n  B  scene={b['scene']}  yaw={np.degrees(b['yaw']):+6.1f}°"
          f"\n  dist={np.hypot(a['x']-b['x'], a['y']-b['y']):.1f} m,  loc={a['location']}")

    print(f"\nTrip 1: {len(samples_a)} keyframes "
          f"({samples_a[-1]['timestamp']*1e-6 - samples_a[0]['timestamp']*1e-6:.1f} s)")
    print(f"Trip 2: {len(samples_b)} keyframes "
          f"({samples_b[-1]['timestamp']*1e-6 - samples_b[0]['timestamp']*1e-6:.1f} s)")

    out1 = os.path.join(OUT_DIR, "1")
    out2 = os.path.join(OUT_DIR, "2")
    poses1 = export_segment(nusc, a["scene"], samples_a, out1)
    poses2 = export_segment(nusc, b["scene"], samples_b, out2)
    print(f"\nExported -> {out1}\nExported -> {out2}")

    overview = os.path.join(OUT_DIR, "overview.png")
    render_overview(poses1, poses2, a, b, overview)
    print(f"Overview -> {overview}")


if __name__ == "__main__":
    main()
