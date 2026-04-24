"""Find a pair of NuScenes keyframes that show the ego vehicle traversing the
same stretch of road in opposite directions (a 'round trip' pass), and render
the six surround-camera images for each pass side-by-side.

Search criteria
---------------
For every keyframe sample we record (x, y, yaw) of the ego vehicle. We then
look for two samples (A, B) that satisfy:
  * same map location
  * |xy_A - xy_B| <= POS_TOL meters
  * heading difference ~ 180 deg  (cos(yaw_A - yaw_B) <= COS_TOL)
  * timestamps at least MIN_DT seconds apart (so they aren't neighbors)

The first qualifying pair is rendered to outputs/round_trip_<sceneA>_<sceneB>.png.
"""

import os
from collections import defaultdict

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

DATAROOT = "/Users/kevin/Data/nuscenes-v1.0-trainval"
VERSION = "v1.0-trainval"
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

CAM_ORDER = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT",
]
# 4-row layout: pass A front, pass A back, pass B front, pass B back.
LAYOUT_ROWS = [
    ("A", ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]),
    ("A", ["CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT"]),
    ("B", ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]),
    ("B", ["CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT"]),
]

POS_TOL = 8.0          # meters
COS_TOL = -0.85        # cos(180 deg) = -1; allow ~ +-30 deg around opposite
MIN_DT  = 30.0         # seconds between the two samples (skip near-duplicates)
GRID    = 20.0         # meters; spatial bucket size for the search


def yaw_from_quat(q):
    """Z-axis yaw from a [w, x, y, z] quaternion."""
    return Quaternion(q).yaw_pitch_roll[0]


def _scene_has_all_cams(nusc, scene):
    """Verify that the first sample's six camera files exist on disk."""
    sample = nusc.get("sample", scene["first_sample_token"])
    for ch in CAM_ORDER:
        sd = nusc.get("sample_data", sample["data"][ch])
        if not os.path.isfile(os.path.join(nusc.dataroot, sd["filename"])):
            return False
    return True


def collect_keyframes(nusc):
    """Return list of dicts: {token, scene, location, x, y, yaw, t}.

    Only scenes whose camera JPGs are actually present on disk are kept.
    """
    log_to_loc = {log["token"]: log["location"] for log in nusc.log}
    good_scenes = [s for s in nusc.scene if _scene_has_all_cams(nusc, s)]
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


def find_round_trip_pairs(rows, max_pairs=10, min_pair_separation=15.0):
    """Find up to `max_pairs` opposite-direction passes.

    To avoid returning many near-duplicate pairs at the same spot, every chosen
    pair 'claims' a `min_pair_separation`-meter exclusion radius around its
    midpoint; subsequent candidates falling inside any claimed radius are
    skipped.
    """
    buckets = defaultdict(list)
    for i, r in enumerate(rows):
        key = (r["location"], int(r["x"] // GRID), int(r["y"] // GRID))
        buckets[key].append(i)

    pairs = []
    used_samples = set()
    claimed = []  # list of (location, mx, my)

    for i, a in enumerate(rows):
        if a["token"] in used_samples:
            continue
        ax, ay = a["x"], a["y"]
        bx_i, by_i = int(ax // GRID), int(ay // GRID)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in buckets.get((a["location"], bx_i + dx, by_i + dy), ()):
                    if j <= i:
                        continue
                    b = rows[j]
                    if b["token"] in used_samples:
                        continue
                    if a["scene_token"] == b["scene_token"]:
                        continue
                    if abs(b["t"] - a["t"]) < MIN_DT:
                        continue
                    if (b["x"] - ax) ** 2 + (b["y"] - ay) ** 2 > POS_TOL ** 2:
                        continue
                    if np.cos(a["yaw"] - b["yaw"]) > COS_TOL:
                        continue
                    mx, my = (ax + b["x"]) / 2, (ay + b["y"]) / 2
                    if any(c[0] == a["location"]
                           and (c[1] - mx) ** 2 + (c[2] - my) ** 2
                               < min_pair_separation ** 2
                           for c in claimed):
                        continue
                    pairs.append((a, b))
                    used_samples.add(a["token"])
                    used_samples.add(b["token"])
                    claimed.append((a["location"], mx, my))
                    if len(pairs) >= max_pairs:
                        return pairs
                    break  # move to next outer i
                else:
                    continue
                break
            else:
                continue
            break
    return pairs


def render_pair(nusc, a, b, out_path):
    fig, axes = plt.subplots(4, 3, figsize=(15, 14))
    samples = {"A": nusc.get("sample", a["token"]),
               "B": nusc.get("sample", b["token"])}
    info = {"A": a, "B": b}
    for r, (tag, channels) in enumerate(LAYOUT_ROWS):
        sample = samples[tag]
        for c, ch in enumerate(channels):
            cam = nusc.get("sample_data", sample["data"][ch])
            img = mpimg.imread(os.path.join(nusc.dataroot, cam["filename"]))
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{tag} - {ch}", fontsize=10)
        # Row label on the left-most subplot.
        meta = info[tag]
        axes[r, 0].set_ylabel(
            f"{meta['scene']}\nyaw={np.degrees(meta['yaw']):+6.1f} deg",
            fontsize=10,
        )
    fig.suptitle(
        f"Same spot, opposite directions  --  {a['location']}  "
        f"(dist={np.hypot(a['x']-b['x'], a['y']-b['y']):.1f} m, "
        f"d_yaw={np.degrees(np.arctan2(np.sin(a['yaw']-b['yaw']), np.cos(a['yaw']-b['yaw']))):+.1f} deg, "
        f"dt={(b['t']-a['t']):.0f} s)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    pair_dir = os.path.join(OUT_DIR, "round_trip")
    os.makedirs(pair_dir, exist_ok=True)
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)

    print("\nIndexing keyframes ...")
    rows = collect_keyframes(nusc)
    print(f"Indexed {len(rows):,} keyframes.")

    print("Searching for opposite-direction passes ...")
    pairs = find_round_trip_pairs(rows, max_pairs=10, min_pair_separation=15.0)
    if not pairs:
        print("No matching pairs found; try relaxing POS_TOL / COS_TOL.")
        return

    print(f"\nFound {len(pairs)} pairs.")
    for k, (a, b) in enumerate(pairs, 1):
        print(f"  [{k:02d}] {a['scene']} <-> {b['scene']}  "
              f"loc={a['location']:25s} "
              f"d={np.hypot(a['x']-b['x'], a['y']-b['y']):4.1f}m  "
              f"d_yaw={np.degrees(np.arctan2(np.sin(a['yaw']-b['yaw']), np.cos(a['yaw']-b['yaw']))):+6.1f} deg")
        out = os.path.join(pair_dir, f"pair_{k:02d}_{a['scene']}_vs_{b['scene']}.png")
        render_pair(nusc, a, b, out)
        print(f"       -> {out}")


if __name__ == "__main__":
    main()
