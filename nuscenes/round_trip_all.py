"""Find ALL round-trip passes in the dataset and export each into its own
subfolder under ``outputs/round_trips/``.

A "round-trip" here means two scene segments (each ~WINDOW_S long, taken from
two different scenes) that traverse the same stretch of road in roughly
opposite directions.

Layout::

    outputs/round_trips/
        index.csv                   # one row per accepted pair
        pair_000/
            overview.png
            1/  cams/ lidar/ manifest.csv calibrations.json
            2/  cams/ lidar/ manifest.csv calibrations.json
        pair_001/
            ...

Reuses helpers from ``round_trip_10s.py`` (segment validation, export).
"""

import csv
import os
from collections import defaultdict

import numpy as np
from nuscenes.nuscenes import NuScenes

from round_trip_10s import (
    CAM_CHANNELS,
    DATAROOT,
    GRID,
    MIN_DT,
    POS_TOL,
    COS_TOL,
    WINDOW_S,
    VERSION,
    _segment_poses,
    _segments_are_round_trip,
    collect_keyframes,
    export_segment,
    render_overview,
    scene_keyframes,
    select_window,
)

OUT_ROOT = os.path.join(os.path.dirname(__file__), "outputs", "round_trips")


def find_all_pairs(nusc, rows, scene_kf_cache):
    """Yield every unique round-trip pair.

    Dedup rule: at most one pair per unordered scene-pair (scene_a, scene_b).
    If multiple anchor candidates match the same scene-pair, the first one
    that passes the full-segment check wins.
    """
    buckets = defaultdict(list)
    for i, r in enumerate(rows):
        buckets[(r["location"], int(r["x"] // GRID), int(r["y"] // GRID))].append(i)

    half = WINDOW_S / 2.0
    accepted_scene_pairs = set()
    pairs = []
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
                    key = tuple(sorted((a["scene_token"], b["scene_token"])))
                    if key in accepted_scene_pairs:
                        continue
                    if abs(b["t"] - a["t"]) < MIN_DT:
                        continue
                    if (b["x"] - ax) ** 2 + (b["y"] - ay) ** 2 > POS_TOL ** 2:
                        continue
                    if np.cos(a["yaw"] - b["yaw"]) > COS_TOL:
                        continue

                    samples_a = select_window(scene_kf_cache[a["scene_token"]],
                                              a["t"], half)
                    samples_b = select_window(scene_kf_cache[b["scene_token"]],
                                              b["t"], half)
                    if len(samples_a) < 2 or len(samples_b) < 2:
                        continue
                    poses_a = _segment_poses(nusc, samples_a)
                    poses_b = _segment_poses(nusc, samples_b)
                    checked += 1
                    if _segments_are_round_trip(poses_a, poses_b):
                        accepted_scene_pairs.add(key)
                        pairs.append((a, b, samples_a, samples_b))
                        print(f"  [{len(pairs):03d}] {a['scene']} <-> {b['scene']}  "
                              f"loc={a['location']}  "
                              f"dist={np.hypot(a['x']-b['x'], a['y']-b['y']):.1f} m  "
                              f"({len(samples_a)}/{len(samples_b)} kf)")
    print(f"\nChecked {checked} anchor candidates; "
          f"accepted {len(pairs)} unique round-trip pairs.")
    return pairs


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)

    print("\nIndexing keyframes ...")
    rows = collect_keyframes(nusc)

    print("Caching per-scene keyframe lists ...")
    scene_kf_cache = {}
    for r in rows:
        if r["scene_token"] not in scene_kf_cache:
            scene_kf_cache[r["scene_token"]] = scene_keyframes(nusc, r["scene_token"])

    print("Searching for round-trip pairs ...")
    pairs = find_all_pairs(nusc, rows, scene_kf_cache)
    if not pairs:
        print("No round-trip pairs found.")
        return

    index_rows = []
    for idx, (a, b, samples_a, samples_b) in enumerate(pairs):
        pair_dir = os.path.join(OUT_ROOT, f"pair_{idx:03d}")
        os.makedirs(pair_dir, exist_ok=True)
        out1 = os.path.join(pair_dir, "1")
        out2 = os.path.join(pair_dir, "2")
        poses1 = export_segment(nusc, a["scene"], samples_a, out1)
        poses2 = export_segment(nusc, b["scene"], samples_b, out2)
        overview = os.path.join(pair_dir, "overview.png")
        render_overview(poses1, poses2, a, b, overview)

        dur_a = samples_a[-1]["timestamp"] * 1e-6 - samples_a[0]["timestamp"] * 1e-6
        dur_b = samples_b[-1]["timestamp"] * 1e-6 - samples_b[0]["timestamp"] * 1e-6
        index_rows.append({
            "pair_idx": idx,
            "location": a["location"],
            "scene_a": a["scene"], "scene_b": b["scene"],
            "anchor_a_token": a["token"], "anchor_b_token": b["token"],
            "anchor_dist_m": f"{np.hypot(a['x']-b['x'], a['y']-b['y']):.2f}",
            "anchor_yaw_a_deg": f"{np.degrees(a['yaw']):.2f}",
            "anchor_yaw_b_deg": f"{np.degrees(b['yaw']):.2f}",
            "anchor_x": f"{a['x']:.2f}", "anchor_y": f"{a['y']:.2f}",
            "kf_a": len(samples_a), "kf_b": len(samples_b),
            "dur_a_s": f"{dur_a:.2f}", "dur_b_s": f"{dur_b:.2f}",
            "dir": os.path.relpath(pair_dir, OUT_ROOT),
        })
        print(f"  pair_{idx:03d} -> {pair_dir}")

    index_path = os.path.join(OUT_ROOT, "index.csv")
    with open(index_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(index_rows[0].keys()))
        w.writeheader()
        w.writerows(index_rows)
    print(f"\nIndex -> {index_path}")
    print(f"Total round-trip pairs exported: {len(pairs)}")


if __name__ == "__main__":
    main()
