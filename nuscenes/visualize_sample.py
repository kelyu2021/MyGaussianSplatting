"""Render a single NuScenes keyframe in a 3-row layout:

  row 1: [fused RADAR BEV (all 5 radars)]   [LIDAR_TOP BEV]
  row 2: CAM_FRONT_LEFT  CAM_FRONT  CAM_FRONT_RIGHT
  row 3: CAM_BACK_LEFT   CAM_BACK   CAM_BACK_RIGHT

Both BEV panels are rendered in the ego (vehicle) frame, with the same range
in meters, so they are directly comparable.

Usage:
    python visualize_sample.py                          # first scene, first sample
    python visualize_sample.py --scene-index 3 --sample-index 5
    python visualize_sample.py --bev-range 60           # +/- 60 m around ego
"""

import argparse
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from pyquaternion import Quaternion

DATAROOT = "/Users/kevin/Data/nuscenes-v1.0-trainval"
VERSION = "v1.0-trainval"
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

RADAR_CHANNELS = [
    "RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT",
]
RADAR_COLORS = {
    "RADAR_FRONT":       "tab:red",
    "RADAR_FRONT_LEFT":  "tab:orange",
    "RADAR_FRONT_RIGHT": "tab:green",
    "RADAR_BACK_LEFT":   "tab:blue",
    "RADAR_BACK_RIGHT":  "tab:purple",
}
FRONT_ROW = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
BACK_ROW  = ["CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT"]


def _sensor_to_ego(nusc, sd_token, pc):
    """Transform a point cloud from sensor frame into the ego frame at sd's timestamp."""
    sd = nusc.get("sample_data", sd_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    pc.rotate(Quaternion(cs["rotation"]).rotation_matrix)
    pc.translate(np.array(cs["translation"]))
    return pc


def load_radar_points_ego(nusc, sample):
    """Return dict[channel] -> Nx2 (x,y in ego frame) for all 5 radars."""
    out = {}
    # Disable aggressive filtering so we keep all returns.
    RadarPointCloud.disable_filters()
    for ch in RADAR_CHANNELS:
        sd_token = sample["data"][ch]
        sd = nusc.get("sample_data", sd_token)
        pc = RadarPointCloud.from_file(os.path.join(nusc.dataroot, sd["filename"]))
        _sensor_to_ego(nusc, sd_token, pc)
        out[ch] = pc.points[:2, :].T  # (N, 2)
    RadarPointCloud.default_filters()
    return out


def load_lidar_points_ego(nusc, sample):
    sd_token = sample["data"]["LIDAR_TOP"]
    sd = nusc.get("sample_data", sd_token)
    pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, sd["filename"]))
    _sensor_to_ego(nusc, sd_token, pc)
    return pc.points[:3, :].T  # (N, 3) -> x, y, z


def draw_ego_marker(ax):
    # Draw the ego vehicle as a small rectangle pointing +x.
    ax.add_patch(plt.Rectangle((-2.0, -0.9), 4.0, 1.8, fill=False,
                               edgecolor="black", linewidth=1.5))
    ax.plot(0, 0, marker="^", color="black", markersize=6)


def _radar_points_in_lidar_frame(nusc, sample):
    """Return dict[channel] -> Nx2 (x,y) of radar points in the LIDAR_TOP frame.

    Transform chain per radar: radar sensor frame -> ego frame (via radar's
    calibrated_sensor) -> LIDAR_TOP sensor frame (inverse of lidar's
    calibrated_sensor). Ego pose is shared across sample_data of one keyframe
    sample, so we ignore inter-sensor ego motion (sub-millisecond at keyframe).
    """
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    lidar_R = Quaternion(lidar_cs["rotation"]).rotation_matrix
    lidar_t = np.array(lidar_cs["translation"])

    out = {}
    RadarPointCloud.disable_filters()
    for ch in RADAR_CHANNELS:
        sd_token = sample["data"][ch]
        sd = nusc.get("sample_data", sd_token)
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        pc = RadarPointCloud.from_file(os.path.join(nusc.dataroot, sd["filename"]))
        # radar sensor -> ego
        pc.rotate(Quaternion(cs["rotation"]).rotation_matrix)
        pc.translate(np.array(cs["translation"]))
        # ego -> lidar sensor
        pc.translate(-lidar_t)
        pc.rotate(lidar_R.T)
        out[ch] = pc.points[:2, :].T
    RadarPointCloud.default_filters()
    return out


def render(nusc, sample, bev_range, out_path):
    fig = plt.figure(figsize=(16, 16))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.6, 1.0, 1.0],
                          hspace=0.18, wspace=0.06)

    lidar_token = sample["data"]["LIDAR_TOP"]

    # ---- Row 1, panel A: fused radars on top of the LIDAR_TOP BEV (for context) ----
    ax_radar = fig.add_subplot(gs[0, 0:3])
    nusc.render_sample_data(lidar_token, with_anns=True, ax=ax_radar,
                            axes_limit=bev_range, verbose=False)
    radar_pts = _radar_points_in_lidar_frame(nusc, sample)
    for ch, xy in radar_pts.items():
        ax_radar.scatter(xy[:, 0], xy[:, 1], s=28, c=RADAR_COLORS[ch],
                         label=ch, alpha=0.95, edgecolors="white", linewidths=0.5)
    ax_radar.set_title("Fused RADARs (overlaid on LIDAR_TOP BEV + 3D boxes)")
    ax_radar.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # ---- Row 1, panel B: LiDAR_TOP BEV (devkit style, with 3D boxes) ----
    ax_lidar = fig.add_subplot(gs[0, 3:6])
    nusc.render_sample_data(lidar_token, with_anns=True, ax=ax_lidar,
                            axes_limit=bev_range, verbose=False)
    ax_lidar.set_title("LIDAR_TOP (devkit BEV with 3D boxes)")

    # ---- Rows 2 & 3: cameras ----
    for row_idx, channels in [(1, FRONT_ROW), (2, BACK_ROW)]:
        for col, ch in enumerate(channels):
            ax = fig.add_subplot(gs[row_idx, col * 2:(col + 1) * 2])
            sd = nusc.get("sample_data", sample["data"][ch])
            img = mpimg.imread(os.path.join(nusc.dataroot, sd["filename"]))
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(ch, fontsize=10)

    fig.suptitle(f"NuScenes keyframe -- sample {sample['token'][:10]}...",
                 fontsize=13)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--bev-range", type=float, default=50.0,
                        help="Half-side of the BEV window in meters (+/- value).")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    sub_dir = os.path.join(OUT_DIR, "visualize_sample")
    os.makedirs(sub_dir, exist_ok=True)
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)

    scene = nusc.scene[args.scene_index]
    sample_token = scene["first_sample_token"]
    for _ in range(args.sample_index):
        nxt = nusc.get("sample", sample_token)["next"]
        if not nxt:
            break
        sample_token = nxt
    sample = nusc.get("sample", sample_token)

    out = os.path.join(
        sub_dir,
        f"layout_{scene['name']}_s{args.sample_index:02d}.png",
    )
    render(nusc, sample, args.bev_range, out)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
