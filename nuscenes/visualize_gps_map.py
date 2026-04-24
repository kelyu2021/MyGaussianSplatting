"""Visualize all ego-vehicle trajectories from the NuScenes dataset overlaid on
each map location.

NuScenes does not store raw GPS lat/lon in the main DB (the CAN bus expansion does),
but every `ego_pose` is given in the map (world) frame for one of four maps:
boston-seaport, singapore-onenorth, singapore-hollandvillage, singapore-queenstown.

This script:
  1. groups all scenes by their map location,
  2. collects every ego_pose along each scene (using the LIDAR_TOP sample_data chain),
  3. renders one figure per location (trajectories overlaid on the drivable area),
  4. renders one combined 2x2 figure with all four locations.
"""

import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes

DATAROOT = "/Users/kevin/Data/nuscenes-v1.0-trainval"
VERSION = "v1.0-trainval"
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def collect_trajectories(nusc):
    log_to_loc = {log["token"]: log["location"] for log in nusc.log}
    trajs_per_loc = defaultdict(list)

    for scene in nusc.scene:
        loc = log_to_loc[scene["log_token"]]

        # Find the first LIDAR_TOP sample_data of the scene.
        sample = nusc.get("sample", scene["first_sample_token"])
        sd_token = sample["data"]["LIDAR_TOP"]
        while True:
            sd = nusc.get("sample_data", sd_token)
            if sd["prev"] == "":
                break
            sd_token = sd["prev"]

        xs, ys = [], []
        while sd_token:
            sd = nusc.get("sample_data", sd_token)
            ego = nusc.get("ego_pose", sd["ego_pose_token"])
            x, y, _ = ego["translation"]
            xs.append(x)
            ys.append(y)
            sd_token = sd["next"]
        trajs_per_loc[loc].append(np.column_stack([xs, ys]))

    return trajs_per_loc


def render_location(nusc, location, trajs, out_path):
    """Render trajectories with the drivable-area mask as a raster background.

    We avoid `NuScenesMap.render_layers` (which uses the broken `descartes`
    package on modern Shapely) and instead rasterize the drivable area via
    `get_map_mask` over the bounding box of the trajectories.
    """
    nusc_map = NuScenesMap(dataroot=nusc.dataroot, map_name=location)

    all_xy = np.concatenate(trajs, axis=0)
    x_min, y_min = all_xy.min(axis=0) - 50
    x_max, y_max = all_xy.max(axis=0) + 50
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    # Rasterize the drivable area for the patch covering the trajectories.
    # canvas_size is (rows=height, cols=width) at 1 px/m.
    canvas = (int(height), int(width))
    mask = nusc_map.get_map_mask(
        patch_box=(cx, cy, height, width),
        patch_angle=0,
        layer_names=["drivable_area"],
        canvas_size=canvas,
    )[0]

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(mask, origin="lower", cmap="gray_r", alpha=0.35,
              extent=[x_min, x_max, y_min, y_max])
    for traj in trajs:
        ax.plot(traj[:, 0], traj[:, 1], linewidth=0.6, alpha=0.7)
    ax.set_aspect("equal")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{location} -- {len(trajs)} scenes "
                 f"({sum(len(t) for t in trajs):,} ego poses)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_combined(trajs_per_loc, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    for ax, (loc, trajs) in zip(axes.flat, trajs_per_loc.items()):
        for traj in trajs:
            ax.plot(traj[:, 0], traj[:, 1], linewidth=0.5, alpha=0.6)
        ax.set_aspect("equal")
        ax.set_title(f"{loc}  ({len(trajs)} scenes)")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    fig.suptitle("NuScenes ego-vehicle trajectories per map", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)

    print("\nCollecting ego_pose trajectories for all scenes ...")
    trajs_per_loc = collect_trajectories(nusc)

    print("\nScenes per location:")
    for loc, trajs in trajs_per_loc.items():
        print(f"  {loc:30s} {len(trajs):4d} scenes  "
              f"{sum(len(t) for t in trajs):>9,} ego poses")

    for loc, trajs in trajs_per_loc.items():
        out = os.path.join(OUT_DIR, f"trajectories_{loc}.png")
        render_location(nusc, loc, trajs, out)
        print(f"Saved {out}")

    out = os.path.join(OUT_DIR, "trajectories_all.png")
    render_combined(trajs_per_loc, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
