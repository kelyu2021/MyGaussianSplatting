"""Visualize the NuScenes v1.0-trainval dataset.

Usage:
    python read_dataset.py                       # render the first scene's first sample
    python read_dataset.py --scene-index 3       # pick a different scene
    python read_dataset.py --render-scene        # render a full scene video (mp4)
"""

import argparse
import os

import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes

DATAROOT = "/Users/kevin/Data/nuscenes-v1.0-trainval"
VERSION = "v1.0-trainval"
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0,
                        help="Which sample (keyframe) within the scene to render.")
    parser.add_argument("--render-scene", action="store_true",
                        help="Render the full scene as an mp4 (slow).")
    parser.add_argument("--channel", default="CAM_FRONT",
                        help="Sensor channel for individual sample-data rendering.")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)

    print(f"\n#scenes={len(nusc.scene)}  #samples={len(nusc.sample)}  "
          f"#sample_data={len(nusc.sample_data)}  #annotations={len(nusc.sample_annotation)}")

    scene = nusc.scene[args.scene_index]
    print(f"\nScene[{args.scene_index}]: {scene['name']} -- {scene['description']}")

    # Walk to the requested keyframe sample.
    sample_token = scene["first_sample_token"]
    for _ in range(args.sample_index):
        nxt = nusc.get("sample", sample_token)["next"]
        if not nxt:
            break
        sample_token = nxt
    sample = nusc.get("sample", sample_token)

    # 1) Render all sensors + LiDAR top-down for this keyframe.
    sample_path = os.path.join(OUT_DIR, f"sample_{scene['name']}_{args.sample_index}.png")
    nusc.render_sample(sample_token, out_path=sample_path, verbose=False)
    plt.close("all")
    print(f"Saved keyframe overview -> {sample_path}")

    # 2) Render a single camera with projected 3D boxes.
    cam_token = sample["data"][args.channel]
    cam_path = os.path.join(OUT_DIR, f"sample_data_{scene['name']}_{args.channel}.png")
    nusc.render_sample_data(cam_token, out_path=cam_path, verbose=False)
    plt.close("all")
    print(f"Saved {args.channel} view  -> {cam_path}")

    # 3) Render the LiDAR top-down with annotations.
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_path = os.path.join(OUT_DIR, f"sample_data_{scene['name']}_LIDAR_TOP.png")
    nusc.render_sample_data(lidar_token, out_path=lidar_path, verbose=False)
    plt.close("all")
    print(f"Saved LIDAR_TOP view  -> {lidar_path}")

    # 4) Optional: render the whole scene as an mp4 video.
    if args.render_scene:
        video_path = os.path.join(OUT_DIR, f"scene_{scene['name']}.mp4")
        nusc.render_scene(scene["token"], out_path=video_path)
        print(f"Saved scene video    -> {video_path}")


if __name__ == "__main__":
    main()
