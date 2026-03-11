"""
GoPro 360 Dataset – drop-in replacement for ``Gaussians/lib/datasets/dataset.py``.

Provides the same interface (``scene_info``, ``train_cameras``, ``test_cameras``)
used by :class:`~lib.models.scene.Scene`, but reads GoPro 360 COLMAP data
instead of KITTI-360.
"""

from __future__ import annotations

import os
import json
import random

from lib.config import cfg
from lib.datasets.base_readers import SceneInfo, storePly
from lib.utils.camera_utils import camera_to_JSON, cameraList_from_camInfos

from lib.datasets.gopro360_readers import readGoPro360SceneInfo


class GoPro360Dataset:
    """Dataset class that reads GoPro 360° COLMAP data.

    Intended as a drop-in replacement for the Gaussians ``Dataset`` class.
    """

    def __init__(self):
        self.model_path  = cfg.model_path
        self.source_path = cfg.source_path

        self.train_cameras = {}
        self.test_cameras  = {}

        # ── Read scene using our gopro360 reader ─────────────────────
        scene_info: SceneInfo = readGoPro360SceneInfo(
            source_path=self.source_path,
            point_cloud_path=cfg.data.get("point_cloud_path", ""),
            split_test=int(cfg.data.get("split_test", 8)),
            mode=cfg.mode,
            workspace=cfg.workspace,
        )

        # ── Save input PLY & cameras.json (mirrors Dataset behaviour) ─
        if cfg.mode == "train":
            os.makedirs(self.model_path, exist_ok=True)
            pcd = scene_info.point_cloud
            input_ply = os.path.join(self.model_path, "input.ply")
            print(f"Saving input pointcloud to {input_ply}")
            storePly(input_ply, pcd.points, pcd.colors)

            json_cams = []
            camlist = list(scene_info.test_cameras) + list(scene_info.train_cameras)
            for cid, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(cid, cam))
            cam_json = os.path.join(self.model_path, "cameras.json")
            print(f"Saving input cameras to {cam_json}")
            with open(cam_json, "w") as fp:
                json.dump(json_cams, fp)

        self.scene_info = scene_info

        # Shuffle training set
        if cfg.data.get("shuffle", True) and cfg.mode == "train":
            random.shuffle(self.scene_info.train_cameras)
            random.shuffle(self.scene_info.test_cameras)

        # Build Camera objects at each resolution scale
        for res_scale in cfg.resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[res_scale] = cameraList_from_camInfos(
                self.scene_info.train_cameras, res_scale
            )
            print("Loading Test Cameras")
            self.test_cameras[res_scale] = cameraList_from_camInfos(
                self.scene_info.test_cameras, res_scale
            )
