"""
Dataset loader – adapted from street_gaussians.

Only the KITTI-360 reader is registered.
"""

import os
import random
import json
from lib.utils.camera_utils import camera_to_JSON, cameraList_from_camInfos
from lib.config import cfg
from lib.datasets.base_readers import storePly, SceneInfo
from lib.datasets.kitti360_readers import readKITTI360SceneInfo

sceneLoadTypeCallbacks = {
    "KITTI360": readKITTI360SceneInfo,
}


class Dataset():
    def __init__(self):
        self.cfg = cfg.data
        self.model_path = cfg.model_path
        self.source_path = cfg.source_path
        self.images = self.cfg.images

        self.train_cameras = {}
        self.test_cameras = {}

        dataset_type = cfg.data.get('type', 'KITTI360')
        assert dataset_type in sceneLoadTypeCallbacks.keys(), \
            f'Could not recognize scene type "{dataset_type}"! Available: {list(sceneLoadTypeCallbacks.keys())}'

        scene_info: SceneInfo = sceneLoadTypeCallbacks[dataset_type](self.source_path, **cfg.data)

        if cfg.mode == 'train':
            print(f'Saving input pointcloud to {os.path.join(self.model_path, "input.ply")}')
            pcd = scene_info.point_cloud
            storePly(os.path.join(self.model_path, "input.ply"), pcd.points, pcd.colors)

            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))

            print(f'Saving input camera to {os.path.join(self.model_path, "cameras.json")}')
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        self.scene_info = scene_info

        if self.cfg.shuffle and cfg.mode == 'train':
            random.shuffle(self.scene_info.train_cameras)
            random.shuffle(self.scene_info.test_cameras)

        for resolution_scale in cfg.resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.train_cameras, resolution_scale)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.test_cameras, resolution_scale)
