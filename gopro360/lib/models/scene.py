"""
Scene – thin wrapper that ties a Dataset to a GaussianModel.
"""

import os
import torch
from typing import Union
from lib.datasets.dataset import Dataset
from lib.models.gaussian_model import GaussianModel
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.config import cfg
from lib.utils.system_utils import searchForMaxCheckpoint


class Scene:
    gaussians: Union[GaussianModel, StreetGaussianModel]
    dataset: Dataset

    def __init__(
        self,
        gaussians: Union[GaussianModel, StreetGaussianModel],
        dataset: Dataset,
    ):
        self.dataset = dataset
        self.gaussians = gaussians

        if cfg.mode == 'train':
            point_cloud = self.dataset.scene_info.point_cloud
            scene_radius = self.dataset.scene_info.metadata['scene_radius']
            print("Creating gaussian model from point cloud")
            self.gaussians.create_from_pcd(point_cloud, scene_radius)

            if cfg.get('to_cuda', False):
                print('Moving training cameras to GPU')
                for camera in self.getTrainCameras():
                    camera.set_device('cuda')
        else:
            assert os.path.exists(cfg.point_cloud_dir)
            if cfg.loaded_iter == -1:
                self.loaded_prefix, self.loaded_iter = searchForMaxCheckpoint(
                    cfg.point_cloud_dir)
            else:
                self.loaded_iter = cfg.loaded_iter
                # Detect prefix from existing files
                epoch_path = os.path.join(
                    cfg.trained_model_dir,
                    f"epoch_{self.loaded_iter}.pth")
                self.loaded_prefix = (
                    'epoch' if os.path.exists(epoch_path) else 'iteration')

            print(f"Loading checkpoint at {self.loaded_prefix} {self.loaded_iter}")
            ckpt_path = os.path.join(
                cfg.trained_model_dir,
                f"{self.loaded_prefix}_{self.loaded_iter}.pth",
            )
            assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
            state_dict = torch.load(ckpt_path, weights_only=False)
            self.gaussians.load_state_dict(state_dict=state_dict)

    def save(self, num, prefix="iteration"):
        pc_path = os.path.join(
            cfg.point_cloud_dir,
            f"{prefix}_{num}",
            "point_cloud.ply",
        )
        self.gaussians.save_ply(pc_path)

    def getTrainCameras(self, scale=1):
        return self.dataset.train_cameras[scale]

    def getTestCameras(self, scale=1):
        return self.dataset.test_cameras[scale]
