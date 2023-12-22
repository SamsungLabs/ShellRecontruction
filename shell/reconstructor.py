"""
Copyright (c) 2023 Samsung Electronics Co., Ltd.

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0/
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE file.
"""

"""
Contains functionalities to perform 3D shape reconstruction based on shell reconstruction technique.
Shell reconstruction do 3D shape reconstruction by predicting "backside" depth given the "front"
depth image and then stitching them together. Arxiv paper: https://arxiv.org/abs/2109.06837
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import torch
from scipy.ndimage import zoom
from shell.models.shell_reconstructor_model import ShellReconstructorModel
from shell.utils.checkpoints import CheckpointIO
from shell.utils import misc


class ShellReconstructor:
    """
    Given a masked depth image and camera_k, this class provide functionalities to do shape
    reconstruction based on shell reconstruction technique
    """

    def __init__(
        self,
        device: torch.device = None,
        model_ckpt_path: Path = Path(__file__).parent.parent
        / "models/shell_v2/model.pt",
    ) -> None:
        """
        Description:
            init shell reconstructor
        Input:
            @param: device <torch.device>
                what device (gpu / cpu) to use for reconstruction. If None, then gpu would be used
                if available, otherwise cpu
            @param: model_ckpt_path <Path>
                path to model checkpoint. default path points to provided trained model
        """
        # Input depth images are first cropped to size self.depth_crop_width around x-mean of
        #   non-zero depth pixels.
        self.depth_crop_width = 480
        # Cropped depth image is then scaled to size (self.network_input_width,
        #   self.network_input_width)
        self.network_input_width = 256

        self.model_ckpt_path = model_ckpt_path
        if device is None:
            device = misc.get_device()
        self.device = device
        # Load model from ckpt
        self.model = ShellReconstructorModel(device=self.device)
        checkpoint_io = CheckpointIO(
            self.model_ckpt_path.parent, model=self.model, optimizer=None
        )
        checkpoint_io.load(self.model_ckpt_path.name, device=self.device)
        self.model.eval()

    def _process_depth_image(
        self, depth: np.ndarray, camera_k: np.ndarray
    ) -> np.ndarray:
        """
        Description:
            Removes outliers from masked depth images by:
                Step-1) Project depth to point cloud
                Step-2) Remove outliers in the point cloud
                Step-3) Project the point cloud back to a depth image
        Input:
            @param: depth <np.ndarray> (H, W)
                depth image. All H, W > 0 are allowed
            @param: camera_k <np.ndarray> (3, 3)
                camera intrinsics
        Output:
            @return clean_depth_img <np.ndarray>
                outlier-free depth image (shape same as input depth)
        """
        o3dpcd_cam_obj = misc.o3d_pointcloud_native(depth, camera_k)
        pcd_cam_obj = np.asarray(o3dpcd_cam_obj.points)
        clean_depth_img = misc.pointcloud3d_to_depth_image(
            pcd_cam_obj.T, depth.shape[1], depth.shape[0], camera_k
        )
        return clean_depth_img

    def _prepare_network_input(
        self, depth: np.ndarray, camera_k: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Description:
            Prepares input for shell reconstruction network
                - Mask out the invalid depth values (<=0)
                - Create a crop of width ShellReconstructore.depth_crop_width around x-center of
                    masked_depth
                - Zoom the crop to the network_input_width
                - Adjust the camera intrinsics accordingly
        Input:
            @param: depth <np.ndarray> (H>=self.depth_crop_width, W>=self.depth_crop_width)
                depth image (Height and Width should be >= self.depth_crop_width)
            @param: camera_k <np.ndarray> (3, 3)
                camera intrinsics
        Output:
            @return Tuple[depth <np.ndarray>, camera_k <np.ndarray>]
                depth image <np.ndarray> (self.network_input_width, self.network_input_width)
                    ready to be processed by network
                camera_k <np.ndarray> (3, 3)
                    adjusted according to changes in input depth
        """

        # Mask out the depth image
        masked_depth = np.copy(depth)
        masked_depth[depth <= 0] = 0

        # select all pixels of idx and compute mean x-value
        mean_x = np.mean(np.where(masked_depth > 0)[1]).astype(np.int)
        # adjust mean x-value based on boundaries
        depth_crop_hw = np.ceil(self.depth_crop_width / 2).astype(np.int)
        mean_x = np.clip(mean_x, depth_crop_hw, masked_depth.shape[1] - depth_crop_hw)
        # crop a 480 x 480 window of the depth image
        crop_depth = masked_depth[:, mean_x - depth_crop_hw : mean_x + depth_crop_hw]

        # downsample the crop
        factor = self.network_input_width / self.depth_crop_width
        # scipy.ndimage.zoom ignores the values which are nan. We want the zero values to remain
        #   at zero. Hence we first set them to nan and then replace with zero after zoom
        crop_depth[crop_depth == 0] = np.nan
        final_depth = zoom(crop_depth, [factor, factor], order=1)
        final_depth = np.nan_to_num(final_depth)
        final_depth = (
            final_depth / 1000
        )  # convert to meter  # XXX: Handle this properly

        # change camera matrix
        final_camera_k = camera_k.copy()
        # Move center-x point by downsampled distance
        final_camera_k[0, 2] = depth_crop_hw - (mean_x - final_camera_k[0, 2])
        # rescale intrinsics to downsampled ratio
        final_camera_k[:2, :] *= factor

        final_depth = final_depth.astype(np.float32)
        final_camera_k = final_camera_k.astype(np.float32)
        final_depth = self._process_depth_image(final_depth, final_camera_k)
        return final_depth, final_camera_k

    def _generate_pcl(
        self, depth: np.ndarray, camera_k: np.ndarray, number_of_points: int, use_mesh: bool
    ) -> np.ndarray:
        """
        Description:
            Generate shell reconstruction mesh by forwarding the input to the model
        Input:
            @param depth_image <np.ndarray> (self.network_input_width, self.network_input_width)
                assumes depth to be outlier-free and of shape self.network_input_width
                (use self._prepare_network_input)
            @param camera_k <np.ndarray>
                3x3 camera intrinsics matrix
            @param number_of_points <int>
                number of points to sample from the shell reconstruction mesh for the return point-
                    cloud
        Output:
            @return pred_pcl <np.ndarray> (number_of_points, 3)
                predicted full pointcloud of the shell reconstruction

        """

        depth = depth.squeeze()
        if len(depth.shape) == 3:
            depth = depth[0][np.newaxis, ...]
        inputs = torch.tensor(depth, device=self.device)
        mask = depth != 0
        inputs = torch.tensor(
            np.stack([depth, mask, np.zeros_like(depth)]),
            device=self.device,
            dtype=torch.float,
        )

        while len(inputs.shape) < 4:
            inputs = inputs.unsqueeze(0)

        # Generators assume negative depth. Everything has to be negated in case depth is positive
        flip_out = False
        if inputs[0].min() >= 0:
            inputs[:, 0] *= -1

            flip_out = True
            inputs = inputs.flip(-2, -1)
            camera_k[0, 2] = inputs.shape[-1] - camera_k[0, 2]
            camera_k[1, 2] = inputs.shape[-1] - camera_k[1, 2]

        input_sample = {"inputs": inputs, "camera_k": camera_k, "mode": "persp"}
        
        # WAT
        #self.model.eval()

        pred_shell = self.model.generate_shell(input_sample)
        if use_mesh:
            pred_mesh = self.model.data_out_to_mesh(pred_shell, camera_k)
            pred_pcl = pred_mesh.sample_points_uniformly(number_of_points=number_of_points)
            pred_pcl = np.asarray(pred_pcl.points)
        else:
            pred_pcl = self.model.generate_point_cloud_from_shell(pred_shell, camera_k)

        if flip_out:
            pred_pcl[:, -1] *= -1

        return pred_pcl

    def reconstruct(
        self, depth_image: np.ndarray, camera_k: np.ndarray, number_points: int = 3000, use_mesh: bool = True,
    ) -> np.ndarray:
        """
        Description:
            Compute shell reconstruction
        Input:
            @param: depth <np.ndarray> (D>=self.depth_crop_width, D>=self.depth_crop_width)
                masked depth image of the target object for which shell reconstruction is required.
                Height and Width should be equal (D).
                D should be >= self.depth_crop_width (see self._prepare_network_input)
            @param camera_k <np.ndarray>
                3x3 camera intrinsics matrix
            @param number_poitns <int>
                number of points to sample from the shell reconstruction mesh for the return point-
                    cloud
        Output:
            @return pred_pcl <np.ndarray> (number_points, 3)
                predicted full pointcloud of the shell reconstruction
        """
        depth_processed, camera_k_processed = self._prepare_network_input(
            depth_image, camera_k
        )
        pred_pcl = self._generate_pcl(depth_processed, camera_k_processed, number_points, use_mesh)
        return pred_pcl
