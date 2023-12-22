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

import torch
import numpy as np
from shell.models.utils import UNet
from shell.utils import misc


class ShellReconstructorModel(torch.nn.Module):
    def __init__(
        self,
        device=None,
    ):
        super().__init__()

        self.height = 4
        self.max_fms = 256
        self.wf = 6
        self.xy_size = 1.2
        self.in_channels = 3
        self.exit_only = False
        self.mask_channel = 1
        self.depth_channel = 0
        self.device = device
        self.unet = UNet(
            in_channels=self.in_channels,
            out_channels=2,
            depth=self.height,
            wf=self.wf,
            normalization="gn",
            padding=True,
            fm_cap=self.max_fms,
            use_skip=True,
        ).to(self.device)

    def data_out_to_mesh(self, data_out, camera_k):
        shell = misc.get_np(data_out[0])
        mesh_out = misc.get_persp_shell_mesh(
            shell, self.xy_size / data_out.shape[-1], camera_k=camera_k
        )
        return mesh_out

    def forward_persp(self, data_in, zero_out=False):
        data_in = data_in.to(self.device)
        data_out = self.unet(data_in.to(self.device))

        if zero_out:
            assert data_in.shape[0] == 1
            if self.mask_channel is None:
                mask = misc.get_np(data_in[0, 0] == 0)
            else:
                mask = data_in[0, self.mask_channel] == 0
            mask = torch.tensor(mask, device=self.device) != 0
            data_out[0, 0][mask] = 0
            data_out[0, 1][mask] = 0

        if self.exit_only:
            data_out[:, 0] = 0
            data_out[0, 0][data_in[0, 0] != 0] = data_in[0, 0][data_in[0, 0] != 0]

        return data_out

    def generate_point_cloud_from_shell(self, data, camera_k):
        input_image = np.expand_dims(data[0,0,:,:].cpu().detach().numpy(), axis=2)
        output_image = np.expand_dims(data[0,1,:,:].cpu().detach().numpy(), axis=2)
        input_cloud = misc.depth_image_to_pointcloud3d(input_image, camera_k)
        output_cloud = misc.depth_image_to_pointcloud3d(output_image, camera_k)
        all_cloud = np.hstack([input_cloud, output_cloud])
        return all_cloud.T

    def generate_shell(self, data):
        """Generates the output mesh.

        Args:
            data (tensor): data tensor
        """
        # WAT
        #self.train()
        data_in = data.get("inputs")
        camera_k = data.get("camera_k")
        if self.in_channels == 1:
            data_in = data_in[:, 0:1]
        return self.forward_persp(data_in, zero_out=True)
 
    def generate_mesh(self, data):
        """Generates the output mesh.

        Args:
            data (tensor): data tensor
        """
        data_out = self.generate_shell(data)
        return self.data_out_to_mesh(data_out, camera_k)