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

import numpy as np
import time
from srl.core import Node
from shell.reconstructor import ShellReconstructor
from shell.utils.misc import get_masked_depth


class ShellNode(Node):
    """
    Wrap the shell-reconstruction api in an SRL Node. This extracts the input from the GAIC-protocol
    message, performs the shell reconstruction, and encodes the output into another GAIC-protocol
    message.
    """

    def __init__(self) -> None:
        """
        Description:
            Initializes ShellReconstructor
        """
        super().__init__("ShellNode")
        self.shell_reconstructor = ShellReconstructor()

    def receive(self, m) -> None:
        """
        Description:
            "receive"s input to shell-reconstructor, computes the reconstruction and "signal"s
            the reconstruction
        Input:
            @param: m <DynoFlex>
                DynoFlex message which should contain the following:
                - masked_depth <np.ndarray> (D>=480, D>=480): Height and width should be same (D).
                    D should be >= 480 pixels.
                - camera_k <np.ndarray> (3,3)
        Output:
            DynoFlex message is signalled containing a dictionary: {"np_pcd": <np.ndarray> (n, 3)}
        """
        masked_depth = data["masked_depth"]
        camera_k = data["camera_k"]
        use_mesh = True
        if "use_mesh" in data.keys():
            use_mesh = data["use_mesh"]
        np_pcd = self.shell_reconstructor.reconstruct(masked_depth, camera_k, use_mesh=use_mesh)
        result_msg = {"np_pcd": np_pcd}
        self.signal(result_msg)



class ShellMultiNode(Node):
    """
    Same as above, but performs shell reconstruction for all objects in the
    mask, and signals each reconstruction as a new message.
    """
    def __init__(self) -> None:
        super().__init__("ShellMultiNode")
        self.shell_reconstructor = ShellReconstructor()
        self.frame_index = -1

    def receive(self, m) -> None:
        """
        Expects: {
            "mask": numpy array, (H, W) mask image
            "depth": numpy array, (H, W) depth image
            "camera_k": numpy array, 3x3 camera intrinsics
        }
        For each object in the mask, if there is a non-zero depth image for that object:
        Signals: {
            "points": numpy arrawy, shell points of object
            "object_id": int, the object id for the signalled shell object
            "frame_index": int, increments for each incoming mask and depth image, can be
                used to determine what objects belong in a new frame client-side
                (since masks usually don't guarantee object tracking stability)
        }
        """
        self.frame_index += 1

        mask = m["mask"]
        depth = m["depth"]
        camera_k = m["camera_k"]
        #rgb = data["rgb"]

        use_mesh = True
        if "use_mesh" in m.keys():
            use_mesh = m["use_mesh"]

        mask_object_ids = np.unique(mask)
        for o in mask_object_ids:
            if o != 0: # elide object 0 (background, we assume)
                masked_depth = get_masked_depth(o, mask, depth)
                any_depth_in_mask = np.any(masked_depth != 0)
                if any_depth_in_mask:
                    np_pcd = self.shell_reconstructor.reconstruct(masked_depth, camera_k, use_mesh=use_mesh)
                    result_msg = {"points": np_pcd, "object_id": o, "frame_index": self.frame_index}
                    self.signal(result_msg)
