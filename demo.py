"""
Copyright (c) 2023 Samsung Electronics Co., Ltd.

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0/
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE file.

Script to run shell on demo data
"""

from pathlib import Path
from shutil import rmtree
import pickle
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d
from shell.reconstructor import ShellReconstructor


def get_demo_data(data_dir: Path):
    # Read input data
    with open(data_dir / "real_data.pkl", "rb") as pkl_file:
        data_dict = pickle.load(pkl_file)
    return data_dict


if __name__ == "__main__":
    # Setup the results directory
    data_dir = Path("demo_data")
    results_dir = data_dir / "reconstruction"
    if results_dir.exists():
        rmtree(results_dir)
    results_dir.mkdir()

    data_dict = get_demo_data(data_dir)
    plt.imsave(results_dir / "observed_rgb.png", data_dict["rgb"])
    depth = np.asarray(data_dict["depth"]).astype(np.float64)
    camera_k = np.asarray(data_dict["camera_k"])
    for object_index_in_mask in data_dict["object_indexs_in_mask"]:
        masked_depth = depth * (data_dict["mask"] == object_index_in_mask)
        masked_depth = masked_depth.astype(np.float64)
        mesh_path = results_dir / Path(f"reconstruct_real_{object_index_in_mask}.ply")
        plt.imsave(
            results_dir / f"input_masked_depth_{mesh_path.stem}.png",
            masked_depth,
            cmap="jet",
        )

        # Perform shell reconstruction
        reconstructor = ShellReconstructor()
        np_pcd = reconstructor.reconstruct(masked_depth, camera_k)

        # Save ply files for visualization
        o3dpcd = o3d.geometry.PointCloud()
        o3dpcd.points = o3d.utility.Vector3dVector(np_pcd)
        o3d.io.write_point_cloud(str(mesh_path), o3dpcd)
        print(f"Shell reconstruction saved in the file {mesh_path}.")
