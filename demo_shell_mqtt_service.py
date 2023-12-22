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
import argparse
from pathlib import Path
from shutil import rmtree
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d
from srl.core.messages import DynoFlex
from srl.mqtt import MQTTContext
from srl.service import MQTTRPCRequester
from demo import get_demo_data


def main():
    # parse the arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-a",
        "--broker_address",
        type=str,
        help="ip address or domain name of MQTT broker",
    )
    arg_parser.add_argument(
        "-p", "--broker_port", type=int, help="port number of MQTT broker"
    )
    arg_parser.add_argument(
        "-t",
        "--shell_reconstruction_reqres_topic_name",
        type=str,
        help="the topic name to use for shell reconstruction request response",
    )
    arg_parser.add_argument(
        "-q", "--qos", type=int, choices=[0, 1, 2], default=0, help="the qos setting"
    )
    args = arg_parser.parse_args()

    # create directory for saving results
    data_dir = Path("demo_data")
    results_dir = data_dir / "reconstruction_gaic_mqtt"
    if results_dir.exists():
        rmtree(results_dir)
    results_dir.mkdir()

    # instantiate the mqtt requester
    mqtt_context = MQTTContext()
    requester = MQTTRPCRequester(
        context=mqtt_context,
        broker_address=args.broker_address,
        broker_port=args.broker_port,
        topic_name=args.shell_reconstruction_reqres_topic_name,
        debug=False,
    )

    # get demo data
    data_dict = get_demo_data(data_dir)
    depth = np.asarray(data_dict["depth"]).astype(np.float64)
    camera_k = np.asarray(data_dict["camera_k"])
    mqtt_context = MQTTContext()
    for object_index_in_mask in data_dict["object_indexs_in_mask"]:
        masked_depth = depth * (data_dict["mask"] == object_index_in_mask)
        masked_depth = masked_depth.astype(np.float64)
        mesh_path = results_dir / Path(f"reconstruct_real_{object_index_in_mask}.ply")
        plt.imsave(
            results_dir / f"input_masked_depth_{mesh_path.stem}.png",
            masked_depth,
            cmap="jet",
        )
        data_to_send = {
            "masked_depth": masked_depth,
            "camera_k": camera_k,
            "use_mesh": False,
        }
        message_to_send = DynoFlex.from_data(data_to_send)
        received_data = DynoFlex.from_msg(requester.make_request(message_to_send))
        np_pcd = received_data["np_pcd"]
        # Save ply files for visualization
        o3dpcd = o3d.geometry.PointCloud()
        o3dpcd.points = o3d.utility.Vector3dVector(np.copy(np_pcd))
        o3d.io.write_point_cloud(str(mesh_path), o3dpcd)
        print(f"Shell reconstruction saved in the file {mesh_path}.")


if __name__ == "__main__":
    main()
