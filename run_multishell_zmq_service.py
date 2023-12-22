"""
Copyright (c) 2023 Samsung Electronics Co., Ltd.

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0/
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE file.

Demonstrates how the shell module can be wrapped as a zmq sub-pub
service which is compatible with GAIC.
"""
import argparse
import srl.core
from srl.service.runners import (
    MetricsConfiguration,
    run_node_as_subpub_zmq_service,
)
from shell_node import ShellMultiNode


def main():

    # fmt: off
    # parse the arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-s", "--sub_address", type=str, help="address to subscribe to")
    arg_parser.add_argument("-p", "--pub_address", type=str, help="address to publish on")
    arg_parser.add_argument("-a", "--broker_address", type=str, help="ip address or domain name of MQTT broker, for metrics publishing only")
    arg_parser.add_argument("-b", "--broker_port", type=int, help="port number of MQTT broker, for metrics publishing only")
    arg_parser.add_argument("-m", "--metrics_topic", type=str, default="metrics/general", help="the topic name to publish metrics to")
    args = arg_parser.parse_args()
    # fmt: on

    # hey, find your own mqtt host for metrics! saicny uses this one:
    metrics_config = MetricsConfiguration(
        host=args.broker_address,
        port=args.broker_port,
        topic=args.metrics_topic,
    )

    # instantiate the node that does the actual work
    shell_node = ShellMultiNode()

    # run that node as a zmq-based sub-pub service.
    # this function will run forever
    run_node_as_subpub_zmq_service(
        shell_node,
        sub_address=args.sub_address,
        pub_address=args.pub_address,
        max_queued_msgs=1,
        metrics_configuration=metrics_config,
    )


if __name__ == "__main__":
    main()
