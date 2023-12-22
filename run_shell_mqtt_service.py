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
Demonstrates how the shell module can be wrapped as an mqtt service which is compatible
with GAIC.
"""

import argparse
import srl.core
from srl.service.runners import (
    MetricsConfiguration,
    run_node_as_mqtt_service,
    ServiceType,
)
from shell_node import ShellNode


def main():

    # fmt: off
    # parse the arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-a", "--broker_address", type=str, help="ip address or domain name of MQTT broker")
    arg_parser.add_argument("-p", "--broker_port", type=int, help="port number of MQTT broker")
    arg_parser.add_argument("-t", "--shell_reconstruction_reqrep_topic_name", type=str, help="the topic name to use for shell reconstruction request response")
    arg_parser.add_argument("-q", "--qos", type=int, choices=[0, 1, 2], default=0, help="the qos setting")
    arg_parser.add_argument("-m", "--metrics_topic", type=str, default="metrics/general", help="the topic name to publish metrics to")
    args = arg_parser.parse_args()
    # fmt: on


    # hey, find your own mqtt host for metrics! saicny uses this one:
    metrics_config = MetricsConfiguration(
        host=args.broker_address,
        port=args.broker_port,
        topic=args.metrics_topic,
    )

    shell_node = ShellNode()
    run_node_as_mqtt_service(
        shell_node,
        service_type=ServiceType.ReqRep,
        broker_address=args.broker_address,
        broker_port=args.broker_port,
        incoming_topic_name=args.shell_reconstruction_reqrep_topic_name,
        qos=args.qos,
        debug=False,
        metrics_configuration=metrics_config,
    )


if __name__ == "__main__":
    main()
