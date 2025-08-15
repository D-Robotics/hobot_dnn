# Copyright (c) 2024，D-Robotics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch.substitutions import TextSubstitution
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import LoadComposableNodes
from ament_index_python.packages import get_package_share_directory, get_package_prefix

from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.descriptions import ComposableNode, ParameterFile
from launch.actions import GroupAction

from ament_index_python import get_package_share_directory
from ament_index_python.packages import get_package_prefix


def generate_launch_description():
    # 拷贝config中文件
    dnn_node_example_path = os.path.join(
        get_package_prefix('dnn_node_example'),
        "lib/dnn_node_example")
    print("dnn_node_example_path is ", dnn_node_example_path)
    cp_cmd = "cp -r " + dnn_node_example_path + "/config ."
    print("cp_cmd is ", cp_cmd)
    os.system(cp_cmd)

    # args that can be set from the command line or a default will be used
    config_file_launch_arg = DeclareLaunchArgument(
        "dnn_example_config_file", default_value=TextSubstitution(text="config/fcosworkconfig.json")
    )
    dump_render_launch_arg = DeclareLaunchArgument(
        "dnn_example_dump_render_img", default_value=TextSubstitution(text="0")
    )
    image_width_launch_arg = DeclareLaunchArgument(
        "dnn_example_image_width", default_value=TextSubstitution(text="960")
    )
    image_height_launch_arg = DeclareLaunchArgument(
        "dnn_example_image_height", default_value=TextSubstitution(text="544")
    )
    msg_pub_topic_name_launch_arg = DeclareLaunchArgument(
        "dnn_example_msg_pub_topic_name", default_value=TextSubstitution(text="hobot_dnn_detection")
    )
    declare_container_name_cmd = DeclareLaunchArgument(
        'container_name', 
        default_value='tros_container',
        description='the name of the container that launches all nodes')
    declare_log_level_cmd = DeclareLaunchArgument(
        'log_level', 
        default_value='warn',
        description='log level')
    # mipi cam图片发布pkg
    mipi_cam_device_arg = DeclareLaunchArgument(
        'device',
        default_value='F37',
        description='mipi camera device')
        
    load_composable_nodes = LoadComposableNodes(
        target_container=LaunchConfiguration('container_name'),
        composable_node_descriptions=[
            ComposableNode(
                package='mipi_cam',
                plugin='mipi_cam::MipiCamNode',
                name='mipi_cam_node',
                parameters=[
                    {"out_format": 'nv12'},
                    {"image_width": LaunchConfiguration('dnn_example_image_width')},
                    {"image_height": LaunchConfiguration('dnn_example_image_height')},
                    {"io_method": 'ros'},
                    {"video_device": LaunchConfiguration('device')},
                    {"frame_ts_type": 'realtime'}
                ],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
            
            # perception node
            ComposableNode(
                package='dnn_node_example',
                plugin='DnnExampleNode',
                name='perc_node',
                parameters=[
                        {'feed_type': 1},
                        {'config_file': LaunchConfiguration('dnn_example_config_file')},
                        {'ros_img_topic_name': '/image_raw'},
                        {'msg_pub_topic_name': LaunchConfiguration(
                            "dnn_example_msg_pub_topic_name")},
                        {'dump_render_img': LaunchConfiguration(
                            'dnn_example_dump_render_img')}
                ],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),

            # jpeg图片编码&发布pkg
            ComposableNode(
                package='hobot_codec',
                plugin='HobotCodecNode',
                name='image_jpeg_encoder_node',
                parameters=[
                    {'sub_topic': '/image_raw'},
                    {'in_format': 'nv12'},
                    {'in_mode': 'ros'},
                    {'out_mode': 'ros'},
                    {'pub_topic': '/image_jpeg'},
                    {'out_format': 'jpeg'}
                ],
                extra_arguments=[{'use_intra_process_comms': True}],
            )
        ]
    )

    # web展示pkg
    web_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('websocket'),
                'launch/websocket.launch.py')),
        launch_arguments={
            'websocket_image_topic': '/image_jpeg',
            'websocket_image_type': 'mjpeg',
            'websocket_smart_topic': LaunchConfiguration("dnn_example_msg_pub_topic_name")
        }.items()
    )

    # component container
    bringup_cmd_group = GroupAction([
        Node(
            name='tros_container',
            package='rclcpp_components',
            executable='component_container_isolated',
            exec_name='tros_container',
            parameters=[
                {'autostart': 'True'}],
            arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
            # prefix=['xterm -e gdb -ex run --args'],
            output='screen')
    ])

    return LaunchDescription([
        mipi_cam_device_arg,
        config_file_launch_arg,
        dump_render_launch_arg,
        image_width_launch_arg,
        image_height_launch_arg,
        msg_pub_topic_name_launch_arg,
        declare_container_name_cmd,
        declare_log_level_cmd, 
        bringup_cmd_group,
        load_composable_nodes,
        # 启动web展示pkg
        web_node
    ])
