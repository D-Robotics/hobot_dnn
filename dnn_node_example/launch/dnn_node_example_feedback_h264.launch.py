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
    # 本地图片发布
    camera_device_arg = DeclareLaunchArgument(
        'publish_image_source', default_value='./config/test.h264', description='feedback h264')

    cam_node = Node(
        package='hobot_image_publisher',
        executable='hobot_image_pub',
        output='screen',
        parameters=[
            {"image_source": LaunchConfiguration('publish_image_source')},
            {"image_format": 'h264'},
            {"msg_pub_topic_name": "/hbmem_h264"},
            {"output_image_w": 0},
            {"output_image_h": 0},
            {"fps": 30},
            {"is_loop": True},
            {"is_shared_mem": True},
            {"is_compressed_img_pub": True},
        ],
        arguments=['--ros-args', '--log-level', 'warn']
    )

    # jpeg图片编码&发布pkg
    jpeg_codec_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('hobot_codec'),
                'launch/hobot_codec_encode.launch.py')),
        launch_arguments={
            'codec_in_mode': 'shared_mem',
            'codec_out_mode': 'ros',
            'codec_sub_topic': '/hbmem_img',
            'codec_pub_topic': '/image'
        }.items()
    )

    # nv12图片解码&发布pkg
    nv12_codec_node = Node(
        package='hobot_codec',
        executable='hobot_codec_republish',
        output='screen',
        parameters=[
            {"in_mode": "shared_mem"},
            {"in_format": 'h264'},
            {"out_mode": "shared_mem"},
            {"out_format": 'nv12'},
            {"sub_topic": "/hbmem_h264"},
            {"pub_topic": "/hbmem_img"},
            {"dump_output": False}
        ],
        arguments=['--ros-args', '--log-level', 'warn']
    )

    # web展示pkg
    web_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('websocket'),
                'launch/websocket.launch.py')),
        launch_arguments={
            'websocket_image_topic': '/image',
            'websocket_image_type': 'mjpeg',
            'websocket_smart_topic': LaunchConfiguration("dnn_example_msg_pub_topic_name")
        }.items()
    )

    # 算法pkg
    dnn_node_example_node = Node(
        package='dnn_node_example',
        executable='example',
        output='screen',
        parameters=[
            {"config_file": LaunchConfiguration('dnn_example_config_file')},
            {"dump_render_img": LaunchConfiguration(
                'dnn_example_dump_render_img')},
            {"feed_type": 1},
            {"is_shared_mem_sub": 1},
            {"msg_pub_topic_name": LaunchConfiguration(
                "dnn_example_msg_pub_topic_name")}
        ],
        arguments=['--ros-args', '--log-level', 'warn']
    )

    shared_mem_node = IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory('hobot_shm'),
                        'launch/hobot_shm.launch.py'))
            )

    return LaunchDescription([
        camera_device_arg,
        config_file_launch_arg,
        dump_render_launch_arg,
        image_width_launch_arg,
        image_height_launch_arg,
        msg_pub_topic_name_launch_arg,
        # 启动零拷贝环境配置node
        shared_mem_node,
        # 图片发布pkg
        cam_node,
        # 图片编解码&发布pkg
        nv12_codec_node,
        # 图片编解码&发布pkg
        jpeg_codec_node,
        # 启动example pkg
        dnn_node_example_node,
        # 启动web展示pkg
        web_node
    ])