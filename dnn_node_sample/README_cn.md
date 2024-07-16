[English](./README.md) | 简体中文

Getting Started with Dnn Node Sample
=======


# 功能介绍

Dnn Node sample package是Dnn Node package的使用示例，通过继承DnnNode虚基类，使用YOLOv5模型和图像数据利用BPU处理器进行算法推理。

图像数据来源于订阅到的图像数据消息，支持使用MIPI/USB摄像头和本地图片发布的图像数据；推理完成后，使用自定义的算法输出解析方法解析算法输出的tensor；解析完成后发布智能结果，可通过web查看实时的渲染效果。

# 开发环境

- 编程语言: C/C++
- 开发平台: X3/X86
- 系统版本：Ubuntu 20.04/Ubuntu 22.04
- 编译工具链:Linux GCC 9.3.0/Linaro GCC 11.4.0

# 编译

- X3版本：支持在X3 Ubuntu系统上编译和在PC上使用docker交叉编译两种方式。
- X86版本：支持在X86 Ubuntu系统上编译一种方式。
同时支持通过编译选项控制编译pkg的依赖和pkg的功能。

## X3 Ubuntu系统上编译 X3版本

1、编译环境确认

- 板端已安装X3 Ubuntu系统。

- 当前编译终端已设置TROS环境变量：`source /opt/tros/setup.bash`。

- 已安装ROS2软件包构建系统ament_cmake。安装命令：`apt update; apt-get install python3-catkin-pkg; pip3 install empy`

- 已安装ROS2编译工具colcon。安装命令：`pip3 install -U colcon-common-extensions`

2、编译

- 编译命令：`colcon build --packages-select dnn_node_sample`

## docker交叉编译 X3版本

1、编译环境确认

- 在docker中编译，并且docker中已经编译好TROS。docker安装、交叉编译、TROS编译和部署说明详见[地平线机器人平台用户手册](https://developer.horizon.ai/api/v1/fileData/TogetherROS/quick_start/cross_compile.html#togetherros)。

2、编译

- 编译命令：

  ```shell
  export TARGET_ARCH=aarch64
  export TARGET_TRIPLE=aarch64-linux-gnu
  export CROSS_COMPILE=/usr/bin/$TARGET_TRIPLE-

  colcon build --packages-select dnn_node_sample \
     --merge-install \
     --cmake-force-configure \
     --cmake-args \
     --no-warn-unused-cli \
     -DCMAKE_TOOLCHAIN_FILE=`pwd`/robot_dev_config/aarch64_toolchainfile.cmake
  ````

# 使用介绍

## X3 Ubuntu系统上运行

包括图像消息发布和WEB展示。

**使用F37 MIPI摄像头发布图片**

```shell
# 配置TogetherROS环境
source /opt/tros/setup.bash

# 复制模型和回灌图片到运行目录
cp -r /opt/tros/${TROS_DISTRO}/lib/dnn_node_sample/config/ .

# 配置MIPI摄像头
export CAM_TYPE=mipi

ros2 launch dnn_node_sample dnn_node_sample.launch.py 
```

**使用USB摄像头发布图片**

```shell
# 配置TogetherROS环境
source /opt/tros/setup.bash

# 复制模型和回灌图片到运行目录
cp -r /opt/tros/${TROS_DISTRO}/lib/dnn_node_sample/config/ .

# 配置USB摄像头
export CAM_TYPE=usb

ros2 launch dnn_node_sample dnn_node_sample.launch.py 
```

**使用本地图片回灌**

```shell
# 配置TogetherROS环境
source /opt/tros/setup.bash

# 复制模型和回灌图片到运行目录
cp -r /opt/tros/${TROS_DISTRO}/lib/dnn_node_sample/config/ .

# 配置本地图片回灌
export CAM_TYPE=fb

# 使用的本地图片为/opt/tros/${TROS_DISTRO}/lib/dnn_node_sample/config/target.jpg
ros2 launch dnn_node_sample dnn_node_sample.launch.py 
```

## 注意事项

1. 切换MIPI摄像头类型

默认使用的MIPI摄像头类型为`F37`，可以修改launch启动脚本中`mipi_node`中的`video_device`配置项修改摄像头类型，支持的配置项为`F37`和`GC4663`。

2. 修改回灌图片

默认使用的回灌图片为`/opt/tros/${TROS_DISTRO}/lib/dnn_node_sample/config/target.jpg`，可以修改launch启动脚本中`fb_node`中的`image_source`配置项修改回灌的图片。

# 结果分析

使用本地图片回灌举例说明。

## X3结果展示

运行命令：

```bash
root@ubuntu:~# source /opt/tros/setup.bash
root@ubuntu:~# export CAM_TYPE=fb
root@ubuntu:~# ros2 launch dnn_node_sample dnn_node_sample.launch.py
```

log输出：

```text
[INFO] [launch]: All log files can be found below /root/.ros/log/2022-09-20-12-47-57-043477-ubuntu-4390
[INFO] [launch]: Default logging verbosity is set to INFO
camera_type is  fb
using feedback
webserver has launch
[INFO] [hobot_image_pub-1]: process started with pid [4396]
[INFO] [hobot_codec_republish-2]: process started with pid [4398]
[INFO] [dnn_node_sample-3]: process started with pid [4400]
[INFO] [websocket-4]: process started with pid [4402]
[dnn_node_sample-3] [C][4400][09-20][12:47:58:604][configuration.cpp:49][EasyDNN]EasyDNN version: 0.4.11
[dnn_node_sample-3] [BPU_PLAT]BPU Platform Version(1.3.1)!
[dnn_node_sample-3] [HBRT] set log level as 0. version = 3.14.5
[dnn_node_sample-3] [DNN] Runtime version = 1.9.7_(3.14.5 HBRT)
[dnn_node_sample-3] [WARN] [1663649278.801689569] [dnn]: Run default SetOutputParser.
[dnn_node_sample-3] [WARN] [1663649278.802293300] [dnn]: Set output parser with default dnn node parser, you will get all output tensors and should parse output_tensors in PostProcess.
[dnn_node_sample-3] [WARN] [1663649280.115318280] [dnn_node_sample]: input fps: 11.49, out fps: 11.64, infer time ms: 65, post process time ms: 78
[dnn_node_sample-3] [WARN] [1663649281.120420769] [dnn_node_sample]: input fps: 11.49, out fps: 9.98, infer time ms: 67, post process time ms: 81
[dnn_node_sample-3] [WARN] [1663649282.217524812] [dnn_node_sample]: input fps: 10.00, out fps: 9.97, infer time ms: 71, post process time ms: 74
[dnn_node_sample-3] [WARN] [1663649283.318315410] [dnn_node_sample]: input fps: 10.00, out fps: 10.00, infer time ms: 71, post process time ms: 74
[dnn_node_sample-3] [WARN] [1663649284.320494664] [dnn_node_sample]: input fps: 10.00, out fps: 9.99, infer time ms: 72, post process time ms: 76
```

log显示订阅图像消息和发布AI消息的帧率都为10fps左右，算法单帧推理耗时为70毫秒左右，算法输出解析耗时为78毫秒左右。

## web效果展示

web效果截图：

![image](./render/webrender.jpg)

渲染出了检测出来的目标检测框和类别。
