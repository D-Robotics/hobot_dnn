English| [简体中文](./README_cn.md)

# Function Introduction

Dnn Benchmark example package is a performance evaluation example of Dnn Node package. By inheriting the DnnNode virtual base class, it utilizes the BPU processor for model inference using models and image data, outputting performance indicators such as frame rate (fps) and single-frame delay (latency). Image data is fed back from local images.

# Development Environment

- Programming Language: C/C++
- Development Platform: X3/Rdkultra/X5/X86
- System Version: Ubuntu 20.04/Ubuntu 22.04
- Compilation Toolchain: Linux GCC 9.3.0/Linaro GCC 11.4.0

# Compilation

- X3 Version: Supports compilation on the X3 Ubuntu system and cross-compilation using Docker on the PC.
- Rdkultra Version: Supports compilation on the Rdkultra Ubuntu system and cross-compilation using Docker on the PC.
- X5 Version: Supports compilation on the X5 Ubuntu system and cross-compilation using Docker on the PC.
- X86 Version: Supports compilation on the X86 Ubuntu system.
Both support controlling the dependencies and functions of the compiled pkg through compilation options.

## Dependency Libraries

Dependency libraries:

- OpenCV: 3.4.5

ROS package:

- dnn node

## Compilation on X3 Ubuntu System - X3 Version

1. Confirmation of Compilation Environment

- X3 Ubuntu system is installed on the board.
- The current compilation terminal has set the TogetherROS environment variable: `source PATH/setup.bash`. Where PATH is the installation path of TogetherROS.
- ROS2 compilation tool colcon is installed. If the installed ROS does not include the compilation tool colcon, manual installation of colcon is required. Installation command for colcon: `pip install -U colcon-common-extensions`
- dnn node package has been compiled.

2. Compilation

- Compilation command: `colcon build --packages-select dnn_benchmark_example`

## Docker Cross-Compilation - X3 Version

1. Confirmation of Compilation Environment

- Compilation within Docker, with TogetherROS already installed in Docker. For instructions on Docker installation, cross-compilation, TogetherROS compilation, and deployment, refer to the README.md in the robot development platform robot_dev_config repository.
- dnn node package has been compiled.

2. Compilation

- Compilation command:```shell
export TARGET_ARCH=aarch64
export TARGET_TRIPLE=aarch64-linux-gnu
export CROSS_COMPILE=/usr/bin/$TARGET_TRIPLE-

colcon build --packages-select dnn_benchmark_example \
   --merge-install \
   --cmake-force-configure \
   --cmake-args \
   --no-warn-unused-cli \
   -DCMAKE_TOOLCHAIN_FILE=`pwd`/robot_dev_config/aarch64_toolchainfile.cmake
```

## Compile x86 version on X86 Ubuntu system

1. Compilation Environment Confirmation

   X86 ubuntu version: ubuntu20.04

2. Compilation

- Compilation command:

  ```shell
  colcon build --packages-select dnn_benchmark_example \
     --merge-install \
     --cmake-force-configure \
     --cmake-args \
     --no-warn-unused-cli \
     -DPLATFORM_X86=ON \
     -DTHIRD_PARTY=`pwd`/../sysroot_docker
  ```

## Notes

# User Guide

## Package Description
  The source code includes the **dnn_benchmark_example package**, which outputs the performance metrics including frame rate (fps) and single frame latency for the specified model inference process.

## Parameters

| Parameter    | Description                   | Required | Default Value | Remarks     |
| ------------ | ----------------------------- | -------- | ------------- | ----------- |
| show_fps_log | FPS switch 0: off 1: on       | No       | 1             |             |
| show_latency_log | Latency switch 0: off 1: on | No       | 1             |             |
| statistic_cycle | Number of images per processing cycle | No | 500  |              |
| is_sync_mode | Synchronous or Asynchronous mode 0: Asynchronous 1: Synchronous | No | 1 |   |
```| config_file | Configuration file path | No | config/hobot_benchmark_config.json | |
| model_file_name | Model file | No | config/multitask_body_kps_960x544.hbm | |
| model_name | Model name | No | "" | |

## Execution

After successful compilation, copy the generated install path to the Horizon X3 development board (if compiled on X3, ignore the copying step) and execute the following command to run.

## Running on X3 Ubuntu System

```shell
export COLCON_CURRENT_PREFIX=./install
source ./install/local_setup.bash

# For the example used in config, input configuration file, and local image for re-insertion
# Copy according to the actual installation path (the installation path in the docker is install/lib/dnn_benchmark_example/config/, the copy command is cp -r install/lib/dnn_benchmark_example/config/ .).
cp -r install/lib/dnn_benchmark_example/config/ .

# Run: Re-insert prediction using local jpg format images in synchronous mode, output performance metrics fps and latency through logging, and set log level to warn
ros2 run dnn_benchmark_example dnn_benchmark_example --ros-args --log-level warn

# Run default model evaluation, turn off fps information output, and change the number of processed cyclic images to 1000
ros2 run dnn_benchmark_example dnn_benchmark_example --ros-args --log-level warn -p show_fps_log:=0 -p statistic_cycle:=1000

# Change evaluation model
# The test model of dnn_benchmark_example is in the /opt/hobot/model/x3/basic/ path of the Horizon X3 development board, installed via `apt install hobot_models_basic_1.0.0_arm64.deb` command, and when running switch the evaluation model using the -p option. If users need to evaluate their own model, place the model in the config folder, similarly reconfigure the model file path and model name through -p. For example, to evaluate the fcos model, change the model file by configuring -p model_file_name:=/opt/hobot/model/x3/basic/fcos_512x512_nv12.bin, and -p statistic_cycle:=50 to change the number of processed cyclic images to 50 (i.e., evaluate the fcos_512x512_nv12.bin model, output performance indicators for every 50 images processed)
ros2 run dnn_benchmark_example dnn_benchmark_example --ros-args --log-level warn -p model_file_name:=/opt/hobot/model/x3/basic/fcos_512x512_nv12.bin -p statistic_cycle:=50
```

## Running on X3 Yocto System

```shell
export ROS_LOG_DIR=/userdata/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./install/lib/

# For the example used in config, copy according to the actual installation path
cp -r install/lib/dnn_benchmark_example/config/ .

# Run: Re-insert prediction using local jpg format images in synchronous mode, output performance metrics fps and latency through logging, and set log level to warn
./install/lib/dnn_benchmark_example/dnn_benchmark_example --ros-args --log-level warn

# Run default model evaluation, turn off fps information output, and change the number of processed cyclic images to 1000
./install/lib/dnn_benchmark_example/dnn_benchmark_example --ros-args --log-level warn -p show_fps_log:=0 -p statistic_cycle:=1000

# Change evaluation model
# The test model of dnn_benchmark_example is in the /opt/hobot/model/x3/basic path of the Horizon X3 development board, installed via `apt install hobot_models_basic_1.0.0_arm64.deb` command, and when running switch the evaluation model using the -p option. If users need to evaluate their own model, place the model in the config folder, similarly reconfigure the model file path and model name through -p. For example, to evaluate the fcos model, change the model file by configuring -p model_file_name:=/opt/hobot/model/x3/basic/fcos_512x512_nv12.bin, and -p statistic_cycle:=50 to change the number of processed cyclic images to 50 (i.e., evaluate the fcos_512x512_nv12.bin model, output performance indicators for every 50 images processed)
./install/lib/dnn_benchmark_example/dnn_benchmark_example --ros-args --log-level warn -p model_file_name:=/opt/hobot/model/x3/basic/fcos_512x512_nv12.bin -p statistic_cycle:=50
```

## Running on X86 Ubuntu System```shell
export COLCON_CURRENT_PREFIX=./install
source ./install/local_setup.bash

# Copy the model, input configuration file, and local images used for the example based on the actual installation path (the installation path in docker is install/lib/dnn_benchmark_example/config/, copy command: cp -r install/lib/dnn_benchmark_example/config/ .).
cp -r install/lib/dnn_benchmark_example/config/ .

# Run: perform inference using local jpg format images in synchronous mode, output performance metrics fps and latency in logs, and set log level to warn
ros2 run dnn_benchmark_example dnn_benchmark_example --ros-args --log-level warn

# Run evaluation with specified model, disable fps information output, and change the number of processed images per cycle to 50
ros2 run dnn_benchmark_example dnn_benchmark_example --ros-args --log-level warn -p model_file_name:=config/mobilenetv1_224x224_nv12_pyramid.bin -p show_fps_log:=0 -p statistic_cycle:=50
```

## Notes

Pay attention to lock the frequency during operation:

```shell
echo performance > /sys/class/devfreq/devfreq0/governor
echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
```

Configuration file `hobot_benchmark_config.json`

```json
{
"input_config": {
      "input_type": "image",
      "height": 544,
      "width": 960,
      "data_type": 1,   //0: BGR 1: NV12
      "image_list_file": "config/image.list", //Image configuration, one image address per line
      "need_pre_load": true,
      "limit": 2,  //Number of input minus release should be < 2
      "need_loop": true,  //Enable looping input
      "max_cache": 10
    }
}
```