[English](./README.md) | 简体中文

# 功能介绍

Dnn Benchmark example package是Dnn Node package的性能评测示例，通过继承DnnNode虚基类，使用模型和图像数据利用BPU处理器进行模型推理，输出帧率fps和单帧延迟latency性能指标。图像数据来源于本地图片回灌。

# 开发环境

- 编程语言: C/C++
- 开发平台: X3/Rdkultra/X5/X86
- 系统版本：Ubuntu 20.04/Ubuntu 22.04
- 编译工具链:Linux GCC 9.3.0/Linaro GCC 11.4.0

# 编译

- X3版本：支持在X3 Ubuntu系统上编译和在PC上使用docker交叉编译两种方式。
- X5版本：支持在X5 Ubuntu系统上编译和在PC上使用docker交叉编译两种方式。
- X86版本：支持在X86 Ubuntu系统上编译一种方式。
同时支持通过编译选项控制编译pkg的依赖和pkg的功能。

## 依赖库

依赖库：

- opencv:3.4.5

ros package：

- dnn node

## X3 Ubuntu系统上编译 X3版本

1、编译环境确认

- 板端已安装X3 Ubuntu系统。
- 当前编译终端已设置TogetherROS环境变量：`source PATH/setup.bash`。其中PATH为TogetherROS的安装路径。
- 已安装ROS2编译工具colcon。安装的ROS不包含编译工具colcon，需要手动安装colcon。colcon安装命令：`pip install -U colcon-common-extensions`
- 已编译dnn node package

2、编译

- 编译命令：`colcon build --packages-select dnn_benchmark_example

## docker交叉编译 X3版本

1、编译环境确认

- 在docker中编译，并且docker中已经安装好TogetherROS。docker安装、交叉编译说明、TogetherROS编译和部署说明详见机器人开发平台robot_dev_config repo中的README.md。
- 已编译dnn node package

2、编译

- 编译命令：

  ```shell
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

## X86 Ubuntu系统上编译 X86版本

1、编译环境确认

  x86 ubuntu版本: ubuntu20.04
  
2、编译

- 编译命令：

  ```shell
  colcon build --packages-select dnn_benchmark_example \
     --merge-install \
     --cmake-force-configure \
     --cmake-args \
     --no-warn-unused-cli \
     -DPLATFORM_X86=ON \
     -DTHIRD_PARTY=`pwd`/../sysroot_docker
  ```

## 注意事项


# 使用介绍

## package说明
  源码包含**dnn_benchmark_example package**，输出指定模型推理过程的帧率fps和单帧延迟latency性能指标。

## 参数

| 参数名         | 解释         | 是否必须   | 默认值        | 备注         |
| ----------- | ---------- | ------ | ---------- | ---------- |
| show_fps_log | fps开关 0:关闭 1:打开| 否      | 1 |            |
| show_latency_log | latency开关  0:关闭 1:打开   | 否  | 1 |  |
| statistic_cycle | 处理周期图片个数 | 否  | 500 |  |
| is_sync_mode | 同步或异步模式 0: 异步 1: 同步| 否  |  1  |  |
| config_file | 配置文件路径 | 否 | config/hobot_benchmark_config.json | |
| model_file_name | 模型文件 | 否  |  config/multitask_body_kps_960x544.hbm   |  |
| model_name | 模型名称 | 否  | "" |  |

## 运行

编译成功后，将生成的install路径拷贝到RDK X3开发板上（如果是在X3上编译，忽略拷贝步骤），并执行如下命令运行。

## X3 Ubuntu系统上运行

```shell
export COLCON_CURRENT_PREFIX=./install
source ./install/local_setup.bash

# config中为example使用的模型，输入的配置文件，回灌使用的本地图片
# 根据实际安装路径进行拷贝（docker中的安装路径为install/lib/dnn_benchmark_example/config/，拷贝命令为cp -r install/lib/dnn_benchmark_example/config/ .）。
cp -r install/lib/dnn_benchmark_example/config/ .

# 运行：使用本地jpg格式图片通过同步模式进行回灌预测，通过日志输出性能指标fps和latency，并设置log级别为warn
ros2 run dnn_benchmark_example dnn_benchmark_example --ros-args --log-level warn

# 运行默认模型评测，关闭fsp信息输出，处理周期图片个数更改为1000
ros2 run dnn_benchmark_example dnn_benchmark_example --ros-args --log-level warn -p show_fps_log:=0 -p statistic_cycle:=1000

# 更换评测模型
# dnn_benchmark_example的测试模型在X3派开发板的/opt/hobot/model/x3/basic/路径下，通过`apt install hobot_models_basic_1.0.0_arm64.deb`命令安装，在运行时通过-p选项更换评测模型。如果用户需要评测自己的模型，将模型放在config文件夹下，同样将模型文件路径和模型名通过-p重新配置即可。例如，对fcos模型进行评测，通过配置参数-p model_file_name:=/opt/hobot/model/x3/basic/fcos_512x512_nv12.bin更改模型文件，-p statistic_cycle:=50将处理周期图片个数更改为50个(即评测fcos_512x512_nv12.bin模型，每处理50张图片输出一次性能指标)
ros2 run dnn_benchmark_example dnn_benchmark_example --ros-args --log-level warn -p model_file_name:=/opt/hobot/model/x3/basic/fcos_512x512_nv12.bin -p statistic_cycle:=50
```

## X3 yocto系统上运行

```shell
export ROS_LOG_DIR=/userdata/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./install/lib/

# config中为示例使用的模型，根据实际安装路径进行拷贝
cp -r install/lib/dnn_benchmark_example/config/ .

# 运行：使用本地jpg格式图片通过同步模式进行回灌预测，通过日志输出性能指标fps和latency，并设置log级别为warn
./install/lib/dnn_benchmark_example/dnn_benchmark_example --ros-args --log-level warn

# 运行默认模型评测，关闭fsp信息输出，处理周期图片个数更改为1000
./install/lib/dnn_benchmark_example/dnn_benchmark_example --ros-args --log-level warn -p show_fps_log:=0 -p statistic_cycle:=1000

# 更换评测模型
# dnn_benchmark_example的测试模型在X3派开发板的/opt/hobot/model/x3/basic路径下，通过`apt install hobot_models_basic_1.0.0_arm64.deb`命令安装，在运行时通过-p选项更换评测模型。如果用户需要评测自己的模型，将模型放在config文件夹下，同样将模型文件路径和模型名通过-p重新配置即可。例如，对fcos模型进行评测，通过配置参数-p model_file_name:=/opt/hobot/model/x3/basic/fcos_512x512_nv12.bin更改模型文件，-p statistic_cycle:=50将处理周期图片个数更改为50个(即评测fcos_512x512_nv12.bin模型，每处理50张图片输出一次性能指标)
./install/lib/dnn_benchmark_example/dnn_benchmark_example --ros-args --log-level warn -p model_file_name:=/opt/hobot/model/x3/basic/fcos_512x512_nv12.bin -p statistic_cycle:=50
```

## X86 Ubuntu系统上运行

```shell
export COLCON_CURRENT_PREFIX=./install
source ./install/local_setup.bash

# config中为example使用的模型，输入的配置文件，回灌使用的本地图片
# 根据实际安装路径进行拷贝（docker中的安装路径为install/lib/dnn_benchmark_example/config/，拷贝命令为cp -r install/lib/dnn_benchmark_example/config/ .）。
cp -r install/lib/dnn_benchmark_example/config/ .


# 运行：使用本地jpg格式图片通过同步模式进行回灌预测，通过日志输出性能指标fps和latency，并设置log级别为warn
ros2 run dnn_benchmark_example dnn_benchmark_example --ros-args --log-level warn

# 运行指定模型评测，关闭fsp信息输出，处理周期图片个数更改为50
ros2 run dnn_benchmark_example dnn_benchmark_example --ros-args --log-level warn -p model_file_name:=config/mobilenetv1_224x224_nv12_pyramid.bin -p show_fps_log:=0 -p statistic_cycle:=50
```

## 注意事项

运行时需要注意锁频：

```shell
echo performance > /sys/class/devfreq/devfreq0/governor
echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
```

配置文件`hobot_benchmark_config.json`

```json
{
"input_config": {
      "input_type": "image",
      "height": 544,
      "width": 960,
      "data_type": 1,   //0: BGR 1: NV12
      "image_list_file": "config/image.list", //图片配置 每行一个图片地址
      "need_pre_load": true,
      "limit": 2,  //input输入个数减去release个数 < 2
      "need_loop": true,  //循环输入
      "max_cache": 10
    }
}
```
