English| [简体中文](./README_cn.md)

# Getting Started with Dnn Node
=======


# Introduction

By reading this document, users can utilize models and image data on the Horizon X3 development board to perform model inference using the BPU processor and process the parsed model outputs.

The Dnn Node package is part of the Horizon Robotics robot development platform, based on the Horizon libdnn and ROS2 Node for secondary development, providing a simpler and more user-friendly model integration development interface for application development. This includes functions such as model management, input processing and result parsing based on model descriptions, and model output memory allocation management.

The DnnNode in the Dnn Node package is a virtual base class that defines the data structures and interfaces for model integration development. Users need to inherit the DnnNode class and implement pre- and post-processing as well as configuration interfaces.

# Development Environment

- Programming Language: C/C++
- Development Platform: X3/Rdkultra/X5/X86
- System Version: Ubuntu 20.04/Ubuntu 22.04
- Compilation Toolchain: Linux GCC 9.3.0/Linaro GCC 11.4.0

# Compilation

- X3 Version: Supports compilation on X3 Ubuntu system and cross-compilation using Docker on PC.

- Rdkultra Version: Supports compilation on Rdkultra Ubuntu system and cross-compilation using Docker on PC.

- X5 Version: Supports compilation on X5 Ubuntu system and cross-compilation using Docker on PC.

- X86 Version: Supports compilation on X86 Ubuntu system.

Compilation options can control the dependencies and functionalities of compiling the package.

## Dependency Libraries

### X3 Dependencies

- dnn: 1.18.4
- opencv: 3.4.5

### Rdkultra Dependencies

- dnn: 1.17.3
- opencv: 3.4.5

### X5 Dependencies

- dnn: 1.23.5
- opencv: 3.4.5

### X86 Dependencies

- dnn: 1.12.3
- opencv: 3.4.5

## Compilation on X3/Rdkultra Ubuntu System

1. Compilation Environment Confirmation

- The current compilation terminal has set the TROS environment variable: `source /opt/tros/setup.bash`.

- The ROS2 software package build system `ament_cmake` has been installed. Installation command: `apt update; apt-get install python3-catkin-pkg; pip3 install empy`.

- The ROS2 compilation tool `colcon` has been installed. Installation command: `pip3 install -U colcon-common-extensions`.

2. Compilation

- Compile the `dnn_node` package: `colcon build --packages-select dnn_node`

## Cross-Compilation for X3/Rdkultra/X5 using Docker

1. Compilation Environment Confirmation

- Compile in Docker, where TROS has been pre-compiled in Docker. For Docker installation, cross-compilation instructions, TROS compilation, and deployment, please refer to [Horizon Robotics Robot Platform User Manual](https://developer.horizon.ai/api/v1/fileData/TogetherROS/quick_start/cross_compile.html#togetherros).

2. Compilation

- Compile the `dnn_node` package:

  ```shell

  # RDK X3
  bash robot_dev_config/build.sh -p X3 -s dnn_node

  # RDK Ultra
  bash robot_dev_config/build.sh -p Rdkultra -s dnn_node

  # RDK X5
  bash robot_dev_config/build.sh -p X5 -s dnn_node
  ```

## Compilation on X86 Ubuntu System for X86 Version

1. Compilation Environment Confirmation

  X86 Ubuntu version: ubuntu20.04

2. Compilation

- Compilation command:

  ```shell
  colcon build --packages-select dnn_node \
     --merge-install \
     --cmake-force-configure \
     --cmake-args \
``````
--no-warn-unused-cli \
     -DPLATFORM_X86=ON \
     -DTHIRD_PARTY=`pwd`/../sysroot_docker
  ```

# Usage

Users need to inherit the DnnNode virtual base class and implement virtual interfaces such as configuration.

For data and interface descriptions in the dnn node package, please refer to the API manual: docs/API-Manual/API-Manual.md

## Workflow

![](./docs/dnnnode_workflow.jpg)

There are two processes involved in usage, namely the initialization process and the runtime process.

The initialization process involves inheriting the dnn node base class, creating a user node, implementing the virtual interfaces `SetNodePara` and `PostProcess`, and using the base class's `Init` interface to complete the initialization operation.

The runtime process involves performing inference and business logic. Taking an algorithm that infers using image data as an example, the process involves subscribing to image messages (sub msg), processing the image into the algorithm's input data type (preprocessing), using the base class's `Run` interface for algorithm inference; after the inference is completed, the `PostProcess` interface callback outputs the tensor data from the algorithm, parsing the tensor data and publishing structured AI data.

## How to Use

### Select the model_task_type for algorithm inference

When configuring the model management and inference parameters `dnn_node_para_ptr_` using the `SetNodePara` interface, specify the `model_task_type` used for algorithm inference. The dnn node automatically creates a corresponding type of task based on the configured type.

Task types include `ModelInferType` and `ModelRoiInferType`, with the default being `ModelInferType`.

Use the `ModelRoiInferType` task type when the algorithm input contains roi (Region of Interest), such as in the case of hand keypoint detection algorithms where the input is an image and the roi coordinates of the hands in the image.

For other cases, select the `ModelInferType` type. For example, algorithms that only take images as input such as YOLO, FCOS, and mobilenetv2 for detection and classification.

### Set the number of inference tasks

A model supports multiple tasks to be executed, enabling parallel inference of multiple frames to improve BPU utilization and algorithm inference output frame rate. During the dnn node initialization phase (when calling the Init interface), inference tasks are created in advance based on the number of tasks configured by the user.

When configuring the model management and inference parameters `dnn_node_para_ptr_` using the `SetNodePara` interface, specify the number of algorithm inference tasks as `task_num`. The default number of inference tasks is set to 2, and if the algorithm inference takes a long time (resulting in a lower frame rate for the output), more tasks need to be specified for inference.

### Prepare algorithm input data (preprocessing)

Preprocessing involves preparing the data into the data type required for algorithm input.

For algorithms that take images as input, the dnn node provides the `hobot::dnn_node::ImageProc::GetNV12PyramidFromNV12Img` interface to generate NV12PyramidInput type data from nv12 encoded image data for algorithm inference.

### Parse the tensor output from the algorithm model

After the inference is completed, the PostProcess interface callbacks the output tensor data from the algorithm. Users need to parse the tensor data for structured AI data.

For example, for detection algorithms, a custom model output parsing method would be as follows:
``````C++
// Define the data type for algorithm output
struct DetResult {
  float xmin;
  float ymin;
  float xmax;
  float ymax;
  float score;
};

// Custom algorithm output parsing method
// - Parameters
//   - [in] node_output The DnnNodeOutput containing the algorithm inference output
//   - [in/out] results Parsed inference results
// - Return Value
//   - 0 Success
//   - -1 Failure
int32_t Parse(const std::shared_ptr<hobot::dnn_node::DnnNodeOutput> &node_output,
              std::vector<std::shared_ptr<DetResult>> &results);
```

Defined the data type DetResult for algorithm output and the algorithm output parsing method Parse, which parses the model output in node_output (containing `std::vector<std::shared_ptr<DNNTensor>> output_tensors`) and stores the structured AI data in results.

In addition, the dnn node built-in various detection, classification, and segmentation algorithm model output parsing methods, please refer to the FAQ for details.

# FAQ

1. How to obtain the algorithm inference input and output frame rates, inference time, etc.?

The `DnnNodeOutput` type data output after inference contains the inference statistics in rt_stat.

Among them, infer_time_ms represents the inference time for the current frame. The input and output frame rates are calculated on a per-second basis. When fps_updated is true, it means that the frame rate statistics have been updated for the current frame.

2. Do the resolution of the subscription images need to be consistent with the algorithm input resolution?

It is not mandatory.

During the preparation of algorithm input data (pre-processing) stage, the dnn node provides the `hobot::dnn_node::ImageProc::GetNV12PyramidFromNV12Img` interface to generate `NV12PyramidInput` type data from nv12 encoded format image data, which is used for algorithm input inference.

When processing images, if the input image resolution is smaller than the model input resolution, the input image is padded to the top-left area; if the input image resolution is larger than the model input resolution, the input image will be cropped to the top-left area.

Also, hobotcv can be used for resizing images, **resize the input image to the algorithm input resolution to retain all the image information**. Refer to the dnn_node_sample for the usage.

3. Using algorithm input information in the algorithm output parsing method and post-processing PostProcess

The input parameter type for PostProcess is hobot::dnn_node::DnnNodeOutput. Users can inherit the DnnNodeOutput data type and add the required data to use.

For example, if PostProcess requires a parameter of type uint64_t and the corresponding parameter for each frame inference input is different, it can be extended as follows:
``````
struct SampleOutput : public DnnNodeOutput {
  uint64_t para;
};
```

Set the value of `para` in PreProcess and use it in PostProcess.

4. Match Algorithm Output with Input

The input parameter type `std::shared_ptr<std_msgs::msg::Header> msg_header` in the input parameters of PostProcess in `hobot::dnn_node::DnnNodeOutput` is filled with the header of the image message subscribed in PreProcess, which can be used to match the corresponding input.

5. Synchronous and Asynchronous Inference

dnn node supports two types of inference modes: synchronous and asynchronous. When calling the Run interface for inference, specify the `is_sync_mode` parameter to set the mode, with the default being the more efficient asynchronous mode.

Synchronous inference: When calling the Run interface for inference, the interface internally blocks until the inference is completed and the PostProcess callback interface finishes processing the model output.

Asynchronous inference: When calling the Run interface for prediction, the interface internally asynchronously uses a thread pool to process the prediction. After the prediction task is sent to the thread pool, it returns directly without waiting for the prediction to finish. When the prediction result is returned (either when the prediction is completed or when there is an error), the task is released using the ReleaseTask interface inside the dnn node, and the parsed model output is returned through the PostProcess interface.

Asynchronous inference mode can fully utilize BPU, improve the algorithm's inference output frame rate, but it cannot guarantee the consistency of algorithm output order with input order. **For scenarios that have requirements on output sequences, it is necessary to determine whether the algorithm output needs to be sorted**.

**Unless there are specific requirements, it is recommended to use the default more efficient asynchronous mode for algorithm inference**.

6. Using the Built-in Model Output Parsing Methods in dnn node

dnn node comes with various model output parsing methods for detection, classification, and segmentation algorithms. After installing TROS on X3, the supported parsing methods are as follows:

```shell
root@ubuntu:~# tree /opt/tros/include/dnn_node/util/output_parser
/opt/tros/include/dnn_node/util/output_parser
├── classification
│   └── ptq_classification_output_parser.h
├── detection
│   ├── fasterrcnn_output_parser.h
│   ├── fcos_output_parser.h
│   ├── nms.h
│   ├── ptq_efficientdet_output_parser.h
│   ├── ptq_ssd_output_parser.h
│   ├── ptq_yolo2_output_parser.h
│   ├── ptq_yolo3_darknet_output_parser.h
│   └── ptq_yolo5_output_parser.h
├── perception_common.h
├── segmentation
│   └── ptq_unet_output_parser.h
└── utils.h

3 directories, 12 files
```You can see that there are three paths `classification`, `detection`, and `segmentation` under the `/opt/tros/include/dnn_node/util/output_parser` directory, corresponding to the output parsing methods of classification, detection, and segmentation algorithms.

`perception_common.h` defines the parsed perception result data type.

The algorithm models and their corresponding output parsing methods are as follows:

| Algorithm Category | Algorithm | Output Parsing Method |
| ---------------------- | ---------------------- | ----------- |
| Object Detection | [FCOS](https://developer.horizon.cc/documents_tros/boxs/detection/fcos.html) | fcos_output_parser.h |
| Object Detection | [EfficientNet_Det](https://developer.horizon.cc/documents_tros/boxs/detection/efficientnet.html) | ptq_efficientdet_output_parser.h |
| Object Detection | [MobileNet_SSD](https://developer.horizon.cc/documents_tros/boxs/detection/mobilenet.html) | ptq_ssd_output_parser.h |
| Object Detection | [YoloV2](https://developer.horizon.cc/documents_tros/boxs/detection/yolo.html) | ptq_yolo2_output_parser.h |
| Object Detection | [YoloV3](https://developer.horizon.cc/documents_tros/boxs/detection/yolo.html) | ptq_yolo3_darknet_output_parser.h |
| Object Detection | [YoloV5](https://developer.horizon.cc/documents_tros/boxs/detection/yolo.html) | ptq_yolo5_output_parser.h |
| Human Detection | [FasterRcnn](https://developer.horizon.cc/documents_tros/boxs/function/mono2d_body_detection.html) | fasterrcnn_output_parser.h |
| Image Classification | [mobilenetv2](https://developer.horizon.cc/documents_tros/boxs/classification/mobilenetv2.html) | ptq_classification_output_parser.h |
| Semantic Segmentation | [mobilenet_unet](https://developer.horizon.cc/documents_tros/boxs/segmentation/mobilenet_unet.html) | ptq_unet_output_parser.h |

In the inference result callback `PostProcess(const std::shared_ptr<hobot::dnn_node::DnnNodeOutput> &node_output)`, using the built-in parsing method in `hobot_dnn` to parse the output of the `YoloV5` algorithm is shown in the following example:

```C++
    // 1 Create parsed output data, DnnParserResult is the algorithm output data type corresponding to the built-in parsing method in hobot_dnn
    std::shared_ptr<DnnParserResult> det_result = nullptr;
    
    // 2 Start the parsing
    if (hobot::dnn_node::parser_yolov5::Parse(node_output, det_result) < 0) {
      RCLCPP_ERROR(rclcpp::get_logger("dnn_node_sample"),
                  "Parse output_tensors fail!");
      return -1;
    }
    
    // 3 Use the parsed algorithm result det_result
```