// Copyright (c) 2022，Horizon Robotics.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <string>
#include <vector>

#include "ai_msgs/msg/capture_targets.hpp"
#include "ai_msgs/msg/perception_targets.hpp"
#include "cv_bridge/cv_bridge.h"
#include "dnn_node/dnn_node.h"
#include "dnn_node/util/image_proc.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

#ifdef SHARED_MEM_ENABLED
#include "hbm_img_msgs/msg/hbm_msg1080_p.hpp"
#endif

#include "include/image_utils.h"

#ifndef DNN_EXAMPLE_NODE_H_
#define DNN_EXAMPLE_NODE_H_

using rclcpp::NodeOptions;

using hobot::dnn_node::DNNInput;
using hobot::dnn_node::DnnNode;
using hobot::dnn_node::DnnNodeOutput;
using hobot::dnn_node::NV12PyramidInput;

using ai_msgs::msg::PerceptionTargets;

// 用于算法推理的图片来源，0：本地图片；1：订阅到的image msg
enum class DnnFeedType { FROM_LOCAL = 0, FROM_SUB = 1 };

// 算法解析方法类型，通过解析算法配置文件中的"dnn_Parser"配置项确定类型
enum class DnnParserType {
  INVALID_PARSER = 0,
  YOLOV2_PARSER = 1,      // 对应dnn_node中yolov2的output_parser算法
  YOLOV3_PARSER,          // 对应dnn_node中yolov3的output_parser算法
  YOLOV5_PARSER,          // 对应dnn_node中yolov5的output_parser算法
  YOLOV5X_PARSER,         // 对应dnn_node中yolov5x的output_Parser算法
  CLASSIFICATION_PARSER,  // 对应dnn_node中classification的output_parser算法
  SSD_PARSER,
  EFFICIENTDET_PARSER,
  FCOS_PARSER,
  UNET_PARSER
  /*define more*/
};

// dnn node输出数据类型
struct DnnExampleOutput : public DnnNodeOutput {
  // resize参数，用于算法检测结果的映射
  float ratio = 1.0;  //缩放比例系数，无需缩放为1

  // 算法推理使用的图像数据，用于本地渲染使用
  std::shared_ptr<hobot::dnn_node::NV12PyramidInput> pyramid = nullptr;

  // 前处理的开始和结束时间，用于发布perf统计信息
  struct timespec preprocess_timespec_start;
  struct timespec preprocess_timespec_end;

  // 订阅到的图像数据和模型输入分辨率，unet算法后处理使用
  int img_w = 0;
  int img_h = 0;
  int model_w = 0;
  int model_h = 0;
};

class DnnExampleNode : public DnnNode {
 public:
  DnnExampleNode(const std::string &node_name,
                 const NodeOptions &options = NodeOptions());
  ~DnnExampleNode() override;

 protected:
  // 集成DnnNode的接口，实现参数配置和后处理
  int SetNodePara() override;
  int PostProcess(const std::shared_ptr<DnnNodeOutput> &outputs) override;

 private:
  // 解析配置文件，包好模型文件路径、解析方法等信息
  int LoadConfig();
  // 用于解析的配置文件，以及解析后的数据
  std::string config_file = "config/fcosworkconfig.json";
  DnnParserType parser = DnnParserType::INVALID_PARSER;
  std::string model_file_name_ = "/opt/hobot/model/x3/basic/fcos_512x512_nv12.bin";
  std::string model_name_ = "";

  // 加载模型后，查询出模型输入分辨率
  int model_input_width_ = -1;
  int model_input_height_ = -1;

  // 用于预测的图片来源，0：本地图片；1：订阅到的image msg
  int feed_type_ = 0;

  // 是否在本地渲染并保存渲染后的图片
  int dump_render_img_ = 0;

  // 使用shared mem通信方式订阅图片
  int is_shared_mem_sub_ = 0;

  // 算法推理的任务数
  int task_num_ = 4;

  // 用于回灌的本地图片信息
  std::string image_file_ = "config/test.jpg";
  // 回灌图片的格式，0: bgr, 1: nv12
  int image_type_ = 0;
  // 回灌图片的分辨率
  int image_width = 0;
  int image_height = 0;
  // 本地回灌进行算法推理
  int FeedFromLocal();

  // 订阅图片进行算法推理
  int FeedFromSubscriber();

  // 发布AI消息的topic和发布者
  std::string msg_pub_topic_name_ = "hobot_dnn_detection";
  rclcpp::Publisher<ai_msgs::msg::PerceptionTargets>::SharedPtr msg_publisher_ =
      nullptr;

  // 订阅图片消息的topic和订阅者
  // 共享内存模式
#ifdef SHARED_MEM_ENABLED
  rclcpp::Subscription<hbm_img_msgs::msg::HbmMsg1080P>::ConstSharedPtr
      sharedmem_img_subscription_ = nullptr;
  std::string sharedmem_img_topic_name_ = "/hbmem_img";
  void SharedMemImgProcess(
      const hbm_img_msgs::msg::HbmMsg1080P::ConstSharedPtr msg);
#endif

  // 非共享内存模式
  rclcpp::Subscription<sensor_msgs::msg::Image>::ConstSharedPtr
      ros_img_subscription_ = nullptr;
  // 目前只支持订阅原图，可以使用压缩图"/image_raw/compressed" topic
  // 和sensor_msgs::msg::CompressedImage格式扩展订阅压缩图
  std::string ros_img_topic_name_ = "/image";
  void RosImgProcess(const sensor_msgs::msg::Image::ConstSharedPtr msg);
};

#endif  // DNN_EXAMPLE_NODE_H_
