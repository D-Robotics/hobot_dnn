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

#include "include/dnn_example_node.h"

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "hobot_cv/hobotcv_imgproc.h"
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "rapidjson/writer.h"
#include "rclcpp/rclcpp.hpp"
#include <cv_bridge/cv_bridge.h>
#include <unistd.h>

#include "dnn_node/dnn_node.h"
#include "dnn_node/util/output_parser/classification/ptq_classification_output_parser.h"
#include "dnn_node/util/output_parser/detection/fcos_output_parser.h"
#include "dnn_node/util/output_parser/detection/ptq_efficientdet_output_parser.h"
#include "dnn_node/util/output_parser/detection/ptq_ssd_output_parser.h"
#include "dnn_node/util/output_parser/detection/ptq_yolo2_output_parser.h"
#include "dnn_node/util/output_parser/detection/ptq_yolo3_darknet_output_parser.h"
#include "dnn_node/util/output_parser/detection/ptq_yolo5_output_parser.h"
#include "dnn_node/util/output_parser/detection/ptq_yolov5x_output_parser.h"
#include "dnn_node/util/output_parser/segmentation/ptq_unet_output_parser.h"

#include "include/image_utils.h"
#include "include/post_process/post_process_unet.h"

// 时间格式转换
builtin_interfaces::msg::Time ConvertToRosTime(
    const struct timespec &time_spec) {
  builtin_interfaces::msg::Time stamp;
  stamp.set__sec(time_spec.tv_sec);
  stamp.set__nanosec(time_spec.tv_nsec);
  return stamp;
}

// 根据起始时间计算耗时
int CalTimeMsDuration(const builtin_interfaces::msg::Time &start,
                      const builtin_interfaces::msg::Time &end) {
  return (end.sec - start.sec) * 1000 + end.nanosec / 1000 / 1000 -
         start.nanosec / 1000 / 1000;
}

// 使用hobotcv resize nv12格式图片，固定图片宽高比
int ResizeNV12Img(const char *in_img_data,
                  const int &in_img_height,
                  const int &in_img_width,
                  const int &scaled_img_height,
                  const int &scaled_img_width,
                  cv::Mat &out_img,
                  float &ratio) {
  cv::Mat src(
      in_img_height * 3 / 2, in_img_width, CV_8UC1, (void *)(in_img_data));
  float ratio_w =
      static_cast<float>(in_img_width) / static_cast<float>(scaled_img_width);
  float ratio_h =
      static_cast<float>(in_img_height) / static_cast<float>(scaled_img_height);
  float dst_ratio = std::max(ratio_w, ratio_h);
  int resized_width, resized_height;
  if (dst_ratio == ratio_w) {
    resized_width = scaled_img_width;
    resized_height = static_cast<float>(in_img_height) / dst_ratio;
  } else if (dst_ratio == ratio_h) {
    resized_width = static_cast<float>(in_img_width) / dst_ratio;
    resized_height = scaled_img_height;
  }

  // hobot_cv要求输出宽度为16的倍数
  int remain = resized_width % 16;
  if (remain != 0) {
    //向下取16倍数，重新计算缩放系数
    resized_width -= remain;
    dst_ratio = static_cast<float>(in_img_width) / resized_width;
    resized_height = static_cast<float>(in_img_height) / dst_ratio;
  }
  //高度向下取偶数
  resized_height =
      resized_height % 2 == 0 ? resized_height : resized_height - 1;
  ratio = dst_ratio;

  return hobot_cv::hobotcv_resize(
      src, in_img_height, in_img_width, out_img, resized_height, resized_width);
}

DnnExampleNode::DnnExampleNode(const std::string &node_name,
                               const NodeOptions &options)
    : DnnNode(node_name, options) {
  // 更新配置
  this->declare_parameter<int>("feed_type", feed_type_);
  this->declare_parameter<std::string>("image", image_file_);
  this->declare_parameter<int>("image_type", image_type_);
  this->declare_parameter<int>("image_width", image_width);
  this->declare_parameter<int>("image_height", image_height);
  this->declare_parameter<int>("dump_render_img", dump_render_img_);
  this->declare_parameter<int>("is_shared_mem_sub", is_shared_mem_sub_);
  this->declare_parameter<std::string>("config_file", config_file);
  this->declare_parameter<std::string>("msg_pub_topic_name",
                                       msg_pub_topic_name_);

  this->get_parameter<int>("feed_type", feed_type_);
  this->get_parameter<std::string>("image", image_file_);
  this->get_parameter<int>("image_type", image_type_);
  this->get_parameter<int>("image_width", image_width);
  this->get_parameter<int>("image_height", image_height);
  this->get_parameter<int>("dump_render_img", dump_render_img_);
  this->get_parameter<int>("is_shared_mem_sub", is_shared_mem_sub_);
  this->get_parameter<std::string>("config_file", config_file);
  this->get_parameter<std::string>("msg_pub_topic_name", msg_pub_topic_name_);

  {
    std::stringstream ss;
    ss << "Parameter:"
       << "\n feed_type(0:local, 1:sub): " << feed_type_
       << "\n image: " << image_file_ << "\n image_type: " << image_type_
       << "\n dump_render_img: " << dump_render_img_
       << "\n is_shared_mem_sub: " << is_shared_mem_sub_
       << "\n config_file: " << config_file
       << "\n msg_pub_topic_name_: " << msg_pub_topic_name_;
    RCLCPP_WARN(rclcpp::get_logger("example"), "%s", ss.str().c_str());
  }
  // 加载配置文件config_file
  if (LoadConfig() < 0) {
    RCLCPP_ERROR(rclcpp::get_logger("example"), "Load config fail!");
    rclcpp::shutdown();
    return;
  }
  {
    std::stringstream ss;
    ss << "Parameter:"
       << "\n model_file_name: " << model_file_name_
       << "\n model_name: " << model_name_;
    RCLCPP_WARN(rclcpp::get_logger("example"), "%s", ss.str().c_str());
  }

  // 使用基类接口初始化，加载模型
  if (Init() != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("example"), "Init failed!");
    rclcpp::shutdown();
    return;
  }

  // 未指定模型名，从加载的模型中查询出模型名
  if (model_name_.empty()) {
    if (!GetModel()) {
      RCLCPP_ERROR(rclcpp::get_logger("example"), "Get model fail.");
    } else {
      model_name_ = GetModel()->GetName();
      RCLCPP_WARN(rclcpp::get_logger("example"), "Get model name: %s from load model.", model_name_.c_str());
    }
  }

  // 加载模型后查询模型输入分辨率
  if (GetModelInputSize(0, model_input_width_, model_input_height_) < 0) {
    RCLCPP_ERROR(rclcpp::get_logger("example"), "Get model input size fail!");
  } else {
    RCLCPP_INFO(rclcpp::get_logger("example"),
                "The model input width is %d and height is %d",
                model_input_width_,
                model_input_height_);
  }

  // 创建AI消息的发布者
  RCLCPP_WARN(rclcpp::get_logger("example"),
              "Create ai msg publisher with topic_name: %s",
              msg_pub_topic_name_.c_str());
  msg_publisher_ = this->create_publisher<ai_msgs::msg::PerceptionTargets>(
      msg_pub_topic_name_, 10);

  if (static_cast<int>(DnnFeedType::FROM_LOCAL) == feed_type_) {
    // 本地图片回灌
    RCLCPP_INFO(rclcpp::get_logger("example"),
                "Dnn node feed with local image: %s",
                image_file_.c_str());
    FeedFromLocal();
  } 
  else if (static_cast<int>(DnnFeedType::FROM_SUB) == feed_type_) {
    // 创建图片消息的订阅者
    RCLCPP_INFO(rclcpp::get_logger("example"),
                "Dnn node feed with subscription");
    if (is_shared_mem_sub_) {
#ifdef SHARED_MEM_ENABLED
      RCLCPP_WARN(rclcpp::get_logger("example"),
                  "Create img hbmem_subscription with topic_name: %s",
                  sharedmem_img_topic_name_.c_str());
      sharedmem_img_subscription_ =
          this->create_subscription<hbm_img_msgs::msg::HbmMsg1080P>(
              sharedmem_img_topic_name_,
              rclcpp::SensorDataQoS(),
              std::bind(&DnnExampleNode::SharedMemImgProcess,
                        this,
                        std::placeholders::_1));
#else
      RCLCPP_ERROR(rclcpp::get_logger("example"), "Unsupport shared mem");
#endif
    } else {
      RCLCPP_WARN(rclcpp::get_logger("example"),
                  "Create img subscription with topic_name: %s",
                  ros_img_topic_name_.c_str());
      ros_img_subscription_ =
          this->create_subscription<sensor_msgs::msg::Image>(
              ros_img_topic_name_,
              10,
              std::bind(
                  &DnnExampleNode::RosImgProcess, this, std::placeholders::_1));
    }
  } else {
    RCLCPP_ERROR(
        rclcpp::get_logger("example"), "Invalid feed_type:%d", feed_type_);
    rclcpp::shutdown();
    return;
  }
}

DnnExampleNode::~DnnExampleNode() {}

int DnnExampleNode::LoadConfig() {
  if (config_file.empty()) {
    RCLCPP_ERROR(rclcpp::get_logger("example"),
                 "Config file [%s] is empty!",
                 config_file.data());
    return -1;
  }
  // Parsing config
  std::ifstream ifs(config_file.c_str());
  if (!ifs) {
    RCLCPP_ERROR(rclcpp::get_logger("example"),
                 "Read config file [%s] fail!",
                 config_file.data());
    return -1;
  }
  rapidjson::IStreamWrapper isw(ifs);
  rapidjson::Document document;
  document.ParseStream(isw);
  if (document.HasParseError()) {
    RCLCPP_ERROR(rclcpp::get_logger("example"),
                 "Parsing config file %s failed",
                 config_file.data());
    return -1;
  }

  if (document.HasMember("model_file")) {
    model_file_name_ = document["model_file"].GetString();
  }
  if (document.HasMember("model_name")) {
    model_name_ = document["model_name"].GetString();
  }

  int ret = 0;
  // 更新parser，后处理中根据parser类型选择解析方法
  if (document.HasMember("dnn_Parser")) {
    std::string str_parser = document["dnn_Parser"].GetString();
    if ("yolov2" == str_parser) {
      parser = DnnParserType::YOLOV2_PARSER;
      ret = hobot::dnn_node::parser_yolov2::LoadConfig(document);
    } else if ("yolov3" == str_parser) {
      parser = DnnParserType::YOLOV3_PARSER;
      ret = hobot::dnn_node::parser_yolov3::LoadConfig(document);
#ifdef PLATFORM_X3
    } else if ("yolov5" == str_parser) {
      parser = DnnParserType::YOLOV5_PARSER;
      ret = hobot::dnn_node::parser_yolov5::LoadConfig(document);
    } else if ("efficient_det" == str_parser) {
      parser = DnnParserType::EFFICIENTDET_PARSER;
      if (document.HasMember("dequanti_file")) {
        std::string dequanti_file = document["dequanti_file"].GetString();
        if (hobot::dnn_node::parser_efficientdet::LoadDequantiFile(
                dequanti_file) < 0) {
          RCLCPP_WARN(rclcpp::get_logger("example"),
                      "Load efficientdet dequanti file [%s] fail",
                      dequanti_file.data());
          return -1;
        }
      } else {
        RCLCPP_WARN(rclcpp::get_logger("example"),
                    "classification file is not set");
      }
#endif
#ifdef PLATFORM_Rdkultra
    } else if ("yolov5x" == str_parser) {
      parser = DnnParserType::YOLOV5X_PARSER;
      ret = hobot::dnn_node::parser_yolov5x::LoadConfig(document);
#endif
#ifdef PLATFORM_X5
    } else if ("yolov5x" == str_parser) {
      parser = DnnParserType::YOLOV5X_PARSER;
      ret = hobot::dnn_node::parser_yolov5x::LoadConfig(document);
#endif
    } else if ("classification" == str_parser) {
      parser = DnnParserType::CLASSIFICATION_PARSER;
      ret = hobot::dnn_node::parser_mobilenetv2::LoadConfig(document);
    } else if ("ssd" == str_parser) {
      parser = DnnParserType::SSD_PARSER;

    } else if ("fcos" == str_parser) {
      parser = DnnParserType::FCOS_PARSER;
      ret = hobot::dnn_node::parser_fcos::LoadConfig(document);
    } else if ("unet" == str_parser) {
      parser = DnnParserType::UNET_PARSER;
    } else {
      std::stringstream ss;
      ss << "Error! Invalid parser: " << str_parser
         << " . Only yolov2, yolov3, yolov5, yolov5x, ssd, fcos"
         << " efficient_det, classification, unet are supported";
      RCLCPP_ERROR(rclcpp::get_logger("example"), "%s", ss.str().c_str());
      return -3;
    }
    if (ret < 0) {
      RCLCPP_ERROR(rclcpp::get_logger("example"),
                   "Load %s Parser config file fail",
                   str_parser.data());
      return -1;
    }
  }

  return 0;
}

int DnnExampleNode::SetNodePara() {
  RCLCPP_INFO(rclcpp::get_logger("example"), "Set node para.");
  if (!dnn_node_para_ptr_) {
    return -1;
  }
  dnn_node_para_ptr_->model_file = model_file_name_;
  dnn_node_para_ptr_->model_name = model_name_;
  dnn_node_para_ptr_->model_task_type =
      hobot::dnn_node::ModelTaskType::ModelInferType;
  dnn_node_para_ptr_->task_num = task_num_;

  RCLCPP_WARN(rclcpp::get_logger("example"),
              "model_file_name_: %s, task_num: %d",
              model_file_name_.data(),
              dnn_node_para_ptr_->task_num);

  return 0;
}

int DnnExampleNode::PostProcess(
    const std::shared_ptr<DnnNodeOutput> &node_output) {
  if (!rclcpp::ok()) {
    return -1;
  }

  // 1. 记录后处理开始时间
  struct timespec time_start = {0, 0};
  clock_gettime(CLOCK_REALTIME, &time_start);

  auto parser_output = std::dynamic_pointer_cast<DnnExampleOutput>(node_output);
  if (parser_output) {
    std::stringstream ss;
    ss << "Output from frame_id: " << parser_output->msg_header->frame_id
       << ", stamp: " << parser_output->msg_header->stamp.sec << "."
       << parser_output->msg_header->stamp.nanosec;
    RCLCPP_INFO(rclcpp::get_logger("example"), "%s", ss.str().c_str());
  }

  // 校验算法输出是否有效
  if (node_output->output_tensors.empty()) {
    RCLCPP_ERROR(rclcpp::get_logger("PostProcessBase"),
                 "Invalid node_output->output_tensors");
    return -1;
  }

  // 2. 解析后的结构化数据
  std::shared_ptr<DnnParserResult> det_result = nullptr;
  int parse_ret = 0;
  // 根据parser类型选择解析方法
  switch (parser) {
    case DnnParserType::YOLOV2_PARSER:
      parse_ret =
          hobot::dnn_node::parser_yolov2::Parse(node_output, det_result);
      break;
    case DnnParserType::YOLOV3_PARSER:
      parse_ret =
          hobot::dnn_node::parser_yolov3::Parse(node_output, det_result);
      break;
  #ifdef PLATFORM_X3
    case DnnParserType::YOLOV5_PARSER:
      parse_ret =
          hobot::dnn_node::parser_yolov5::Parse(node_output, det_result);
      break;
    case DnnParserType::EFFICIENTDET_PARSER:
      parse_ret =
          hobot::dnn_node::parser_efficientdet::Parse(node_output, det_result);
      break;
  #endif
  #ifdef PLATFORM_Rdkultra
    case DnnParserType::YOLOV5X_PARSER:
      parse_ret =
          hobot::dnn_node::parser_yolov5x::Parse(node_output, det_result);
      break;
  #endif
  #ifdef PLATFORM_X5
    case DnnParserType::YOLOV5X_PARSER:
      parse_ret =
          hobot::dnn_node::parser_yolov5x::Parse(node_output, det_result);
      break;
  #endif
    case DnnParserType::CLASSIFICATION_PARSER:
      parse_ret =
          hobot::dnn_node::parser_mobilenetv2::Parse(node_output, det_result);
      break;
    case DnnParserType::SSD_PARSER:
      parse_ret = hobot::dnn_node::parser_ssd::Parse(node_output, det_result);
      break;
    case DnnParserType::FCOS_PARSER:
      parse_ret = hobot::dnn_node::parser_fcos::Parse(node_output, det_result);
      break;
    case DnnParserType::UNET_PARSER:
      parse_ret = hobot::dnn_node::parser_unet::Parse(node_output, 
                                            parser_output->img_w,
                                            parser_output->img_h,
                                            parser_output->model_w,
                                            parser_output->model_h,
                                            dump_render_img_,
                                            det_result);
      break;
    default:
      RCLCPP_ERROR(rclcpp::get_logger("example"), "Inlvaid parser: %d", parser);
      return -1;
  }

  if (parse_ret < 0) {
    RCLCPP_ERROR(rclcpp::get_logger("example"), "Parse fail");
    return -1;
  }

  // 3. 创建用于发布的AI消息
  if (!msg_publisher_) {
    RCLCPP_ERROR(rclcpp::get_logger("example"), "Invalid msg_publisher_");
    return -1;
  }
  ai_msgs::msg::PerceptionTargets::UniquePtr pub_data(
      new ai_msgs::msg::PerceptionTargets());
  // 3.1 发布检测AI消息
  RCLCPP_INFO(rclcpp::get_logger("PostProcessBase"),
              "out box size: %d",
              det_result->perception.det.size());
  for (auto &rect : det_result->perception.det) {
    if (rect.bbox.xmin < 0) rect.bbox.xmin = 0;
    if (rect.bbox.ymin < 0) rect.bbox.ymin = 0;
    if (rect.bbox.xmax >= model_input_width_) {
      rect.bbox.xmax = model_input_width_ - 1;
    }
    if (rect.bbox.ymax >= model_input_height_) {
      rect.bbox.ymax = model_input_height_ - 1;
    }

    std::stringstream ss;
    ss << "det rect: " << rect.bbox.xmin << " " << rect.bbox.ymin << " "
       << rect.bbox.xmax << " " << rect.bbox.ymax
       << ", det type: " << rect.class_name << ", score:" << rect.score;
    RCLCPP_INFO(rclcpp::get_logger("PostProcessBase"), "%s", ss.str().c_str());

    ai_msgs::msg::Roi roi;
    roi.set__type(rect.class_name);
    roi.rect.set__x_offset(rect.bbox.xmin);
    roi.rect.set__y_offset(rect.bbox.ymin);
    roi.rect.set__width(rect.bbox.xmax - rect.bbox.xmin);
    roi.rect.set__height(rect.bbox.ymax - rect.bbox.ymin);
    roi.set__confidence(rect.score);

    ai_msgs::msg::Target target;
    target.set__type(rect.class_name);
    target.rois.emplace_back(roi);
    pub_data->targets.emplace_back(std::move(target));
  }

  // 3.2 发布分类AI消息
  RCLCPP_INFO(rclcpp::get_logger("ClassificationPostProcess"),
              "out cls size: %d",
              det_result->perception.cls.size());
  for (auto &cls : det_result->perception.cls) {
    std::string clsname = cls.class_name;
    std::stringstream ss;
    ss << "class type:" << cls.class_name << ", score:" << cls.score;
    RCLCPP_INFO(rclcpp::get_logger("ClassificationPostProcess"),
                "%s",
                ss.str().c_str());

    auto xmin = model_input_width_ / 2;
    auto ymin = model_input_height_ / 2;
    ai_msgs::msg::Roi roi;
    roi.rect.set__x_offset(xmin);
    roi.rect.set__y_offset(ymin);
    roi.rect.set__width(0);
    roi.rect.set__height(0);

    ai_msgs::msg::Target target;
    target.set__type(cls.class_name);
    target.rois.emplace_back(roi);
    pub_data->targets.emplace_back(std::move(target));
  }

  // 3.3 发布分割AI消息
  auto &seg = det_result->perception.seg;
  if (seg.height != 0 && seg.width != 0) {
    if (dump_render_img_) {
      hobot::dnn_node::parser_unet::RenderUnet(node_output, seg);
    }

    ai_msgs::msg::Capture capture;
    capture.features.swap(seg.data);
    capture.img.height = seg.height;
    capture.img.width = seg.width;
    capture.img.step = model_input_width_ / seg.width;

    RCLCPP_INFO(rclcpp::get_logger("SegmentationPostProcess"),
                "features size: %d, width: %d, height: %d, num_classes: %d, step: %d",
                capture.features.size(),
                capture.img.width,
                capture.img.height,
                seg.num_classes,
                capture.img.step);

    ai_msgs::msg::Target target;
    target.set__type("parking_space");
    
    ai_msgs::msg::Attribute attribute;
    attribute.set__type("segmentation_label_count");
    attribute.set__value(seg.num_classes);
    target.attributes.emplace_back(std::move(attribute));

    target.captures.emplace_back(std::move(capture));
    pub_data->targets.emplace_back(std::move(target));
  }

  pub_data->header.set__stamp(parser_output->msg_header->stamp);
  pub_data->header.set__frame_id(parser_output->msg_header->frame_id);

  // 如果开启了渲染，本地渲染并存储图片
  if (dump_render_img_ && parser_output->pyramid) {
    ImageUtils::Render(parser_output->pyramid, pub_data);
  }

  if (parser_output->ratio != 1.0) {
    // 前处理有对图片进行resize，需要将坐标映射到对应的订阅图片分辨率
    for (auto &target : pub_data->targets) {
      for (auto &roi : target.rois) {
        roi.rect.x_offset *= parser_output->ratio;
        roi.rect.y_offset *= parser_output->ratio;
        roi.rect.width *= parser_output->ratio;
        roi.rect.height *= parser_output->ratio;
      }
    }
  }

  // 填充perf性能统计信息
  // 前处理统计
  ai_msgs::msg::Perf perf_preprocess;
  perf_preprocess.set__type(model_name_ + "_preprocess");
  perf_preprocess.set__stamp_start(
      ConvertToRosTime(parser_output->preprocess_timespec_start));
  perf_preprocess.set__stamp_end(
      ConvertToRosTime(parser_output->preprocess_timespec_end));
  perf_preprocess.set__time_ms_duration(CalTimeMsDuration(
      perf_preprocess.stamp_start, perf_preprocess.stamp_end));
  pub_data->perfs.emplace_back(perf_preprocess);

  // dnn node有输出统计信息
  if (node_output->rt_stat) {
    struct timespec time_now = {0, 0};
    clock_gettime(CLOCK_REALTIME, &time_now);

    // 推理统计
    ai_msgs::msg::Perf perf;
    perf.set__type(model_name_ + "_predict_infer");
    perf.stamp_start =
        ConvertToRosTime(node_output->rt_stat->infer_timespec_start);
    perf.stamp_end = ConvertToRosTime(node_output->rt_stat->infer_timespec_end);
    perf.set__time_ms_duration(node_output->rt_stat->infer_time_ms);
    pub_data->perfs.push_back(perf);

    perf.set__type(model_name_ + "_predict_parse");
    perf.stamp_start =
        ConvertToRosTime(node_output->rt_stat->parse_timespec_start);
    perf.stamp_end = ConvertToRosTime(node_output->rt_stat->parse_timespec_end);
    perf.set__time_ms_duration(node_output->rt_stat->parse_time_ms);
    pub_data->perfs.push_back(perf);

    // 后处理统计
    ai_msgs::msg::Perf perf_postprocess;
    perf_postprocess.set__type(model_name_ + "_postprocess");
    perf_postprocess.stamp_start = ConvertToRosTime(time_start);
    clock_gettime(CLOCK_REALTIME, &time_now);
    perf_postprocess.stamp_end = ConvertToRosTime(time_now);
    perf_postprocess.set__time_ms_duration(CalTimeMsDuration(
        perf_postprocess.stamp_start, perf_postprocess.stamp_end));
    pub_data->perfs.emplace_back(perf_postprocess);

    // 从发布图像到发布AI结果的延迟
    ai_msgs::msg::Perf perf_pipeline;
    perf_pipeline.set__type(model_name_ + "_pipeline");
    perf_pipeline.set__stamp_start(pub_data->header.stamp);
    perf_pipeline.set__stamp_end(perf_postprocess.stamp_end);
    perf_pipeline.set__time_ms_duration(
        CalTimeMsDuration(perf_pipeline.stamp_start, perf_pipeline.stamp_end));
    pub_data->perfs.push_back(perf_pipeline);

    if (parser_output) {
      // Output time delay info
      RCLCPP_DEBUG_STREAM(rclcpp::get_logger("example"),
        "frame_id: " << parser_output->msg_header->frame_id
        << ", stamp: " << parser_output->msg_header->stamp.sec << "."
        << parser_output->msg_header->stamp.nanosec
        << ", recv delay: " << CalTimeMsDuration(parser_output->msg_header->stamp, ConvertToRosTime(parser_output->preprocess_timespec_start))
        << ", preprocess time ms: " << static_cast<int>(perf_preprocess.time_ms_duration)
        << ", infer time ms: " << node_output->rt_stat->infer_time_ms
        << ", post process time ms: " << static_cast<int>(perf_postprocess.time_ms_duration)
        << ", pipeline time ms: " << static_cast<int>(perf_pipeline.time_ms_duration)
        << ", infer_timespec_start: " << std::to_string(node_output->rt_stat->infer_timespec_start.tv_sec) << "." << std::to_string(node_output->rt_stat->infer_timespec_start.tv_nsec)
        << ", infer_timespec_end: " << std::to_string(node_output->rt_stat->infer_timespec_end.tv_sec) << "." << std::to_string(node_output->rt_stat->infer_timespec_end.tv_nsec)
        << ", parse_timespec_start: " << std::to_string(node_output->rt_stat->parse_timespec_start.tv_sec) << "." << std::to_string(node_output->rt_stat->parse_timespec_start.tv_nsec)
        << ", parse_timespec_end: " << std::to_string(node_output->rt_stat->parse_timespec_end.tv_sec) << "." << std::to_string(node_output->rt_stat->parse_timespec_end.tv_nsec)
      );
    }

    // 推理输出帧率统计
    pub_data->set__fps(round(node_output->rt_stat->output_fps));

    // 如果当前帧有更新统计信息，输出统计信息
    if (node_output->rt_stat->fps_updated) {
      RCLCPP_WARN(rclcpp::get_logger("example"),
                  "Sub img fps: %.2f, Smart fps: %.2f, infer time ms: %d, "
                  "post process time ms: %d",
                  node_output->rt_stat->input_fps,
                  node_output->rt_stat->output_fps,
                  node_output->rt_stat->infer_time_ms,
                  static_cast<int>(perf_postprocess.time_ms_duration));
    }
  }

  // 发布AI消息
  msg_publisher_->publish(std::move(pub_data));
  return 0;
}

int DnnExampleNode::FeedFromLocal() {
  if (access(image_file_.c_str(), R_OK) == -1) {
    RCLCPP_ERROR(
        rclcpp::get_logger("example"), "Image: %s not exist!", image_file_.c_str());
    return -1;
  }

  // 1. 将图片处理成模型输入数据类型DNNInput
  // 使用图片生成pym，NV12PyramidInput为DNNInput的子类
  std::shared_ptr<hobot::dnn_node::NV12PyramidInput> pyramid = nullptr;
  if (static_cast<int>(ImageType::BGR) == image_type_) {
    // bgr img，支持将图片resize到模型输入size
    pyramid = hobot::dnn_node::ImageProc::GetNV12PyramidFromBGR(
        image_file_, model_input_height_, model_input_width_);
    if (!pyramid) {
      RCLCPP_ERROR(rclcpp::get_logger("example"),
                   "Get Nv12 pym fail with image: %s",
                   image_file_.c_str());
      return -1;
    }
  } else if (static_cast<int>(ImageType::NV12) == image_type_) {
    std::ifstream ifs(image_file_, std::ios::in | std::ios::binary);
    if (!ifs) {
      return -1;
    }
    ifs.seekg(0, std::ios::end);
    int len = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    char *data = new char[len];
    ifs.read(data, len);
    pyramid = hobot::dnn_node::ImageProc::GetNV12PyramidFromNV12Img(
        data,
        image_height,
        image_width,
        model_input_height_,
        model_input_width_);
    if (!pyramid) {
      RCLCPP_ERROR(rclcpp::get_logger("example"),
                   "Get Nv12 pym fail with image: %s",
                   image_file_.c_str());
      return -1;
    }

  } else if (static_cast<int>(ImageType::BIN) == image_type_) {
    // 读取bin文件送模型推理
  } else {
    RCLCPP_ERROR(
        rclcpp::get_logger("example"), "Invalid image type: %d", image_type_);
    return -1;
  }

  // 2. 使用pyramid创建DNNInput对象inputs
  // inputs将会作为模型的输入通过RunInferTask接口传入
  auto inputs = std::vector<std::shared_ptr<DNNInput>>{pyramid};
  auto dnn_output = std::make_shared<DnnExampleOutput>();
  dnn_output->msg_header = std::make_shared<std_msgs::msg::Header>();
  dnn_output->msg_header->set__frame_id("feedback");

  if (dump_render_img_) {
    dnn_output->pyramid = pyramid;
  }

  uint32_t ret = 0;
  // 3. 开始预测
  ret = Run(inputs, dnn_output, nullptr, true);

  // 4. 处理预测结果，如渲染到图片或者发布预测结果
  if (ret != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("example"), "Run predict failed!");
    return ret;
  }
  return 0;
}

void DnnExampleNode::RosImgProcess(
    const sensor_msgs::msg::Image::ConstSharedPtr img_msg) {
  if (!img_msg) {
    RCLCPP_DEBUG(rclcpp::get_logger("example"), "Get img failed");
    return;
  }

  if (!rclcpp::ok()) {
    return;
  }

  std::stringstream ss;
  ss << "Recved img encoding: " << img_msg->encoding
     << ", h: " << img_msg->height << ", w: " << img_msg->width
     << ", step: " << img_msg->step
     << ", frame_id: " << img_msg->header.frame_id
     << ", stamp: " << img_msg->header.stamp.sec << "_"
     << img_msg->header.stamp.nanosec
     << ", data size: " << img_msg->data.size();
  RCLCPP_INFO(rclcpp::get_logger("example"), "%s", ss.str().c_str());

  // dump recved img msg
  // std::ofstream ofs("img_" + img_msg->header.frame_id +
  //    std::to_string(img_msg->header.stamp.sec) + "_" +
  //    std::to_string(img_msg->header.stamp.nanosec) + "." +
  //    img_msg->encoding);
  // ofs.write(reinterpret_cast<const char*>(img_msg->data.data()),
  //   img_msg->data.size());

  auto tp_start = std::chrono::system_clock::now();
  auto dnn_output = std::make_shared<DnnExampleOutput>();
  // 1. 将图片处理成模型输入数据类型DNNInput
  // 使用图片生成pym，NV12PyramidInput为DNNInput的子类
  std::shared_ptr<hobot::dnn_node::NV12PyramidInput> pyramid = nullptr;
  if ("rgb8" == img_msg->encoding) {
    auto cv_img =
        cv_bridge::cvtColorForDisplay(cv_bridge::toCvShare(img_msg), "bgr8");
    // dump recved img msg after convert
    // cv::imwrite("dump_raw_" +
    //     std::to_string(img_msg->header.stamp.sec) + "." +
    //     std::to_string(img_msg->header.stamp.nanosec) + ".jpg",
    //     cv_img->image);

    {
      auto tp_now = std::chrono::system_clock::now();
      auto interval = std::chrono::duration_cast<std::chrono::milliseconds>(
                          tp_now - tp_start)
                          .count();
      RCLCPP_DEBUG(rclcpp::get_logger("example"),
                   "after cvtColorForDisplay cost ms: %d",
                   interval);
    }
    pyramid = hobot::dnn_node::ImageProc::GetNV12PyramidFromBGRImg(
        cv_img->image, model_input_height_, model_input_width_);
  } else if ("bgr8" == img_msg->encoding) {
    auto cv_img =
        cv_bridge::cvtColorForDisplay(cv_bridge::toCvShare(img_msg), "bgr8");
    // dump recved img msg after convert
    // cv::imwrite("dump_raw_" +
    //     std::to_string(img_msg->header.stamp.sec) + "." +
    //     std::to_string(img_msg->header.stamp.nanosec) + ".jpg",
    //     cv_img->image);

    {
      auto tp_now = std::chrono::system_clock::now();
      auto interval = std::chrono::duration_cast<std::chrono::milliseconds>(
                          tp_now - tp_start)
                          .count();
      RCLCPP_DEBUG(rclcpp::get_logger("example"),
                   "after cvtColorForDisplay cost ms: %d",
                   interval);
    }
    pyramid = hobot::dnn_node::ImageProc::GetNV12PyramidFromBGRImg(
        cv_img->image, model_input_height_, model_input_width_);
  } else if ("nv12" == img_msg->encoding) {  // nv12格式使用hobotcv resize
    if (img_msg->height != static_cast<uint32_t>(model_input_height_) ||
        img_msg->width != static_cast<uint32_t>(model_input_width_)) {
      // 需要做resize处理
      cv::Mat out_img;
      if (ResizeNV12Img(reinterpret_cast<const char *>(img_msg->data.data()),
                        img_msg->height,
                        img_msg->width,
                        model_input_height_,
                        model_input_width_,
                        out_img,
                        dnn_output->ratio) < 0) {
        RCLCPP_ERROR(rclcpp::get_logger("dnn_node_example"),
                     "Resize nv12 img fail!");
        return;
      }

      uint32_t out_img_width = out_img.cols;
      uint32_t out_img_height = out_img.rows * 2 / 3;
      pyramid = hobot::dnn_node::ImageProc::GetNV12PyramidFromNV12Img(
          reinterpret_cast<const char *>(out_img.data),
          out_img_height,
          out_img_width,
          model_input_height_,
          model_input_width_);
    } else {  //不需要进行resize
      pyramid = hobot::dnn_node::ImageProc::GetNV12PyramidFromNV12Img(
          reinterpret_cast<const char *>(img_msg->data.data()),
          img_msg->height,
          img_msg->width,
          model_input_height_,
          model_input_width_);
    }
  }

  if (!pyramid) {
    RCLCPP_ERROR(rclcpp::get_logger("example"), "Get Nv12 pym fail");
    return;
  }

  {
    auto tp_now = std::chrono::system_clock::now();
    auto interval =
        std::chrono::duration_cast<std::chrono::milliseconds>(tp_now - tp_start)
            .count();
    RCLCPP_DEBUG(rclcpp::get_logger("example"),
                 "after GetNV12Pyramid cost ms: %d",
                 interval);
  }

  // 2. 使用pyramid创建DNNInput对象inputs
  // inputs将会作为模型的输入通过RunInferTask接口传入
  auto inputs = std::vector<std::shared_ptr<DNNInput>>{pyramid};

  // 3. 初始化输出
  if (parser == DnnParserType::UNET_PARSER) {
    dnn_output->img_w = img_msg->width;
    dnn_output->img_h = img_msg->height;
    dnn_output->model_w = model_input_width_;
    dnn_output->model_h = model_input_height_;
  }
  dnn_output->msg_header = std::make_shared<std_msgs::msg::Header>();
  dnn_output->msg_header->set__frame_id(img_msg->header.frame_id);
  dnn_output->msg_header->set__stamp(img_msg->header.stamp);

  if (dump_render_img_) {
    dnn_output->pyramid = pyramid;
  }

  // 4. 开始预测
  if (Run(inputs, dnn_output, nullptr) != 0) {
    RCLCPP_INFO(rclcpp::get_logger("example"), "Run predict failed!");
    return;
  }
}

#ifdef SHARED_MEM_ENABLED
void DnnExampleNode::SharedMemImgProcess(
    const hbm_img_msgs::msg::HbmMsg1080P::ConstSharedPtr img_msg) {
  if (!img_msg) {
    return;
  }

  if (!rclcpp::ok()) {
    return;
  }

  struct timespec time_start = {0, 0};
  clock_gettime(CLOCK_REALTIME, &time_start);

  std::stringstream ss;
  ss << "Recved img encoding: "
     << std::string(reinterpret_cast<const char *>(img_msg->encoding.data()))
     << ", h: " << img_msg->height << ", w: " << img_msg->width
     << ", step: " << img_msg->step << ", index: " << img_msg->index
     << ", stamp: " << img_msg->time_stamp.sec << "_"
     << img_msg->time_stamp.nanosec << ", data size: " << img_msg->data_size;
  RCLCPP_INFO(rclcpp::get_logger("example"), "%s", ss.str().c_str());

  rclcpp::Time msg_ts = img_msg->time_stamp;
  rclcpp::Duration dura = this->now() - msg_ts;
  float duration_ms = dura.nanoseconds() / 1000.0 / 1000.0;
  RCLCPP_WARN_THROTTLE(this->get_logger(),
    *this->get_clock(), 3000,
    "%s, comm delay [%.4f]ms",
    ss.str().c_str(), duration_ms);

  auto tp_start = std::chrono::system_clock::now();

  // 1. 将图片处理成模型输入数据类型DNNInput
  // 使用图片生成pym，NV12PyramidInput为DNNInput的子类
  std::shared_ptr<hobot::dnn_node::NV12PyramidInput> pyramid = nullptr;
  auto dnn_output = std::make_shared<DnnExampleOutput>();
  if ("nv12" ==
      std::string(reinterpret_cast<const char *>(img_msg->encoding.data()))) {
    if (img_msg->height != static_cast<uint32_t>(model_input_height_) ||
        img_msg->width != static_cast<uint32_t>(model_input_width_)) {
      // 需要做resize处理
      cv::Mat out_img;
      if (ResizeNV12Img(reinterpret_cast<const char *>(img_msg->data.data()),
                        img_msg->height,
                        img_msg->width,
                        model_input_height_,
                        model_input_width_,
                        out_img,
                        dnn_output->ratio) < 0) {
        RCLCPP_ERROR(rclcpp::get_logger("dnn_node_example"),
                     "Resize nv12 img fail!");
        return;
      }

      uint32_t out_img_width = out_img.cols;
      uint32_t out_img_height = out_img.rows * 2 / 3;
      pyramid = hobot::dnn_node::ImageProc::GetNV12PyramidFromNV12Img(
          reinterpret_cast<const char *>(out_img.data),
          out_img_height,
          out_img_width,
          model_input_height_,
          model_input_width_);
    } else {
      //不需要进行resize
      pyramid = hobot::dnn_node::ImageProc::GetNV12PyramidFromNV12Img(
          reinterpret_cast<const char *>(img_msg->data.data()),
          img_msg->height,
          img_msg->width,
          model_input_height_,
          model_input_width_);
    }
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("example"),
                 "Unsupported img encoding: %s, only nv12 img encoding is "
                 "supported for shared mem.",
                 img_msg->encoding.data());
    return;
  }

  // 如果运行的是unet算法，设置后处理需要的参数
  if (parser == DnnParserType::UNET_PARSER) {
    dnn_output->img_w = img_msg->width;
    dnn_output->img_h = img_msg->height;
    dnn_output->model_w = model_input_width_;
    dnn_output->model_h = model_input_height_;
  }

  // 生成pyramid数据失败
  if (!pyramid) {
    RCLCPP_ERROR(rclcpp::get_logger("example"), "Get Nv12 pym fail");
    return;
  }

  {
    auto tp_now = std::chrono::system_clock::now();
    auto interval =
        std::chrono::duration_cast<std::chrono::milliseconds>(tp_now - tp_start)
            .count();
    RCLCPP_DEBUG(rclcpp::get_logger("example"),
                 "after GetNV12Pyramid cost ms: %d",
                 interval);
  }

  // 2. 使用pyramid创建DNNInput对象inputs
  // inputs将会作为模型的输入通过RunInferTask接口传入
  auto inputs = std::vector<std::shared_ptr<DNNInput>>{pyramid};
  // 使用订阅到的msg配置msg_header
  dnn_output->msg_header = std::make_shared<std_msgs::msg::Header>();
  dnn_output->msg_header->set__frame_id(std::to_string(img_msg->index));
  dnn_output->msg_header->set__stamp(img_msg->time_stamp);

  // 如果开启了本地渲染功能，缓存pyramid数据
  if (dump_render_img_) {
    dnn_output->pyramid = pyramid;
  }

  // 更新前处理的perf信息
  dnn_output->preprocess_timespec_start = time_start;
  struct timespec time_now = {0, 0};
  clock_gettime(CLOCK_REALTIME, &time_now);
  dnn_output->preprocess_timespec_end = time_now;

  // 3. 开始预测
  if (Run(inputs, dnn_output, nullptr) != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("example"), "Run predict failed!");
    return;
  }

  {
    auto tp_now = std::chrono::system_clock::now();
    auto interval =
        std::chrono::duration_cast<std::chrono::milliseconds>(tp_now - tp_start)
            .count();
    RCLCPP_DEBUG(
        rclcpp::get_logger("example"), "after Predict cost ms: %d", interval);
  }
}
#endif
