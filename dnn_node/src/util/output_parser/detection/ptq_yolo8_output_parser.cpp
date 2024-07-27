// Copyright (c) 2024，D-Robotics.
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
#include "dnn_node/util/output_parser/detection/ptq_yolo8_output_parser.h"

#include <arm_neon.h>

#include <iostream>
#include <queue>
#include <fstream>
#include <future>

#include "rapidjson/document.h"
#include "rclcpp/rclcpp.hpp"

#include "dnn_node/util/output_parser/utils.h"
#include "dnn_node/util/output_parser/detection/nms.h"

namespace hobot {
namespace dnn_node {
namespace parser_yolov8 {


inline float fastExp(float x) {
  union {
    uint32_t i;
    float f;
  } v;
  v.i = (12102203.1616540672f * x + 1064807160.56887296f);
  return v.f;
}


/**
 * Finds the greatest element in the range [first, last)
 * @tparam[in] ForwardIterator: iterator type
 * @param[in] first: fist iterator
 * @param[in] last: last iterator
 * @return Iterator to the greatest element in the range [first, last)
 */
template <class ForwardIterator>
inline size_t argmax(ForwardIterator first, ForwardIterator last) {
  return std::distance(first, std::max_element(first, last));
}

#define BSWAP_32(x) static_cast<int32_t>(__builtin_bswap32(x))

#define r_int32(x, big_endian) \
  (big_endian) ? BSWAP_32((x)) : static_cast<int32_t>((x))

/**
 * Config definition for Yolo8
 */
struct PTQYolo8Config {
  std::vector<int> strides;
  int class_num;
  int reg_max;
  std::vector<std::string> class_names;
  std::vector<std::vector<float>> dequantize_scale;

  std::string Str() {
    std::stringstream ss;
    ss << "strides: ";
    for (const auto &stride : strides) {
      ss << stride << " ";
    }

    ss << "; class_num: " << class_num;
    ss << "; reg_max: " << reg_max;
    return ss.str();
  }
};

PTQYolo8Config default_ptq_yolo8_config = {
    {8, 16, 32},
    80,
    16,
    {"person",        "bicycle",      "car",
     "motorcycle",    "airplane",     "bus",
     "train",         "truck",        "boat",
     "traffic light", "fire hydrant", "stop sign",
     "parking meter", "bench",        "bird",
     "cat",           "dog",          "horse",
     "sheep",         "cow",          "elephant",
     "bear",          "zebra",        "giraffe",
     "backpack",      "umbrella",     "handbag",
     "tie",           "suitcase",     "frisbee",
     "skis",          "snowboard",    "sports ball",
     "kite",          "baseball bat", "baseball glove",
     "skateboard",    "surfboard",    "tennis racket",
     "bottle",        "wine glass",   "cup",
     "fork",          "knife",        "spoon",
     "bowl",          "banana",       "apple",
     "sandwich",      "orange",       "broccoli",
     "carrot",        "hot dog",      "pizza",
     "donut",         "cake",         "chair",
     "couch",         "potted plant", "bed",
     "dining table",  "toilet",       "tv",
     "laptop",        "mouse",        "remote",
     "keyboard",      "cell phone",   "microwave",
     "oven",          "toaster",      "sink",
     "refrigerator",  "book",         "clock",
     "vase",          "scissors",     "teddy bear",
     "hair drier",    "toothbrush"}};

PTQYolo8Config yolo8_config_ = default_ptq_yolo8_config;
float score_threshold_ = 0.4;
static bool is_performance_ = true;
float nms_threshold_ = 0.5;
int nms_top_k_ = 5000;

int InitClassNum(const int &class_num) {
  if(class_num > 0){
    yolo8_config_.class_num = class_num;
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo8_detection_parser"),
                 "class_num = %d is not allowed, only support class_num > 0",
                 class_num);
    return -1;
  }
  return 0;
}

int InitClassNames(const std::string &cls_name_file) {
  std::ifstream fi(cls_name_file);
  if (fi) {
    yolo8_config_.class_names.clear();
    std::string line;
    while (std::getline(fi, line)) {
      yolo8_config_.class_names.push_back(line);
    }
    int size = yolo8_config_.class_names.size();
    if(size != yolo8_config_.class_num){
      RCLCPP_ERROR(rclcpp::get_logger("Yolo8_detection_parser"),
                 "class_names length %d is not equal to class_num %d",
                 size, yolo8_config_.class_num);
      return -1;
    }
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo8_detection_parser"),
                 "can not open cls name file: %s",
                 cls_name_file.c_str());
    return -1;
  }
  return 0;
}

int InitRegMax(const int &reg_max) {
  if(reg_max > 0){
    yolo8_config_.reg_max = reg_max;
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo8_detection_parser"),
                 "reg_max = %d is not allowed, only support class_num > 0",
                 reg_max);
    return -1;
  }
  return 0;
}


int InitStrides(const std::vector<int> &strides, const int &model_output_count){
  int size = strides.size();
  if(size != model_output_count){
    RCLCPP_ERROR(rclcpp::get_logger("yolo8_detection_parser"),
                "strides size %d is not equal to model_output_count %d",
                size, model_output_count);
    return -1;
  }
  yolo8_config_.strides.clear();
  for (size_t i = 0; i < strides.size(); i++){
    yolo8_config_.strides.push_back(strides[i]);
  }
  return 0;
}


int LoadConfig(const rapidjson::Document &document) {
  int model_output_count = 0;
  if (document.HasMember("model_output_count")) {
    model_output_count = document["model_output_count"].GetInt();
    if (model_output_count <= 0){
      RCLCPP_ERROR(rclcpp::get_logger("Yolo8_detection_parser"),
              "model_output_count = %d <= 0 is not allowed", model_output_count);
      return -1;
    }
  }
  if (document.HasMember("class_num")){
    int class_num = document["class_num"].GetInt();
    if (InitClassNum(class_num) < 0) {
      return -1;
    }
  } 
  if (document.HasMember("cls_names_list")) {
    std::string cls_name_file = document["cls_names_list"].GetString();
    if (InitClassNames(cls_name_file) < 0) {
      return -1;
    }
  }
  if (document.HasMember("reg_max")){
    int reg_max = document["reg_max"].GetInt();
    if (InitRegMax(reg_max) < 0) {
      return -1;
    }
  } 
  if (document.HasMember("strides")) {
    std::vector<int> strides;
    for(size_t i = 0; i < document["strides"].Size(); i++){
      strides.push_back(document["strides"][i].GetInt());
    }
    if (InitStrides(strides, model_output_count) < 0){
      return -1;
    }
  }
  if (document.HasMember("score_threshold")) {
    score_threshold_ = document["score_threshold"].GetFloat();
  }
  if (document.HasMember("nms_threshold")) {
    nms_threshold_ = document["nms_threshold"].GetFloat();
  }
  if (document.HasMember("nms_top_k")) {
    nms_top_k_ = document["nms_top_k"].GetInt();
  }

  return 0;
}

int PostProcess(std::vector<std::shared_ptr<DNNTensor>> &output_tensors,
                Perception &perception);

double Dequanti(int32_t data,
                int layer,
                bool big_endian,
                int offset,
                hbDNNTensorProperties &properties);



void ParseTensor(std::shared_ptr<DNNTensor> clses,
                 std::shared_ptr<DNNTensor> boxes,
                 int layer,
                 std::vector<Detection> &dets) {
  hbSysFlushMem(&(clses->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  hbSysFlushMem(&(boxes->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  int num_classes = yolo8_config_.class_num;
  int reg_max = yolo8_config_.reg_max;
  int stride = yolo8_config_.strides[layer];

  std::vector<float> class_pred(yolo8_config_.class_num, 0.0);
  int height, width;
  auto ret =
      hobot::dnn_node::output_parser::get_tensor_hw(boxes, &height, &width);
  if (ret != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("yolo8_detection_parser"),
                 "get_tensor_hw failed");
  }

  auto *cls_data = reinterpret_cast<float *>(clses->sysMem[0].virAddr);
  auto *box_data = reinterpret_cast<float *>(boxes->sysMem[0].virAddr);
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      float *cur_cls_data = cls_data;
      float *cur_box_data = box_data;

      cls_data += num_classes;
      box_data += reg_max * 4;

      int id = argmax(cur_cls_data, cur_cls_data + num_classes);
      double confidence = 1 / (1 + std::exp(-cur_cls_data[id]));

      if (confidence < score_threshold_) {
        continue;
      }
      
      std::vector<double> decoded_boxes(4, 0);
      for (int i = 0; i < 4; ++i) {
        double sum = 0;
        for (int reg = 0; reg < reg_max; ++reg) {
          double distribute_score;
          if (is_performance_) {
            distribute_score = fastExp(cur_box_data[i * reg_max + reg]);
          } else {
            distribute_score = std::exp(cur_box_data[i * reg_max + reg]);
          }
          sum += distribute_score;
          decoded_boxes[i] += distribute_score * reg;
        }
        decoded_boxes[i] /= sum;
      }

      double xmin = (w + 0.5 - decoded_boxes[0]) * stride;
      double ymin = (h + 0.5 - decoded_boxes[1]) * stride;
      double xmax = (w + 0.5 + decoded_boxes[2]) * stride;
      double ymax = (h + 0.5 + decoded_boxes[3]) * stride;

      if (xmax <= 0 || ymax <= 0) {
        continue;
      }

      if (xmin > xmax || ymin > ymax) {
        continue;
      }

      Bbox bbox(xmin, ymin, xmax, ymax);
      dets.emplace_back(
          static_cast<int>(id),
          confidence,
          bbox,
          yolo8_config_.class_names[static_cast<int>(id)].c_str());
      
    }
  }
}

int32_t Parse(
    const std::shared_ptr<hobot::dnn_node::DnnNodeOutput> &node_output, 
    std::shared_ptr<DnnParserResult> &result) {
  if (!result) {
    result = std::make_shared<DnnParserResult>();
  }

  int ret = PostProcess(node_output->output_tensors, 
                        result->perception);
  if (ret != 0) {
    RCLCPP_INFO(rclcpp::get_logger("Yolo8_detection_parser"),
                "postprocess return error, code = %d",
                ret);
  }

  std::stringstream ss;
  ss << "Yolo8_detection_parser parse finished, predict result: "
     << result->perception;
  RCLCPP_DEBUG(
      rclcpp::get_logger("Yolo8_detection_parser"), "%s", ss.str().c_str());
  return ret;
}


int PostProcess(std::vector<std::shared_ptr<DNNTensor>> &output_tensors,
                Perception &perception) {
  perception.type = Perception::DET;
  std::vector<Detection> dets;

  auto ts_start = std::chrono::steady_clock::now();
  std::vector<std::future<std::shared_ptr<std::vector<Detection>>>> futs;
  auto output_size = output_tensors.size() / 2;
  for (size_t i = 0; i < output_size; i++) {
    auto fut = std::async(std::launch::async, [&output_tensors, i](){
      std::shared_ptr<std::vector<Detection>> sp_det = nullptr;
      std::vector<Detection> _dets;
      auto start = std::chrono::steady_clock::now();
      ParseTensor(output_tensors[i * 2], 
                  output_tensors[i * 2 + 1], 
                  static_cast<int>(i), _dets);
      int time_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - start)
              .count();
      RCLCPP_DEBUG_STREAM(rclcpp::get_logger("yolo8_detection_parser"),
                      "parse tensor "
                      << i
                      << " cost [" << time_ms << "]"
                      );
      if (!_dets.empty()) {
        sp_det = std::make_shared<std::vector<Detection>>(_dets);
      }
      return sp_det;
    });
    futs.push_back(std::move(fut));
  }
  for (size_t i = 0; i < futs.size(); i++) {
    if (!futs[i].valid()) {
      RCLCPP_ERROR(rclcpp::get_logger("yolo8_detection_parser"),
                  "fut is not valid");
      return -1;
    }
    futs[i].wait();
    auto det = futs[i].get();
    if (det) {
      dets.insert(dets.end(), std::make_move_iterator(det->begin()),
                  std::make_move_iterator(det->end()));
    }
  }
  int parse_tensor_time_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - ts_start)
          .count();
  ts_start = std::chrono::steady_clock::now();

  nms(dets, nms_threshold_, nms_top_k_, perception.det, false);
  
  int nms_time_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - ts_start)
          .count();

  RCLCPP_DEBUG_STREAM(rclcpp::get_logger("yolo8_detection_parser"),
                   "output_tensors size: "
                   << output_tensors.size()
                   << ", parse_tensor_time_ms [" << parse_tensor_time_ms
                   << "] nms_time_ms [" << nms_time_ms << "]"
                   );

  return 0;
}

double Dequanti(int32_t data,
                int layer,
                bool big_endian,
                int offset,
                hbDNNTensorProperties &properties) {
  return static_cast<double>(r_int32(data, big_endian)) *
         yolo8_config_.dequantize_scale[layer][offset];
}

}  // namespace parser_yolov8
}  // namespace dnn_node
}  // namespace hobot
