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
#include "dnn_node/util/output_parser/detection/ultralytics_yolo_output_parser.h"

#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <queue>
#include <thread>

#include "rapidjson/document.h"
#include "rclcpp/rclcpp.hpp"

#include "dnn_node/util/output_parser/detection/nms.h"
#include "dnn_node/util/output_parser/neon_utils.h"
#include "dnn_node/util/output_parser/utils.h"

namespace hobot {
namespace dnn_node {
namespace parser_ultralytics_yolo {

inline float fastExp(float x) {
  union {
    uint32_t i;
    float f;
  } v;
  v.i = (12102203.1616540672f * x + 1064807160.56887296f);
  return v.f;
}

/**
 * Config definition for Ultralytics YOLO (v5-v26)
 */
struct UltralyticsYoloConfig {
  std::vector<int> strides;
  int class_num;
  int reg_max;
  std::vector<std::string> class_names;
  std::vector<int> output_order;

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

UltralyticsYoloConfig default_yolo_config = {
    {8, 16, 32}, 80, 16,
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

UltralyticsYoloConfig yolo_config_ = default_yolo_config;
float score_threshold_ = 0.4;
static bool is_performance_ = true;
float nms_threshold_ = 0.5;
int nms_top_k_ = 5000;

int InitClassNum(const int &class_num) {
  if(class_num > 0){
    yolo_config_.class_num = class_num;
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("ultralytics_yolo_parser"),
                 "class_num = %d is not allowed, only support class_num > 0",
                 class_num);
    return -1;
  }
  return 0;
}

int InitClassNames(const std::string &cls_name_file) {
  std::ifstream fi(cls_name_file);
  if (fi) {
    yolo_config_.class_names.clear();
    std::string line;
    while (std::getline(fi, line)) {
      yolo_config_.class_names.push_back(line);
    }
    int size = yolo_config_.class_names.size();
    if(size != yolo_config_.class_num){
      RCLCPP_ERROR(rclcpp::get_logger("ultralytics_yolo_parser"),
                 "class_names length %d is not equal to class_num %d",
                 size, yolo_config_.class_num);
      return -1;
    }
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("ultralytics_yolo_parser"),
                 "can not open cls name file: %s",
                 cls_name_file.c_str());
    return -1;
  }
  return 0;
}

int InitRegMax(const int &reg_max) {
  if(reg_max > 0){
    yolo_config_.reg_max = reg_max;
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("ultralytics_yolo_parser"),
                 "reg_max = %d is not allowed, only support class_num > 0",
                 reg_max);
    return -1;
  }
  return 0;
}

int InitStrides(const std::vector<int> &strides, const int &model_output_count){
  int size = strides.size();
  if(size * 2 != model_output_count){
    RCLCPP_ERROR(rclcpp::get_logger("ultralytics_yolo_parser"),
                "strides size %d is not equal to model_output_count %d",
                size, model_output_count);
    return -1;
  }
  yolo_config_.strides.clear();
  for (size_t i = 0; i < strides.size(); i++){
    yolo_config_.strides.push_back(strides[i]);
  }
  return 0;
}

int LoadConfig(const rapidjson::Document &document) {
  int model_output_count = 0;
  if (document.HasMember("model_output_count")) {
    model_output_count = document["model_output_count"].GetInt();
    if (model_output_count <= 0){
      RCLCPP_ERROR(rclcpp::get_logger("ultralytics_yolo_parser"),
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

  score_threshold_ = -log(1 / score_threshold_ - 1);

  if (document.HasMember("nms_top_k")) {
    nms_top_k_ = document["nms_top_k"].GetInt();
  }
  if (document.HasMember("is_performance")) {
    is_performance_ = document["is_performance"].GetBool();
  }
  if (document.HasMember("output_order")) {
    yolo_config_.output_order.clear();
    for(size_t i = 0; i < document["output_order"].Size(); i++){
      yolo_config_.output_order.push_back(document["output_order"][i].GetInt());
    }
    if(InitOutputOrder(yolo_config_.output_order, model_output_count) < 0){
      return -1;
    }
  } else {
    for (int i = 0; i < model_output_count; i++) {
      yolo_config_.output_order.push_back(i);
    }
  }

  return 0;
}

int PostProcess(std::vector<std::shared_ptr<DNNTensor>> &output_tensors,
                Perception &perception);

void SortByOrder(std::vector<std::shared_ptr<DNNTensor>> &output_tensors,
                 std::vector<int> order);

void ParseTensor(std::shared_ptr<DNNTensor> clses,
                 std::shared_ptr<DNNTensor> boxes,
                 int layer,
                 std::vector<Detection> &dets) {
  clses->CACHE_INVALIDATE();
  boxes->CACHE_INVALIDATE();
  int num_classes = yolo_config_.class_num;
  int reg_max = yolo_config_.reg_max;
  int stride = yolo_config_.strides[layer];

  int height, width;
  auto ret = hobot::dnn_node::output_parser::get_tensor_hw(boxes, &height, &width);
  if (ret != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("ultralytics_yolo_parser"),
                 "get_tensor_hw failed");
    return;
  }

  float *cls_data = clses->GetTensorData<float>();
  float *box_data = boxes->GetTensorData<float>();

  // Pre-compute grid center coordinates
  std::vector<float> col_center(width);
  std::vector<float> row_center(height);
  for (int w = 0; w < width; ++w) {
    col_center[w] = (static_cast<float>(w) + 0.5f) * stride;
  }
  for (int h = 0; h < height; ++h) {
    row_center[h] = (static_cast<float>(h) + 0.5f) * stride;
  }

  int det_count = 0;
  dets.reserve(dets.size() + height * width / 20);

  for (int h = 0; h < height; ++h) {
    float gc_y = row_center[h];
    for (int w = 0; w < width; ++w) {
      float *cur_cls_data = cls_data;
      float *cur_box_data = box_data;

      cls_data += num_classes;
      box_data += reg_max * 4;

      // Inline argmax: faster than std::max_element + std::distance on ARM
      float max_logit = cur_cls_data[0];
      int id = 0;
      for (int c = 1; c < num_classes; ++c) {
        if (cur_cls_data[c] > max_logit) {
          max_logit = cur_cls_data[c];
          id = c;
        }
      }

      if (max_logit < score_threshold_) {
        continue;
      }

      float confidence = 1.0f / (1.0f + std::exp(-max_logit));

      // Stack-based decoded boxes (avoid heap allocation per cell)
      float decoded_boxes[4];
      if (reg_max == 1) {
        std::memcpy(decoded_boxes, cur_box_data, 4 * sizeof(float));
      } else {
        decoded_boxes[0] = decoded_boxes[1] =
            decoded_boxes[2] = decoded_boxes[3] = 0.0f;
        size_t box_id = 0;
        for (size_t i = 0; i < 4; ++i) {
          float sum = 0.0f;
          for (int reg = 0; reg < reg_max; ++reg) {
            float distribute_score;
            if (is_performance_) {
              distribute_score = fastExp(cur_box_data[box_id]);
            } else {
              distribute_score = std::exp(cur_box_data[box_id]);
            }
            sum += distribute_score;
            decoded_boxes[i] += distribute_score * static_cast<float>(reg);
            ++box_id;
          }
          decoded_boxes[i] /= sum;
        }
      }

      float gc_x = col_center[w];
      float xmin = gc_x - decoded_boxes[0] * stride;
      float ymin = gc_y - decoded_boxes[1] * stride;
      float xmax = gc_x + decoded_boxes[2] * stride;
      float ymax = gc_y + decoded_boxes[3] * stride;

      if (xmax <= 0 || ymax <= 0) continue;
      if (xmin > xmax || ymin > ymax) continue;

      Bbox bbox(xmin, ymin, xmax, ymax);
      dets.emplace_back(
          static_cast<int>(id),
          confidence,
          bbox,
          yolo_config_.class_names[static_cast<int>(id)].c_str());
      det_count++;
    }
  }
  RCLCPP_DEBUG(rclcpp::get_logger("ultralytics_yolo_parser"),
              "ParseTensor layer=%d detections=%d", layer, det_count);
}

int32_t Parse(
    const std::shared_ptr<hobot::dnn_node::DnnNodeOutput> &node_output,
    std::shared_ptr<DnnParserResult> &result) {
  if (!result) {
    result = std::make_shared<DnnParserResult>();
  }
  SortByOrder(node_output->output_tensors, yolo_config_.output_order);
  int ret = PostProcess(node_output->output_tensors,
                        result->perception);
  if (ret != 0) {
    RCLCPP_INFO(rclcpp::get_logger("ultralytics_yolo_parser"),
                "postprocess return error, code = %d",
                ret);
  }

  std::stringstream ss;
  ss << "ultralytics_yolo_parser parse finished, predict result: "
     << result->perception;
  RCLCPP_DEBUG(
      rclcpp::get_logger("ultralytics_yolo_parser"), "%s", ss.str().c_str());
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
      ParseTensor(output_tensors[i * 2],
                  output_tensors[i * 2 + 1],
                  static_cast<int>(i), _dets);
      if (!_dets.empty()) {
        sp_det = std::make_shared<std::vector<Detection>>(_dets);
      }
      return sp_det;
    });
    futs.push_back(std::move(fut));
  }
  for (size_t i = 0; i < futs.size(); i++) {
    if (!futs[i].valid()) {
      RCLCPP_ERROR(rclcpp::get_logger("ultralytics_yolo_parser"),
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
  int parse_tensor_time_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - ts_start)
          .count();
  ts_start = std::chrono::steady_clock::now();

  nms(dets, nms_threshold_, nms_top_k_, perception.det, false);

  int nms_time_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - ts_start)
          .count();

  RCLCPP_INFO(rclcpp::get_logger("ultralytics_yolo_parser"),
              "PostProcess timing: output_tensors=%zu dets=%zu "
              "parse=%dus nms=%dus",
              output_tensors.size(), dets.size(),
              parse_tensor_time_us, nms_time_us);

  return 0;
}

void SortByOrder(std::vector<std::shared_ptr<DNNTensor>> &outputs,
                 std::vector<int> order){
  std::vector<std::shared_ptr<DNNTensor>> outputs_sorted(outputs.size());
  for(size_t i = 0; i < outputs.size(); i++){
    outputs_sorted[i] = outputs[order[i]];
  }
  outputs = outputs_sorted;
}

}  // namespace parser_ultralytics_yolo
}  // namespace dnn_node
}  // namespace hobot
