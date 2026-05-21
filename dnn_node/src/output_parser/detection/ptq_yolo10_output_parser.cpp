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
#include "dnn_node/util/output_parser/detection/ptq_yolo10_output_parser.h"

#include <arm_neon.h>

#include <iostream>
#include <queue>
#include <fstream>
#include <future>
#include <numeric>
#include <algorithm>

#include "rapidjson/document.h"
#include "rclcpp/rclcpp.hpp"

#include "dnn_node/util/output_parser/detection/nms.h"
#include "dnn_node/util/output_parser/utils.h"


namespace hobot {
namespace dnn_node {
namespace parser_yolov10 {


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
 * Config definition for Yolo10
 */
struct PTQYolo10Config {
  std::vector<int> strides;
  int class_num;
  int reg_max;
  std::vector<std::string> class_names;
  std::vector<std::vector<float>> dequantize_scale;
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

PTQYolo10Config default_ptq_yolo10_config = {
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

PTQYolo10Config yolo10_config_ = default_ptq_yolo10_config;
float score_threshold_ = 0.4;
static bool is_performance_ = true;
int top_k_ = 300;

int InitClassNum(const int &class_num) {
  if(class_num > 0){
    yolo10_config_.class_num = class_num;
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo10_detection_parser"),
                 "class_num = %d is not allowed, only support class_num > 0",
                 class_num);
    return -1;
  }
  return 0;
}

int InitClassNames(const std::string &cls_name_file) {
  std::ifstream fi(cls_name_file);
  if (fi) {
    yolo10_config_.class_names.clear();
    std::string line;
    while (std::getline(fi, line)) {
      yolo10_config_.class_names.push_back(line);
    }
    int size = yolo10_config_.class_names.size();
    if(size != yolo10_config_.class_num){
      RCLCPP_ERROR(rclcpp::get_logger("Yolo10_detection_parser"),
                 "class_names length %d is not equal to class_num %d",
                 size, yolo10_config_.class_num);
      return -1;
    }
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo10_detection_parser"),
                 "can not open cls name file: %s",
                 cls_name_file.c_str());
    return -1;
  }
  return 0;
}

int InitRegMax(const int &reg_max) {
  if(reg_max > 0){
    yolo10_config_.reg_max = reg_max;
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo10_detection_parser"),
                 "reg_max = %d is not allowed, only support class_num > 0",
                 reg_max);
    return -1;
  }
  return 0;
}


int InitStrides(const std::vector<int> &strides, const int &model_output_count){
  int size = strides.size();
  if(size * 2 != model_output_count){
    RCLCPP_ERROR(rclcpp::get_logger("yolo10_detection_parser"),
                "strides size %d is not equal to model_output_count %d",
                size, model_output_count);
    return -1;
  }
  yolo10_config_.strides.clear();
  for (size_t i = 0; i < strides.size(); i++){
    yolo10_config_.strides.push_back(strides[i]);
  }
  return 0;
}

int LoadConfig(const rapidjson::Document &document) {
  int model_output_count = 0;
  if (document.HasMember("model_output_count")) {
    model_output_count = document["model_output_count"].GetInt();
    if (model_output_count <= 0){
      RCLCPP_ERROR(rclcpp::get_logger("Yolo10_detection_parser"),
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

  score_threshold_ = -log(1 / score_threshold_ - 1);

  if (document.HasMember("top_k")) {
    top_k_ = document["top_k"].GetInt();
  }
  if (document.HasMember("is_performance")) {
    is_performance_ = document["is_performance"].GetBool();
  }
  if (document.HasMember("output_order")) {
    for(size_t i = 0; i < document["output_order"].Size(); i++){
      yolo10_config_.output_order.push_back(document["output_order"][i].GetInt());
    }
    if(InitOutputOrder(yolo10_config_.output_order, model_output_count) < 0){
      return -1;
    }
  } else {
    for (int i = 0; i < model_output_count; i++) {
      yolo10_config_.output_order.push_back(i);
    }
  }
  return 0;
}

int PostProcess(std::vector<std::shared_ptr<DNNTensor>> &output_tensors,
                Perception &perception);

float DequantiScale(int32_t data,
                    bool big_endian,
                    float &scale_value);

void SortByOrder(std::vector<std::shared_ptr<DNNTensor>> &output_tensors,std::vector<int> order);

void ParseTensor(std::shared_ptr<DNNTensor> clses,
                 std::shared_ptr<DNNTensor> boxes,
                 int layer,
                 std::vector<Detection> &dets) {
  clses->CACHE_INVALIDATE();
  boxes->CACHE_INVALIDATE();
  int num_classes = yolo10_config_.class_num;
  int reg_max = yolo10_config_.reg_max;
  int stride = yolo10_config_.strides[layer];

  std::vector<float> class_pred(yolo10_config_.class_num, 0.0);
  int height, width;
  auto ret =
      hobot::dnn_node::output_parser::get_tensor_hw(boxes, &height, &width);
  if (ret != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("yolo10_detection_parser"),
                 "get_tensor_hw failed");
  }

  float *cls_data = clses->GetTensorData<float>();
  int32_t *box_data = boxes->GetTensorData<int32_t>();
  auto *box_scale_data = reinterpret_cast<float *>(boxes->properties.scale.scaleData);
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      float *cur_cls_data = cls_data;
      int32_t *cur_box_data = box_data;

      cls_data += num_classes;
      box_data += reg_max * 4;

      int id = argmax(cur_cls_data, cur_cls_data + num_classes);

      if (cur_cls_data[id] < score_threshold_) {
        continue;
      }
      
      double confidence = 1 / (1 + std::exp(-cur_cls_data[id]));
      float sum, distribute_score;
      size_t box_id = 0;
      std::vector<float> decoded_boxes(4, 0);
      for (size_t i = 0; i < 4; ++i) {
        sum = 0.;
        for (int reg = 0; reg < reg_max; ++reg) {
          if (is_performance_) {
            distribute_score = fastExp(DequantiScale(cur_box_data[box_id], false, box_scale_data[box_id]));
          } else {
            distribute_score = std::exp(DequantiScale(cur_box_data[box_id], false, box_scale_data[box_id]));
          }
          sum += distribute_score;
          decoded_boxes[i] += distribute_score * reg;
          ++box_id;
        }
        decoded_boxes[i] /= sum;
      }

      float xmin = (w + 0.5 - decoded_boxes[0]) * stride;
      float ymin = (h + 0.5 - decoded_boxes[1]) * stride;
      float xmax = (w + 0.5 + decoded_boxes[2]) * stride;
      float ymax = (h + 0.5 + decoded_boxes[3]) * stride;

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
          yolo10_config_.class_names[static_cast<int>(id)].c_str());
      
    }
  }
}

int32_t Parse(
    const std::shared_ptr<hobot::dnn_node::DnnNodeOutput> &node_output, 
    std::shared_ptr<DnnParserResult> &result) {
  if (!result) {
    result = std::make_shared<DnnParserResult>();
  }

  SortByOrder(node_output->output_tensors,yolo10_config_.output_order);

  int ret = PostProcess(node_output->output_tensors, 
                        result->perception);
  if (ret != 0) {
    RCLCPP_INFO(rclcpp::get_logger("Yolo10_detection_parser"),
                "postprocess return error, code = %d",
                ret);
  }

  std::stringstream ss;
  ss << "Yolo10_detection_parser parse finished, predict result: "
     << result->perception;
  RCLCPP_DEBUG(
      rclcpp::get_logger("Yolo10_detection_parser"), "%s", ss.str().c_str());
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
      RCLCPP_DEBUG_STREAM(rclcpp::get_logger("yolo10_detection_parser"),
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
      RCLCPP_ERROR(rclcpp::get_logger("yolo10_detection_parser"),
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

  std::vector<size_t> index(static_cast<int>(dets.size()));
  std::iota(index.begin(), index.end(), 0); 
  std::sort(index.begin(), index.end(), [&](const auto l, const auto r){
    return dets[l].score > dets[r].score;
  });

  for (size_t i = 0; i < static_cast<int>(top_k_) && i < dets.size(); ++i) {
    perception.det.emplace_back(dets[index[i]]);
  }

  int sort_time_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - ts_start)
          .count();

  RCLCPP_DEBUG_STREAM(rclcpp::get_logger("yolo10_detection_parser"),
                   "output_tensors size: "
                   << output_tensors.size()
                   << ", parse_tensor_time_ms [" << parse_tensor_time_ms
                   << "] sort_time_ms [" << sort_time_ms << "]"
                   );

  return 0;
}


float DequantiScale(int32_t data,
                    bool big_endian,
                    float &scale_value) {
  return static_cast<float>(r_int32(data, big_endian)) * scale_value;
}

void SortByOrder(std::vector<std::shared_ptr<DNNTensor>> &outputs,std::vector<int> order){
  std::vector<std::shared_ptr<DNNTensor>>  outputs_sorted(outputs.size());
  for(int i = 0; i < outputs.size(); i++){
    outputs_sorted[i] = outputs[order[i]];
  }
  outputs = outputs_sorted;
}

}  // namespace parser_yolov10
}  // namespace dnn_node
}  // namespace hobot
