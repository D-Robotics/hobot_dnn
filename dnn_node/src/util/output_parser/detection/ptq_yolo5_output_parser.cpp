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
#include "dnn_node/util/output_parser/detection/ptq_yolo5_output_parser.h"

#include <arm_neon.h>

#include <iostream>
#include <queue>
#include <fstream>
#include <future>

#include "dnn_node/util/output_parser/detection/nms.h"
#include "dnn_node/util/output_parser/utils.h"
#include "rapidjson/document.h"
#include "rclcpp/rclcpp.hpp"

namespace hobot {
namespace dnn_node {
namespace parser_yolov5 {

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
 * Config definition for Yolo5
 */
struct PTQYolo5Config {
  std::vector<int> strides;
  std::vector<std::vector<std::pair<double, double>>> anchors_table;
  int class_num;
  std::vector<std::string> class_names;
  std::vector<std::vector<float>> dequantize_scale;

  std::string Str() {
    std::stringstream ss;
    ss << "strides: ";
    for (const auto &stride : strides) {
      ss << stride << " ";
    }

    ss << "; anchors_table: ";
    for (const auto &anchors : anchors_table) {
      for (auto data : anchors) {
        ss << "[" << data.first << "," << data.second << "] ";
      }
    }
    ss << "; class_num: " << class_num;
    return ss.str();
  }
};

PTQYolo5Config default_ptq_yolo5_config = {
    {8, 16, 32},
    {{{10, 13}, {16, 30}, {33, 23}},
     {{30, 61}, {62, 45}, {59, 119}},
     {{116, 90}, {156, 198}, {373, 326}}},
    80,
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

PTQYolo5Config yolo5_config_ = default_ptq_yolo5_config;
float score_threshold_ = 0.4;
float nms_threshold_ = 0.5;
int nms_top_k_ = 5000;

int InitClassNum(const int &class_num) {
  if(class_num > 0){
    yolo5_config_.class_num = class_num;
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo5_detection_parser"),
                 "class_num = %d is not allowed, only support class_num > 0",
                 class_num);
    return -1;
  }
  return 0;
}

int InitClassNames(const std::string &cls_name_file) {
  std::ifstream fi(cls_name_file);
  if (fi) {
    yolo5_config_.class_names.clear();
    std::string line;
    while (std::getline(fi, line)) {
      yolo5_config_.class_names.push_back(line);
    }
    int size = yolo5_config_.class_names.size();
    if(size != yolo5_config_.class_num){
      RCLCPP_ERROR(rclcpp::get_logger("Yolo5_detection_parser"),
                 "class_names length %d is not equal to class_num %d",
                 size, yolo5_config_.class_num);
      return -1;
    }
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo5_detection_parser"),
                 "can not open cls name file: %s",
                 cls_name_file.c_str());
    return -1;
  }
  return 0;
}

int InitStrides(const std::vector<int> &strides, const int &model_output_count){
  int size = strides.size();
  if(size != model_output_count){
    RCLCPP_ERROR(rclcpp::get_logger("Yolo5_detection_parser"),
                "strides size %d is not equal to model_output_count %d",
                size, model_output_count);
    return -1;
  }
  yolo5_config_.strides.clear();
  for (size_t i = 0; i < strides.size(); i++){
    yolo5_config_.strides.push_back(strides[i]);
  }
  return 0;
}

int InitAnchorsTables(const std::vector<std::vector<std::vector<double>>> &anchors_tables, 
                      const int &model_output_count){
  int size = anchors_tables.size();
  if(size != model_output_count){
    RCLCPP_ERROR(rclcpp::get_logger("Yolo5_detection_parser"),
                "anchors_tables size %d is not equal to model_output_count %d",
                size, model_output_count);
    return -1;
  }
  yolo5_config_.anchors_table.clear();
  for (size_t i = 0; i < anchors_tables.size(); i++){
    if(anchors_tables[i].size() != 3){
      RCLCPP_ERROR(rclcpp::get_logger("Yolo5_detection_parser"),
                  "anchors_tables[%d] size is not equal to 3", i);
      return -1;      
    }
    std::vector<std::pair<double, double>> tables;
    for (size_t j = 0; j < anchors_tables[i].size(); j++){
      if(anchors_tables[i][j].size() != 2){
        RCLCPP_ERROR(rclcpp::get_logger("Yolo5_detection_parser"),
                    "anchors_tables[%d][%d] size is not equal to 2", i, j);
        return -1;
      }
      std::pair<double, double> table;
      table.first = anchors_tables[i][j][0];
      table.second = anchors_tables[i][j][1];
      tables.push_back(table);
    }
    yolo5_config_.anchors_table.push_back(tables);
  }
  return 0;
}

int LoadConfig(const rapidjson::Document &document) {
  int model_output_count = 0;
  if (document.HasMember("model_output_count")) {
    model_output_count = document["model_output_count"].GetInt();
    if (model_output_count <= 0){
      RCLCPP_ERROR(rclcpp::get_logger("Yolo5_detection_parser"),
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
  if (document.HasMember("strides")) {
    std::vector<int> strides;
    for(size_t i = 0; i < document["strides"].Size(); i++){
      strides.push_back(document["strides"][i].GetInt());
    }
    if (InitStrides(strides, model_output_count) < 0){
      return -1;
    }
  }
  if (document.HasMember("anchors_table")) {
    std::vector<std::vector<std::vector<double>>> anchors_tables;
    for(size_t i = 0; i < document["anchors_table"].Size(); i++){
      std::vector<std::vector<double>> anchors_table;
      for(size_t j = 0; j < document["anchors_table"][i].Size(); j++){
        std::vector<double> table;
        for(size_t k = 0; k < document["anchors_table"][i][j].Size(); k++){
          table.push_back(document["anchors_table"][i][j][k].GetDouble());
        }
        anchors_table.push_back(table);
      }
      anchors_tables.push_back(anchors_table);
    }
    if (InitAnchorsTables(anchors_tables, model_output_count) < 0){
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

void ParseTensor(std::shared_ptr<DNNTensor> tensor,
                 int layer,
                 std::vector<Detection> &dets) {
  hbSysFlushMem(&(tensor->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  int num_classes = yolo5_config_.class_num;
  int stride = yolo5_config_.strides[layer];
  int num_pred = yolo5_config_.class_num + 4 + 1;

  std::vector<float> class_pred(yolo5_config_.class_num, 0.0);
  std::vector<std::pair<double, double>> &anchors =
      yolo5_config_.anchors_table[layer];

  //  int *shape = tensor->data_shape.d;
  int height, width;
  auto ret =
      hobot::dnn_node::output_parser::get_tensor_hw(tensor, &height, &width);
  if (ret != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo5_detection_parser"),
                 "get_tensor_hw failed");
  }

  int anchor_num = anchors.size();
  auto *data = reinterpret_cast<float *>(tensor->sysMem[0].virAddr);
  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      for (int k = 0; k < anchor_num; k++) {
        double anchor_x = anchors[k].first;
        double anchor_y = anchors[k].second;
        float *cur_data = data + k * num_pred;
        float objness = cur_data[4];

        int id = argmax(cur_data + 5, cur_data + 5 + num_classes);
        double x1 = 1 / (1 + std::exp(-objness)) * 1;
        double x2 = 1 / (1 + std::exp(-cur_data[id + 5]));
        double confidence = x1 * x2;

        if (confidence < score_threshold_) {
          continue;
        }

        float center_x = cur_data[0];
        float center_y = cur_data[1];
        float scale_x = cur_data[2];
        float scale_y = cur_data[3];

        double box_center_x =
            ((1.0 / (1.0 + std::exp(-center_x))) * 2 - 0.5 + w) * stride;
        double box_center_y =
            ((1.0 / (1.0 + std::exp(-center_y))) * 2 - 0.5 + h) * stride;

        double box_scale_x =
            std::pow((1.0 / (1.0 + std::exp(-scale_x))) * 2, 2) * anchor_x;
        double box_scale_y =
            std::pow((1.0 / (1.0 + std::exp(-scale_y))) * 2, 2) * anchor_y;

        double xmin = (box_center_x - box_scale_x / 2.0);
        double ymin = (box_center_y - box_scale_y / 2.0);
        double xmax = (box_center_x + box_scale_x / 2.0);
        double ymax = (box_center_y + box_scale_y / 2.0);

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
            yolo5_config_.class_names[static_cast<int>(id)].c_str());
      }
      data = data + num_pred * anchors.size();
    }
  }
}

int32_t Parse(
    const std::shared_ptr<hobot::dnn_node::DnnNodeOutput> &node_output,
    std::shared_ptr<DnnParserResult> &result) {
  if (!result) {
    result = std::make_shared<DnnParserResult>();
  }

  int ret = PostProcess(node_output->output_tensors, result->perception);
  if (ret != 0) {
    RCLCPP_INFO(rclcpp::get_logger("Yolo5_detection_parser"),
                "postprocess return error, code = %d",
                ret);
  }

  std::stringstream ss;
  ss << "Yolo5_detection_parser parse finished, predict result: "
     << result->perception;
  RCLCPP_DEBUG(
      rclcpp::get_logger("Yolo5_detection_parser"), "%s", ss.str().c_str());
  return ret;
}

int PostProcess(std::vector<std::shared_ptr<DNNTensor>> &output_tensors,
                Perception &perception) {
  perception.type = Perception::DET;
  std::vector<Detection> dets;

  auto ts_start = std::chrono::steady_clock::now();
  std::vector<std::future<std::shared_ptr<Detection>>> futs;
  for (size_t i = 0; i < output_tensors.size(); i++) {
    // ParseTensor(output_tensors[i], static_cast<int>(i), dets);

    auto fut = std::async(std::launch::async, [&output_tensors, i](){
      std::shared_ptr<Detection> sp_det = nullptr;
      std::vector<Detection> _dets;
      auto start = std::chrono::steady_clock::now();
      ParseTensor(output_tensors[i], static_cast<int>(i), _dets);
      int time_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - start)
              .count();
      RCLCPP_DEBUG_STREAM(rclcpp::get_logger("Yolo5_detection_parser"),
                      "parse tensor "
                      << i
                      << " cost [" << time_ms << "]"
                      );
      if (!_dets.empty()) {
        sp_det = std::make_shared<Detection>(_dets.front());
      }
      return sp_det;
    });
    
    futs.push_back(std::move(fut));
  }
  for (size_t i = 0; i < futs.size(); i++) {
    if (!futs[i].valid()) {
      RCLCPP_ERROR(rclcpp::get_logger("Yolo5_detection_parser"),
                  "fut is not valid");
      return -1;
    }
    futs[i].wait();
    auto det = futs[i].get();
    if (det) {
      dets.emplace_back(*det);
    }
  }

  int parse_tensor_time_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - ts_start)
          .count();
  ts_start = std::chrono::steady_clock::now();

  yolo5_nms(dets, nms_threshold_, nms_top_k_, perception.det, false);
  
  int nms_time_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - ts_start)
          .count();

  RCLCPP_DEBUG_STREAM(rclcpp::get_logger("Yolo5_detection_parser"),
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
         yolo5_config_.dequantize_scale[layer][offset];
}

}  // namespace parser_yolov5
}  // namespace dnn_node
}  // namespace hobot
