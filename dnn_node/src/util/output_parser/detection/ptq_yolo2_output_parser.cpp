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

#include "dnn_node/util/output_parser/detection/ptq_yolo2_output_parser.h"

#include <queue>
#include <fstream>

#include "dnn_node/util/output_parser/detection/nms.h"
#include "dnn_node/util/output_parser/utils.h"
#include "rclcpp/rclcpp.hpp"

namespace hobot {
namespace dnn_node {
namespace parser_yolov2 {

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
 * Config definition for Yolo2
 */
struct PTQYolo2Config {
  int stride;
  std::vector<std::pair<double, double>> anchors_table;
  int class_num;
  std::vector<std::string> class_names;

  std::string Str() {
    std::stringstream ss;
    ss << "stride: " << stride;
    ss << "; anchors_table: ";
    for (auto anchors : anchors_table) {
      ss << "[" << anchors.first << "," << anchors.second << "] ";
    }
    ss << "; class_num: " << class_num;
    return ss.str();
  }
};

PTQYolo2Config default_ptq_yolo2_config = {
    32,
    {{0.57273, 0.677385},
     {1.87446, 2.06253},
     {3.33843, 5.47434},
     {7.88282, 3.52778},
     {9.77052, 9.16828}},
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

int PostProcess(std::vector<std::shared_ptr<DNNTensor>> &output_tensors,
                Perception &perception);

int PostProcessQuantiSCALE(std::vector<std::shared_ptr<DNNTensor>> &output_tensors,
                Perception &perception);
                
PTQYolo2Config yolo2_config_ = default_ptq_yolo2_config;
float score_threshold_ = 0.3;
float nms_threshold_ = 0.45;
int nms_top_k_ = 500;

int InitClassNum(const int &class_num) {
  if(class_num > 0){
    yolo2_config_.class_num = class_num;
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo2_detection_parser"),
                 "class_num = %d is not allowed, only support class_num > 0",
                 class_num);
    return -1;
  }
  return 0;
}

int InitClassNames(const std::string &cls_name_file) {
  std::ifstream fi(cls_name_file);
  if (fi) {
    yolo2_config_.class_names.clear();
    std::string line;
    while (std::getline(fi, line)) {
      yolo2_config_.class_names.push_back(line);
    }
    int size = yolo2_config_.class_names.size();
    if(size != yolo2_config_.class_num){
      RCLCPP_ERROR(rclcpp::get_logger("Yolo2_detection_parser"),
                 "class_names length %d is not equal to class_num %d",
                 size, yolo2_config_.class_num);
      return -1;
    }
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo2_detection_parser"),
                 "can not open cls name file: %s",
                 cls_name_file.c_str());
    return -1;
  }
  return 0;
}

int InitStride(const int &stride){
  yolo2_config_.stride = stride;
  return 0;
}

int InitAnchorsTables(const std::vector<std::vector<double>> &anchors_tables){
  yolo2_config_.anchors_table.clear();
  for (size_t i = 0; i < anchors_tables.size(); i++){
    if(anchors_tables[i].size() != 2){
      RCLCPP_ERROR(rclcpp::get_logger("Yolo2_detection_parser"),
                  "anchors_tables[%zu] size is not equal to 2", i);
      return -1;
    }
    std::pair<double, double> table;
    table.first = anchors_tables[i][0];
    table.second = anchors_tables[i][1];
    yolo2_config_.anchors_table.push_back(table);
  }
  return 0;
}

int LoadConfig(const rapidjson::Document &document) {
  int model_output_count = 0;
  if (document.HasMember("model_output_count")) {
    model_output_count = document["model_output_count"].GetInt();
    if (model_output_count <= 0){
      RCLCPP_ERROR(rclcpp::get_logger("Yolo2_detection_parser"),
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
  if (document.HasMember("stride")) {
    InitStride(document["stride"].GetInt());
  }
  if (document.HasMember("anchors_table")) {
    std::vector<std::vector<double>> anchors_tables;
    for(size_t i = 0; i < document["anchors_table"].Size(); i++){
      std::vector<double> anchors_table;
      for(size_t j = 0; j < document["anchors_table"][i].Size(); j++){
        anchors_table.push_back(document["anchors_table"][i][j].GetDouble());
      }
      anchors_tables.push_back(anchors_table);
    }
    if (InitAnchorsTables(anchors_tables) < 0){
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

int32_t Parse(
    const std::shared_ptr<hobot::dnn_node::DnnNodeOutput> &node_output,
    std::shared_ptr<DnnParserResult> &result) {
  if (!result) {
    result = std::make_shared<DnnParserResult>();
  }

  auto quanti_type = node_output->output_tensors[0]->properties.quantiType;
  int ret = 0;
  if (quanti_type == hbDNNQuantiType::SCALE) {
    ret = PostProcessQuantiSCALE(node_output->output_tensors, result->perception);
  } else if (quanti_type == hbDNNQuantiType::NONE) {
    ret = PostProcess(node_output->output_tensors, result->perception);
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo2_detection_parser"),
            "error quanti_type: %d", quanti_type);
    return -1;
  }
  if (ret != 0) {
    RCLCPP_INFO(rclcpp::get_logger("Yolo2_detection_parser"),
                "postprocess return error, code = %d",
                ret);
  }
  std::stringstream ss;
  ss << "Yolo2_detection_parser parse finished, predict result: "
     << result->perception;
  RCLCPP_DEBUG(
      rclcpp::get_logger("Yolo2_detection_parser"), "%s", ss.str().c_str());
  return ret;
}

int PostProcess(std::vector<std::shared_ptr<DNNTensor>> &tensors,
                Perception &perception) {
  perception.type = Perception::DET;
  hbSysFlushMem(&(tensors[0]->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  auto *data = reinterpret_cast<float *>(tensors[0]->sysMem[0].virAddr);
  auto &anchors_table = yolo2_config_.anchors_table;
  int num_classes = yolo2_config_.class_num;
  float stride = static_cast<float>(yolo2_config_.stride);
  int num_pred = num_classes + 4 + 1;
  std::vector<Detection> dets;
  std::vector<float> class_pred(num_classes, 0.0);

  int height, width;
  hobot::dnn_node::output_parser::get_tensor_hw(tensors[0], &height, &width);
  // int *shape = tensor->data_shape.d;
  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      for (size_t k = 0; k < anchors_table.size(); k++) {
        double anchor_x = anchors_table[k].first;
        double anchor_y = anchors_table[k].second;
        float *cur_data = data + k * num_pred;

        float objness = cur_data[4];
        for (int index = 0; index < num_classes; ++index) {
          class_pred[index] = cur_data[5 + index];
        }

        float id = argmax(class_pred.begin(), class_pred.end());

        float confidence = (1.f / (1 + std::exp(-objness))) *
                           (1.f / (1 + std::exp(-class_pred[id])));

        if (confidence < score_threshold_) {
          continue;
        }

        float center_x = cur_data[0];
        float center_y = cur_data[1];
        float scale_x = cur_data[2];
        float scale_y = cur_data[3];

        double box_center_x =
            ((1.0 / (1.0 + std::exp(-center_x))) + w) * stride;
        double box_center_y =
            ((1.0 / (1.0 + std::exp(-center_y))) + h) * stride;

        double box_scale_x = std::exp(scale_x) * anchor_x * stride;
        double box_scale_y = std::exp(scale_y) * anchor_y * stride;

        double xmin = (box_center_x - box_scale_x / 2.0);
        double ymin = (box_center_y - box_scale_y / 2.0);
        double xmax = (box_center_x + box_scale_x / 2.0);
        double ymax = (box_center_y + box_scale_y / 2.0);

        if (xmin > xmax || ymin > ymax) {
          continue;
        }

        Bbox bbox(xmin, ymin, xmax, ymax);
        dets.emplace_back(
            Detection(static_cast<int>(id),
                      confidence,
                      bbox,
                      yolo2_config_.class_names[static_cast<int>(id)].c_str()));
      }
      data = data + num_pred * anchors_table.size();
    }
  }

  nms(dets, nms_threshold_, nms_top_k_, perception.det, false);
  return 0;
}

float DequantiScale(int32_t data,
                    bool big_endian,
                    float &scale_value) {
  return static_cast<float>(r_int32(data, big_endian)) * scale_value;
}

int PostProcessQuantiSCALE(std::vector<std::shared_ptr<DNNTensor>> &tensors,
                          Perception &perception) {
  perception.type = Perception::DET;
  hbSysFlushMem(&(tensors[0]->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  int32_t *data = reinterpret_cast<int32_t *>(tensors[0]->sysMem[0].virAddr);

  float *scale = tensors[0]->properties.scale.scaleData;

  auto &anchors_table = yolo2_config_.anchors_table;
  int num_classes = yolo2_config_.class_num;
  float stride = static_cast<float>(yolo2_config_.stride);
  int num_pred = num_classes + 4 + 1;
  std::vector<Detection> dets;
  std::vector<float> class_pred(num_classes, 0.0);

  int height = tensors[0]->properties.validShape.dimensionSize[1];
  int width = tensors[0]->properties.validShape.dimensionSize[2];
  int channel_valid = tensors[0]->properties.validShape.dimensionSize[3];
  int channel_aligned = tensors[0]->properties.alignedShape.dimensionSize[3];

  for (uint32_t h = 0; h < height; h++) {
    for (uint32_t w = 0; w < width; w++) {
      for (int k = 0; k < anchors_table.size(); k++) {
        double anchor_x = anchors_table[k].first;
        double anchor_y = anchors_table[k].second;

        int32_t *cur_data = data + k * num_pred;
        float *cur_scale = scale + k * num_pred;
        float objness = DequantiScale(cur_data[4], false, cur_scale[4]);

        for (int index = 0; index < num_classes; ++index) {
          class_pred[index] =
              DequantiScale(cur_data[5 + index], false, cur_scale[5 + index]);
        }

        float id = argmax(class_pred.begin(), class_pred.end());

        float confidence = (1.f / (1 + std::exp(-objness))) *
                           (1.f / (1 + std::exp(-class_pred[id])));

        if (confidence < score_threshold_) {
          continue;
        }

        float center_x = DequantiScale(cur_data[0], false, cur_scale[0]);
        float center_y = DequantiScale(cur_data[1], false, cur_scale[1]);
        float scale_x = DequantiScale(cur_data[2], false, cur_scale[2]);
        float scale_y = DequantiScale(cur_data[3], false, cur_scale[3]);

        double box_center_x =
            ((1.0 / (1.0 + std::exp(-center_x))) + w) * stride;
        double box_center_y =
            ((1.0 / (1.0 + std::exp(-center_y))) + h) * stride;

        double box_scale_x = std::exp(scale_x) * anchor_x * stride;
        double box_scale_y = std::exp(scale_y) * anchor_y * stride;

        double xmin = (box_center_x - box_scale_x / 2.0);
        double ymin = (box_center_y - box_scale_y / 2.0);
        double xmax = (box_center_x + box_scale_x / 2.0);
        double ymax = (box_center_y + box_scale_y / 2.0);

        if (xmin > xmax || ymin > ymax) {
          continue;
        }

        Bbox bbox(xmin, ymin, xmax, ymax);
        dets.emplace_back(
            Detection((int)id,
                      confidence,
                      bbox,
                      yolo2_config_.class_names[(int)id].c_str()));
      }
      data = data + channel_aligned;
    }
  }
  nms(dets, nms_threshold_, nms_top_k_, perception.det, false);
  return 0;
}

}  // namespace parser_yolov2
}  // namespace dnn_node
}  // namespace hobot
