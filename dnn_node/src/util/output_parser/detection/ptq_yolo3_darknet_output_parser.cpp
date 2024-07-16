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

#include "dnn_node/util/output_parser/detection/ptq_yolo3_darknet_output_parser.h"

#include <queue>
#include <fstream>

#include "dnn_node/util/output_parser/detection/nms.h"
#include "dnn_node/util/output_parser/utils.h"
#include "rclcpp/rclcpp.hpp"

namespace hobot {
namespace dnn_node {
namespace parser_yolov3 {

#define BSWAP_32(x) static_cast<int32_t>(__builtin_bswap32(x))

#define r_int32(x, big_endian) \
  (big_endian) ? BSWAP_32((x)) : static_cast<int32_t>((x))

float DequantiScale(int32_t data,
                                                      bool big_endian,
                                                      float &scale_value) {
  return static_cast<float>(r_int32(data, big_endian)) * scale_value;
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

/**
 * Config definition for Yolo3
 */
struct PTQYolo3DarknetConfig {
  std::vector<int> strides;
  std::vector<std::vector<std::pair<double, double>>> anchors_table;
  int class_num;
  std::vector<std::string> class_names;

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

PTQYolo3DarknetConfig default_ptq_yolo3_darknet_config = {
    {32, 16, 8},
    {{{3.625, 2.8125}, {4.875, 6.1875}, {11.65625, 10.1875}},
     {{1.875, 3.8125}, {3.875, 2.8125}, {3.6875, 7.4375}},
     {{1.25, 1.625}, {2.0, 3.75}, {4.125, 2.875}}},
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

void PostProcessNHWC(std::shared_ptr<DNNTensor> tensor,
                     int layer,
                     std::vector<Detection> &dets);

void PostProcessNCHW(std::shared_ptr<DNNTensor> tensor,
                     int layer,
                     std::vector<Detection> &dets);

void PostProcessQuantiScaleNHWC(std::shared_ptr<DNNTensor> tensor,
                                int layer,
                                std::vector<Detection> &dets);

PTQYolo3DarknetConfig yolo3_config_ = default_ptq_yolo3_darknet_config;
float score_threshold_ = 0.3;
float nms_threshold_ = 0.45;
int nms_top_k_ = 500;

int InitClassNum(const int &class_num) {
  if(class_num > 0){
    yolo3_config_.class_num = class_num;
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo3Darknet_detection_parser"),
                 "class_num = %d is not allowed, only support class_num > 0",
                 class_num);
    return -1;
  }
  return 0;
}

int InitClassNames(const std::string &cls_name_file) {
  std::ifstream fi(cls_name_file);
  if (fi) {
    yolo3_config_.class_names.clear();
    std::string line;
    while (std::getline(fi, line)) {
      yolo3_config_.class_names.push_back(line);
    }
    int size = yolo3_config_.class_names.size();
    if(size != yolo3_config_.class_num){
      RCLCPP_ERROR(rclcpp::get_logger("Yolo3Darknet_detection_parser"),
                 "class_names length %d is not equal to class_num %d",
                 size, yolo3_config_.class_num);
      return -1;
    }
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo3Darknet_detection_parser"),
                 "can not open cls name file: %s",
                 cls_name_file.c_str());
    return -1;
  }
  return 0;
}

int InitStrides(const std::vector<int> &strides, const int &model_output_count){
  int size = strides.size();
  if(size != model_output_count){
    RCLCPP_ERROR(rclcpp::get_logger("Yolo3Darknet_detection_parser"),
                "strides size %d is not equal to model_output_count %d",
                size, model_output_count);
    return -1;
  }
  yolo3_config_.strides.clear();
  for (size_t i = 0; i < strides.size(); i++){
    yolo3_config_.strides.push_back(strides[i]);
  }
  return 0;
}

int InitAnchorsTables(const std::vector<std::vector<std::vector<double>>> &anchors_tables, 
                      const int &model_output_count){
  int size = anchors_tables.size();
  if(size != model_output_count){
    RCLCPP_ERROR(rclcpp::get_logger("Yolo3Darknet_detection_parser"),
                "anchors_tables size %d is not equal to model_output_count %d",
                size, model_output_count);
    return -1;
  }
  yolo3_config_.anchors_table.clear();
  for (size_t i = 0; i < anchors_tables.size(); i++){
    if(anchors_tables[i].size() != 3){
      RCLCPP_ERROR(rclcpp::get_logger("Yolo3Darknet_detection_parser"),
                  "anchors_tables[%d] size is not equal to 3", i);
      return -1;      
    }
    std::vector<std::pair<double, double>> tables;
    for (size_t j = 0; j < anchors_tables[i].size(); j++){
      if(anchors_tables[i][j].size() != 2){
        RCLCPP_ERROR(rclcpp::get_logger("Yolo3Darknet_detection_parser"),
                    "anchors_tables[%d][%d] size is not equal to 2", i, j);
        return -1;
      }
      std::pair<double, double> table;
      table.first = anchors_tables[i][j][0];
      table.second = anchors_tables[i][j][1];
      tables.push_back(table);
    }
    yolo3_config_.anchors_table.push_back(tables);
  }
  return 0;
}

int LoadConfig(const rapidjson::Document &document){
  int model_output_count = 0;
  if (document.HasMember("model_output_count")) {
    model_output_count = document["model_output_count"].GetInt();
    if (model_output_count <= 0){
      RCLCPP_ERROR(rclcpp::get_logger("Yolo3Darknet_detection_parser"),
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

int32_t Parse(
    const std::shared_ptr<hobot::dnn_node::DnnNodeOutput> &node_output,
    std::shared_ptr<DnnParserResult> &result) {
  if (!result) {
    result = std::make_shared<DnnParserResult>();
  }

  int ret = PostProcess(node_output->output_tensors, result->perception);
  if (ret != 0) {
    RCLCPP_INFO(rclcpp::get_logger("Yolo3Darknet_detection_parser"),
                "postprocess return error, code = %d",
                ret);
  }
  std::stringstream ss;
  ss << "PTQYolo3DarknetOutputParser parse finished, predict result: "
     << result->perception;
  RCLCPP_DEBUG(rclcpp::get_logger("Yolo3Darknet_detection_parser"),
               "%s",
               ss.str().c_str());
  return ret;
}

int PostProcess(std::vector<std::shared_ptr<DNNTensor>> &tensors,
                Perception &perception) {
  perception.type = Perception::DET;
  std::vector<Detection> dets;
  for (size_t i = 0; i < yolo3_config_.strides.size(); i++) {
    auto quanti_type = tensors[i]->properties.quantiType;
    if (quanti_type == hbDNNQuantiType::NONE) {
      if (tensors[i]->properties.tensorLayout == HB_DNN_LAYOUT_NHWC) {
        PostProcessNHWC(tensors[i], i, dets);
      } else if (tensors[i]->properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
        PostProcessNCHW(tensors[i], i, dets);
      } else {
        RCLCPP_ERROR(rclcpp::get_logger("dnn_ptq_yolo3"), "tensor layout error.");
      }
    } else if (quanti_type == hbDNNQuantiType::SCALE) {
      if (tensors[i]->properties.tensorLayout == HB_DNN_LAYOUT_NHWC) {
        PostProcessQuantiScaleNHWC(tensors[i], i, dets);
      } else {
        RCLCPP_ERROR(rclcpp::get_logger("dnn_ptq_yolo3"), "tensor layout error.");
      }
    } else {
      RCLCPP_ERROR(rclcpp::get_logger("dnn_ptq_yolo3"), "tensor quanti type error.");
    }
  }
  nms(dets, nms_threshold_, nms_top_k_, perception.det, false);
  return 0;
}

void PostProcessNHWC(std::shared_ptr<DNNTensor> tensor,
                     int layer,
                     std::vector<Detection> &dets) {
  hbSysFlushMem(&(tensor->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  auto *data = reinterpret_cast<float *>(tensor->sysMem[0].virAddr);
  int num_classes = yolo3_config_.class_num;
  int stride = yolo3_config_.strides[layer];
  int num_pred = yolo3_config_.class_num + 4 + 1;

  std::vector<float> class_pred(yolo3_config_.class_num, 0.0);
  std::vector<std::pair<double, double>> &anchors =
      yolo3_config_.anchors_table[layer];

  int height, width;
  auto ret =
      hobot::dnn_node::output_parser::get_tensor_hw(tensor, &height, &width);
  if (ret != 0) {
    RCLCPP_WARN(rclcpp::get_logger("dnn_ptq_yolo3"), "get_tensor_hw failed");
  }

  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      for (size_t k = 0; k < anchors.size(); k++) {
        double anchor_x = anchors[k].first;
        double anchor_y = anchors[k].second;
        float *cur_data = data + k * num_pred;
        float objness = cur_data[4];
        for (int index = 0; index < num_classes; ++index) {
          class_pred[index] = cur_data[5 + index];
        }

        float id = argmax(class_pred.begin(), class_pred.end());
        double x1 = 1 / (1 + std::exp(-objness)) * 1;
        double x2 = 1 / (1 + std::exp(-class_pred[id]));
        double confidence = x1 * x2;

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
        dets.push_back(
            Detection(static_cast<int>(id),
                      confidence,
                      bbox,
                      yolo3_config_.class_names[static_cast<int>(id)].c_str()));
      }
      data = data + num_pred * anchors.size();
    }
  }
}

void PostProcessNCHW(std::shared_ptr<DNNTensor> tensor,
                     int layer,
                     std::vector<Detection> &dets) {
  hbSysFlushMem(&(tensor->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  auto *data = reinterpret_cast<float *>(tensor->sysMem[0].virAddr);
  int num_classes = yolo3_config_.class_num;
  int stride = yolo3_config_.strides[layer];
  int num_pred = yolo3_config_.class_num + 4 + 1;

  std::vector<float> class_pred(yolo3_config_.class_num, 0.0);
  std::vector<std::pair<double, double>> &anchors =
      yolo3_config_.anchors_table[layer];

  int height, width;
  auto ret =
      hobot::dnn_node::output_parser::get_tensor_hw(tensor, &height, &width);
  if (ret != 0) {
    RCLCPP_WARN(rclcpp::get_logger("dnn_ptq_yolo3"), "get_tensor_hw failed");
  }
  int aligned_h = tensor->properties.validShape.dimensionSize[2];
  int aligned_w = tensor->properties.validShape.dimensionSize[3];
  int aligned_hw = aligned_h * aligned_w;

  for (size_t k = 0; k < anchors.size(); k++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        double anchor_x = anchors[k].first;
        double anchor_y = anchors[k].second;
        int stride_hw = h * aligned_w + w;

        float objness = data[(k * num_pred + 4) * aligned_hw + stride_hw];
        for (int index = 0; index < num_classes; ++index) {
          class_pred[index] =
              data[(k * num_pred + index + 5) * aligned_hw + stride_hw];
        }

        float id = argmax(class_pred.begin(), class_pred.end());
        double x1 = 1 / (1 + std::exp(-objness)) * 1;
        double x2 = 1 / (1 + std::exp(-class_pred[id]));
        double confidence = x1 * x2;

        if (confidence < score_threshold_) {
          continue;
        }

        float center_x = data[(k * num_pred) * aligned_hw + stride_hw];
        float center_y = data[(k * num_pred + 1) * aligned_hw + stride_hw];
        float scale_x = data[(k * num_pred + 2) * aligned_hw + stride_hw];
        float scale_y = data[(k * num_pred + 3) * aligned_hw + stride_hw];

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
        dets.push_back(
            Detection(static_cast<int>(id),
                      confidence,
                      bbox,
                      yolo3_config_.class_names[static_cast<int>(id)].c_str()));
      }
    }
  }
}

void PostProcessQuantiScaleNHWC(std::shared_ptr<DNNTensor> tensor,
                                int layer,
                                std::vector<Detection> &dets) {
  hbSysFlushMem(&(tensor->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  auto *data = reinterpret_cast<int32_t *>(tensor->sysMem[0].virAddr);
  float *scale = tensor->properties.scale.scaleData;
  int num_classes = yolo3_config_.class_num;
  int stride = yolo3_config_.strides[layer];
  int num_pred = yolo3_config_.class_num + 4 + 1;

  std::vector<float> class_pred(yolo3_config_.class_num, 0.0);
  std::vector<std::pair<double, double>> &anchors =
      yolo3_config_.anchors_table[layer];

  int height = tensor->properties.validShape.dimensionSize[1];
  int width = tensor->properties.validShape.dimensionSize[2];

  int channel_valid = tensor->properties.validShape.dimensionSize[3];
  int channel_aligned = tensor->properties.alignedShape.dimensionSize[3];

  for (uint32_t h = 0; h < height; h++) {
    for (uint32_t w = 0; w < width; w++) {
      for (int k = 0; k < anchors.size(); k++) {
        double anchor_x = anchors[k].first;
        double anchor_y = anchors[k].second;

        int32_t *cur_data = data + k * num_pred;
        float *cur_scale = scale + k * num_pred;
        float objness = DequantiScale(cur_data[4], false, cur_scale[4]);

        for (int index = 0; index < num_classes; ++index) {
          class_pred[index] =
              DequantiScale(cur_data[5 + index], false, cur_scale[5 + index]);
        }

        float id = argmax(class_pred.begin(), class_pred.end());
        double x1 = 1 / (1 + std::exp(-objness)) * 1;
        double x2 = 1 / (1 + std::exp(-class_pred[id]));
        double confidence = x1 * x2;

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
        dets.push_back(Detection((int)id,
                                 confidence,
                                 bbox,
                                 yolo3_config_.class_names[(int)id].c_str()));
      }
      data = data + channel_aligned;
    }
  }
}

}  // namespace parser_yolov3
}  // namespace dnn_node
}  // namespace hobot
