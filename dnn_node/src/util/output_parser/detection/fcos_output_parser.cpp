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

#include "dnn_node/util/output_parser/detection/fcos_output_parser.h"

#include <iostream>
#include <queue>
#include <fstream>

#include "dnn_node/util/output_parser/detection/nms.h"
#include "dnn_node/util/output_parser/utils.h"
#include "rclcpp/rclcpp.hpp"

namespace hobot {
namespace dnn_node {
namespace parser_fcos {

struct ScoreId {
  float score;
  int id;
};

// FcosConfig
struct FcosConfig {
  std::vector<int> strides;
  int class_num;
  std::vector<std::string> class_names;
  std::string det_name_list;
};

FcosConfig default_fcos_config = {
    {{8, 16, 32, 64, 128}},
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
     "hair drier",    "toothbrush"},
    ""};

void GetBboxAndScoresNHWC(std::vector<std::shared_ptr<DNNTensor>> &tensors,
                          std::vector<Detection> &dets);

void GetBboxAndScoresNCHW(std::vector<std::shared_ptr<DNNTensor>> &tensors,
                          std::vector<Detection> &dets);

int PostProcess(std::vector<std::shared_ptr<DNNTensor>> &tensors,
                Perception &perception);

FcosConfig fcos_config_ = default_fcos_config;
float score_threshold_ = 0.5;
float nms_threshold_ = 0.6;
int nms_top_k_ = 500;

int InitClassNum(const int &class_num) {
  if(class_num > 0){
    fcos_config_.class_num = class_num;
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("fcos_detection_parser"),
                 "class_num = %d is not allowed, only support class_num > 0",
                 class_num);
    return -1;
  }
  return 0;
}

int InitClassNames(const std::string &cls_name_file) {
  std::ifstream fi(cls_name_file);
  if (fi) {
    fcos_config_.class_names.clear();
    std::string line;
    while (std::getline(fi, line)) {
      fcos_config_.class_names.push_back(line);
    }
    int size = fcos_config_.class_names.size();
    if(size != fcos_config_.class_num){
      RCLCPP_ERROR(rclcpp::get_logger("fcos_detection_parser"),
                 "class_names length %d is not equal to class_num %d",
                 size, fcos_config_.class_num);
      return -1;
    }
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("fcos_detection_parser"),
                 "can not open cls name file: %s",
                 cls_name_file.c_str());
    return -1;
  }
  return 0;
}

int InitStrides(const std::vector<int> &strides, const int &model_output_count){
  int size = strides.size() * 3;
  if(size != model_output_count){
    RCLCPP_ERROR(rclcpp::get_logger("fcos_detection_parser"),
                "strides size %d is not realated to model_output_count %d",
                size, model_output_count);
    return -1;
  }
  fcos_config_.strides.clear();
  for (size_t i = 0; i < strides.size(); i++){
    fcos_config_.strides.push_back(strides[i]);
  }
  return 0;
}

int LoadConfig(const rapidjson::Document &document) {
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
    RCLCPP_INFO(rclcpp::get_logger("fcos_detection_parser"),
                "postprocess return error, code = %d",
                ret);
  }

  return ret;
}

void GetBboxAndScoresNHWC(std::vector<std::shared_ptr<DNNTensor>> &tensors,
                          std::vector<Detection> &dets) {
  // fcos stride is {8, 16, 32, 64, 128}
  for (size_t i = 0; i < fcos_config_.strides.size(); ++i) {
    auto *cls_data = reinterpret_cast<float *>(tensors[i]->sysMem[0].virAddr);
    auto *bbox_data =
        reinterpret_cast<float *>(tensors[i + 5]->sysMem[0].virAddr);
    auto *ce_data =
        reinterpret_cast<float *>(tensors[i + 10]->sysMem[0].virAddr);

    // 同一个尺度下，tensor[i],tensor[i+5],tensor[i+10]出来的hw都一致，64*64/32*32/...
    int *shape = tensors[i]->properties.alignedShape.dimensionSize;
    int tensor_h = shape[1];
    int tensor_w = shape[2];
    int tensor_c = shape[3];

    for (int h = 0; h < tensor_h; h++) {
      int offset = h * tensor_w;
      for (int w = 0; w < tensor_w; w++) {
        // get score
        int ce_offset = offset + w;
        ce_data[ce_offset] = 1.0 / (1.0 + exp(-ce_data[ce_offset]));

        int cls_offset = ce_offset * tensor_c;
        ScoreId tmp_score = {cls_data[cls_offset], 0};
        for (int cls_c = 1; cls_c < tensor_c; cls_c++) {
          int cls_index = cls_offset + cls_c;
          if (cls_data[cls_index] > tmp_score.score) {
            tmp_score.id = cls_c;
            tmp_score.score = cls_data[cls_index];
          }
        }
        tmp_score.score = 1.0 / (1.0 + exp(-tmp_score.score));
        tmp_score.score = std::sqrt(tmp_score.score * ce_data[ce_offset]);
        if (tmp_score.score <= score_threshold_) continue;

        // get detection box
        int index = 4 * (h * tensor_w + w);
        double xmin = ((w + 0.5) * fcos_config_.strides[i] - bbox_data[index]);
        double ymin =
            ((h + 0.5) * fcos_config_.strides[i] - bbox_data[index + 1]);
        double xmax =
            ((w + 0.5) * fcos_config_.strides[i] + bbox_data[index + 2]);
        double ymax =
            ((h + 0.5) * fcos_config_.strides[i] + bbox_data[index + 3]);

        Detection detection;
        detection.bbox.xmin = xmin;
        detection.bbox.ymin = ymin;
        detection.bbox.xmax = xmax;
        detection.bbox.ymax = ymax;
        detection.score = tmp_score.score;
        detection.id = tmp_score.id;
        detection.class_name = fcos_config_.class_names[tmp_score.id].c_str();
        dets.push_back(detection);
      }
    }
  }
}

void GetBboxAndScoresNCHW(std::vector<std::shared_ptr<DNNTensor>> &tensors,
                          std::vector<Detection> &dets) {
  auto &strides = fcos_config_.strides;
  for (size_t i = 0; i < strides.size(); ++i) {
    auto *cls_data = reinterpret_cast<float *>(tensors[i]->sysMem[0].virAddr);
    auto *bbox_data =
        reinterpret_cast<float *>(tensors[i + 5]->sysMem[0].virAddr);
    auto *ce_data =
        reinterpret_cast<float *>(tensors[i + 10]->sysMem[0].virAddr);

    // 同一个尺度下，tensor[i],tensor[i+5],tensor[i+10]出来的hw都一致，64*64/32*32/...
    int *shape = tensors[i]->properties.alignedShape.dimensionSize;
    int tensor_c = shape[1];
    int tensor_h = shape[2];
    int tensor_w = shape[3];
    int aligned_hw = tensor_h * tensor_w;

    for (int h = 0; h < tensor_h; h++) {
      int offset = h * tensor_w;
      for (int w = 0; w < tensor_w; w++) {
        // get score
        int ce_offset = offset + w;
        ce_data[ce_offset] = 1.0 / (1.0 + exp(-ce_data[ce_offset]));

        ScoreId tmp_score = {cls_data[offset + w], 0};
        for (int cls_c = 1; cls_c < tensor_c; cls_c++) {
          int cls_index = cls_c * aligned_hw + offset + w;
          if (cls_data[cls_index] > tmp_score.score) {
            tmp_score.id = cls_c;
            tmp_score.score = cls_data[cls_index];
          }
        }
        tmp_score.score = 1.0 / (1.0 + exp(-tmp_score.score));
        tmp_score.score = std::sqrt(tmp_score.score * ce_data[ce_offset]);
        if (tmp_score.score <= score_threshold_) continue;

        // get detection box
        int index = 4 * (h * tensor_w + w);
        double xmin = ((w + 0.5) * fcos_config_.strides[i] - bbox_data[index]);
        double ymin =
            ((h + 0.5) * fcos_config_.strides[i] - bbox_data[index + 1]);
        double xmax =
            ((w + 0.5) * fcos_config_.strides[i] + bbox_data[index + 2]);
        double ymax =
            ((h + 0.5) * fcos_config_.strides[i] + bbox_data[index + 3]);

        Detection detection;
        detection.bbox.xmin = xmin;
        detection.bbox.ymin = ymin;
        detection.bbox.xmax = xmax;
        detection.bbox.ymax = ymax;
        detection.score = tmp_score.score;
        detection.id = tmp_score.id;
        detection.class_name = fcos_config_.class_names[tmp_score.id].c_str();
        dets.push_back(detection);
      }
    }
  }
}

int PostProcess(std::vector<std::shared_ptr<DNNTensor>> &tensors,
                Perception &perception) {
  if (!tensors[0]) {
    RCLCPP_INFO(rclcpp::get_logger("fcos_example"), "tensor layout error.");
    return -1;
  }

  int h_index, w_index, c_index;
  int ret = hobot::dnn_node::output_parser::get_tensor_hwc_index(
      tensors[0], &h_index, &w_index, &c_index);
  if (ret != 0 &&
      static_cast<int32_t>(fcos_config_.class_names.size()) !=
          tensors[0]->properties.alignedShape.dimensionSize[c_index]) {
    RCLCPP_INFO(rclcpp::get_logger("fcos_detection_parser"),
                "User det_name_list in config file: %s, is not compatible with "
                "this model, %zu  %d",
                fcos_config_.det_name_list.c_str(),
                fcos_config_.class_names.size(),
                tensors[0]->properties.alignedShape.dimensionSize[c_index]);
  }
  for (size_t i = 0; i < tensors.size(); i++) {
    if (!tensors[i]) {
      RCLCPP_INFO(rclcpp::get_logger("fcos_example"),
                  "tensor layout null, error.");
      return -1;
    }
    hbSysFlushMem(&(tensors[i]->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  }

  std::vector<std::vector<ScoreId>> scores;
  std::vector<Detection> dets;
  if (tensors[0]->properties.tensorLayout == HB_DNN_LAYOUT_NHWC) {
    GetBboxAndScoresNHWC(tensors, dets);
  } else if (tensors[0]->properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
    GetBboxAndScoresNCHW(tensors, dets);
  } else {
    RCLCPP_INFO(rclcpp::get_logger("fcos_example"), "tensor layout error.");
  }

  yolo5_nms(dets, nms_threshold_, nms_top_k_, perception.det, false);
  return 0;
}

}  // namespace parser_fcos
}  // namespace dnn_node
}  // namespace hobot
