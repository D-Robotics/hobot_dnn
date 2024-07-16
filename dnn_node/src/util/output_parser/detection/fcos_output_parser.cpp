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

#include <arm_neon.h>

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

void CqatGetBboxAndScoresScaleNHWC(std::vector<std::shared_ptr<DNNTensor>> &tensors,
                          std::vector<Detection> &dets);

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
bool community_qat_ = false;

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
  if (document.HasMember("community_qat")) {
    community_qat_ = document["community_qat"].GetBool();
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

static inline uint32x4x4_t CalculateIndex(uint32_t idx,
                                          float32x4_t a,
                                          float32x4_t b,
                                          uint32x4x4_t c) {
  uint32x4_t mask{0};
  mask = vcltq_f32(b, a);
  uint32x4_t vec_idx = {idx, idx + 1, idx + 2, idx + 3};
  uint32x4x4_t res = {{vbslq_u32(mask, vec_idx, c.val[0]), 0, 0, 0}};
  return res;
}

static inline float32x2_t CalculateMax(float32x4_t in) {
  auto pmax = vpmax_f32(vget_high_f32(in), vget_low_f32(in));
  return vpmax_f32(pmax, pmax);
}

static inline uint32_t CalculateVectorIndex(uint32x4x4_t vec_res_idx,
                                            float32x4_t vec_res_value) {
  uint32x4_t res_idx_mask{0};
  uint32x4_t mask_ones = vdupq_n_u32(0xFFFFFFFF);

  auto pmax = CalculateMax(vec_res_value);
  auto mask = vceqq_f32(vec_res_value, vcombine_f32(pmax, pmax));
  res_idx_mask = vandq_u32(vec_res_idx.val[0], mask);
  res_idx_mask = vaddq_u32(res_idx_mask, mask_ones);
  auto pmin =
      vpmin_u32(vget_high_u32(res_idx_mask), vget_low_u32(res_idx_mask));
  pmin = vpmin_u32(pmin, pmin);
  uint32_t res = vget_lane_u32(pmin, 0);
  return (res - 0xFFFFFFFF);
}

static std::pair<float, int> MaxScoreID(int32_t *input,
                                        float *scale,
                                        int length) {
  float init_res_value = input[0] * scale[0];
  float32x4_t vec_res_value = vdupq_n_f32(init_res_value);
  uint32x4x4_t vec_res_idx{{0}};
  int i = 0;
  for (; i <= (length - 4); i += 4) {
    int32x4_t vec_input = vld1q_s32(input + i);
    float32x4_t vec_scale = vld1q_f32(scale + i);

    float32x4_t vec_elements = vmulq_f32(vcvtq_f32_s32(vec_input), vec_scale);
    float32x4_t temp_vec_res_value = vmaxq_f32(vec_elements, vec_res_value);
    vec_res_idx =
        CalculateIndex(i, temp_vec_res_value, vec_res_value, vec_res_idx);
    vec_res_value = temp_vec_res_value;
  }

  uint32_t idx = CalculateVectorIndex(vec_res_idx, vec_res_value);
  float res = vget_lane_f32(CalculateMax(vec_res_value), 0);

  // Compute left elements
  for (; i < length; ++i) {
    float score = input[i] * scale[i];
    if (score > res) {
      idx = i;
      res = score;
    }
  }
  std::pair<float, int> result_id_score = {res, idx};
  return result_id_score;
}

void CqatGetBboxAndScoresScaleNHWC(std::vector<std::shared_ptr<DNNTensor>> &tensors,
                          std::vector<Detection> &dets) {

  // fcos stride is {8, 16, 32, 64, 128}
  for (int i = 0; i < 5; i++) {
    auto *cls_data = reinterpret_cast<int32_t *>(tensors[i]->sysMem[0].virAddr);
    auto *bbox_data =
        reinterpret_cast<int32_t *>(tensors[i + 5]->sysMem[0].virAddr);
    auto *ce_data =
        reinterpret_cast<int32_t *>(tensors[i + 10]->sysMem[0].virAddr);

    float *cls_scale = tensors[i]->properties.scale.scaleData;
    float *bbox_scale = tensors[i + 5]->properties.scale.scaleData;
    float *ce_scale = tensors[i + 10]->properties.scale.scaleData;

    // 同一个尺度下，tensor[i],tensor[i+5],tensor[i+10]出来的hw都一致，64*64/32*32/...
    int *shape = tensors[i]->properties.alignedShape.dimensionSize;
    int tensor_h = shape[1];
    int tensor_w = shape[2];
    int tensor_c = shape[3];
    int32_t bbox_c_stride{
        tensors[i + 5]->properties.alignedShape.dimensionSize[3]};
    int32_t ce_c_stride{
        tensors[i + 10]->properties.alignedShape.dimensionSize[3]};

    for (int h = 0; h < tensor_h; h++) {
      for (int w = 0; w < tensor_w; w++) {

        // get score
        int ce_offset = (h * tensor_w + w) * ce_c_stride;
        float ce_data_offset =
            1.0 / (1.0 + exp(-ce_data[ce_offset] * ce_scale[0]));
        // argmax + neon
        int cls_offset = (h * tensor_w + w) * tensor_c;
        auto max_score_id =
            MaxScoreID(cls_data + cls_offset, cls_scale, tensor_c);

        // filter
        float cls_data_offset = 1.0 / (1.0 + exp(-max_score_id.first));
        float score = std::sqrt(cls_data_offset * ce_data_offset);

        if (score <= score_threshold_) continue;

        // get detection box
        Detection detection;
        int index = bbox_c_stride * (h * tensor_w + w);
        auto &strides = fcos_config_.strides;

        float xmin = std::max(0.f, bbox_data[index] * bbox_scale[0]);
        float ymin = std::max(0.f, bbox_data[index + 1] * bbox_scale[1]);
        float xmax = std::max(0.f, bbox_data[index + 2] * bbox_scale[2]);
        float ymax = std::max(0.f, bbox_data[index + 3] * bbox_scale[3]);

        detection.bbox.xmin = ((w + 0.5) - xmin) * strides[i];
        detection.bbox.ymin = ((h + 0.5) - ymin) * strides[i];
        detection.bbox.xmax = ((w + 0.5) + xmax) * strides[i];
        detection.bbox.ymax = ((h + 0.5) + ymax) * strides[i];

        detection.score = score;
        detection.id = max_score_id.second;
        detection.class_name = fcos_config_.class_names[detection.id].c_str();
        dets.push_back(detection);
      }
    }
  }
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
    RCLCPP_ERROR(rclcpp::get_logger("fcos_example"), "tensor layout error.");
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
      RCLCPP_ERROR(rclcpp::get_logger("fcos_example"),
                  "tensor layout null, error.");
      return -1;
    }
    if (hbSysFlushMem(&(tensors[i]->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE) != 0) {
      RCLCPP_ERROR(rclcpp::get_logger("fcos_example"),
                  "FlushMem tensors[%d] failed.", i);
      return -1;
    }
  }

  std::vector<std::vector<ScoreId>> scores;
  std::vector<Detection> dets;

  if (community_qat_) {
    CqatGetBboxAndScoresScaleNHWC(tensors, dets);
    yolo5_nms(dets, nms_threshold_, nms_top_k_, perception.det, false);
    return 0;
  }
  if (tensors[0]->properties.tensorLayout == HB_DNN_LAYOUT_NHWC) {
    GetBboxAndScoresNHWC(tensors, dets);
  } else if (tensors[0]->properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
    GetBboxAndScoresNCHW(tensors, dets);
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("fcos_example"), "tensor layout error.");
  }

  yolo5_nms(dets, nms_threshold_, nms_top_k_, perception.det, false);
  return 0;
}

}  // namespace parser_fcos
}  // namespace dnn_node
}  // namespace hobot
