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

#include "dnn_node/util/output_parser/detection/ptq_efficientdet_output_parser.h"

#include <arm_neon.h>

#include <queue>
#include <fstream>

#include "dnn_node/util/output_parser/detection/nms.h"
#include "dnn_node/util/output_parser/utils.h"
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "rclcpp/rclcpp.hpp"

namespace hobot {
namespace dnn_node {
namespace parser_efficientdet {

const int kEfficientDetClassNum = 80;

/**
 * Config definition for EfficientDet
 */
struct EfficientDetConfig {
  std::vector<std::vector<double>> anchor_scales;
  std::vector<double> anchor_ratio;
  std::vector<int> feature_strides;
  int class_num;
  std::vector<std::string> class_names;
  std::vector<std::vector<float>> scales;
};

struct EDBaseAnchor {
  EDBaseAnchor(float x1, float y1, float x2, float y2)
      : x1_(x1), y1_(y1), x2_(x2), y2_(y2) {}
  float x1_;
  float y1_;
  float x2_;
  float y2_;
};

struct EDAnchor {
  EDAnchor(float c_x, float c_y, float w, float h)
      : c_x_(c_x), c_y_(c_y), w_(w), h_(h) {}
  float c_x_;
  float c_y_;
  float w_;
  float h_;
};

EfficientDetConfig default_efficient_det_config = {
    {{4.0, 5.039684199579493, 6.3496042078727974},
     {4.0, 5.039684199579493, 6.3496042078727974},
     {4.0, 5.039684199579493, 6.3496042078727974},
     {4.0, 5.039684199579493, 6.3496042078727974},
     {4.0, 5.039684199579493, 6.3496042078727974}},
    {0.5, 1, 2},
    {8, 16, 32, 64, 128},
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
     {}};

/**
 * Post process
 * @param[in] tensor: Model output tensors
 * @param[in] image_tensor: Input image tensor
 * @param[out] perception: Perception output data
 * @return 0 if success
 */
int PostProcess(std::vector<std::shared_ptr<DNNTensor>> &output_tensors,
                Perception &perception);

int GetAnchors(std::vector<EDAnchor> &anchors,
               int layer,
               int feat_height,
               int feat_width);

int GetBboxAndScores(std::shared_ptr<DNNTensor> c_tensor,
                     std::shared_ptr<DNNTensor> bbox_tensor,
                     std::vector<Detection> &dets,
                     std::vector<EDAnchor> &anchors,
                     int class_num,
                     int tensor_index);

EfficientDetConfig efficient_det_config_ = default_efficient_det_config;
float score_threshold_ = 0.05;
float nms_threshold_ = 0.5;
int nms_top_k_ = 100;
bool anchor_init_ = false;
std::vector<std::vector<EDAnchor>> anchors_table_;

std::string dequanti_file_ = "";
bool has_dequanti_node_ = true;
std::mutex anchors_mtx;

int LoadDequantiFile(const std::string &dequanti_file) {
  if (dequanti_file.empty()) return 0;
  dequanti_file_ = dequanti_file;
  // resize vector size
  efficient_det_config_.scales.resize(10);
  for (int i = 0; i < 5; i++) {
    efficient_det_config_.scales[i].resize(720);
    efficient_det_config_.scales[i + 5].resize(36);
  }
  // read scales for tensors
  std::fstream infile;
  infile.open(dequanti_file_.c_str(), std::ios_base::in);
  if (!infile.is_open()) {
    RCLCPP_DEBUG(rclcpp::get_logger("EfficientDetOutputParser"),
                 "Open file: %s failed!",
                 dequanti_file_.c_str());
    return -1;
  }
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < static_cast<int>(efficient_det_config_.scales[i].size()); j++) {
      infile >> efficient_det_config_.scales[i][j];
    }
  }
  has_dequanti_node_ = false;
  return 0;
};

int32_t Parse(
    const std::shared_ptr<hobot::dnn_node::DnnNodeOutput> &node_output,
    std::shared_ptr<DnnParserResult> &result) {
  if (!result) {
    result = std::make_shared<DnnParserResult>();
  }

  int ret = PostProcess(node_output->output_tensors, result->perception);
  if (ret != 0) {
    RCLCPP_INFO(rclcpp::get_logger("EfficientDetOutputParser"),
                "postprocess return error, code = %d",
                ret);
  }
  std::stringstream ss;
  ss << "EfficientDetOutputParser parse finished, predict result: "
     << result->perception;
  RCLCPP_DEBUG(
      rclcpp::get_logger("EfficientDetOutputParser"), "%s", ss.str().c_str());
  return ret;
}

int GetAnchors(std::vector<EDAnchor> &anchors,
               int layer,
               int feat_height,
               int feat_width) {
  int stride = default_efficient_det_config.feature_strides[layer];
  auto scales = default_efficient_det_config.anchor_scales[layer];
  const auto &ratios = default_efficient_det_config.anchor_ratio;
  int w = stride, h = stride;
  int size = w * h;
  float x_ctr = 0.5 * (w - 1.f);
  float y_ctr = 0.5 * (h - 1.f);
  // base anchor
  std::vector<EDBaseAnchor> base_anchors;
  for (const auto &ratio : ratios) {
    for (const auto &scale : scales) {
      double size_ratio = std::floor(size / ratio);
      double new_w = std::floor(std::sqrt(size_ratio) + 0.5) * scale;
      double new_h = std::floor(new_w / scale * ratio + 0.5) * scale;
      base_anchors.push_back(EDBaseAnchor(x_ctr - 0.5f * (new_w - 1.f),
                                          y_ctr - 0.5f * (new_h - 1.f),
                                          x_ctr + 0.5f * (new_w - 1.f),
                                          y_ctr + 0.5f * (new_h - 1.f)));
    }
  }

  for (int i = 0; i < feat_height; ++i) {
    for (int j = 0; j < feat_width; ++j) {
      auto ori_y = i * stride;
      auto ori_x = j * stride;
      for (const auto &base_anchor : base_anchors) {
        float x1 = base_anchor.x1_ + ori_x;
        float y1 = base_anchor.y1_ + ori_y;
        float x2 = base_anchor.x2_ + ori_x;
        float y2 = base_anchor.y2_ + ori_y;
        float width = x2 - x1 + 1.f;
        float height = y2 - y1 + 1.f;
        float ctr_x = x1 + 0.5f * (width - 1.f);
        float ctr_y = y1 + 0.5f * (height - 1.f);

        anchors.push_back(EDAnchor(ctr_x, ctr_y, width, height));
      }
    }
  }
  return 0;
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

int GetBboxAndScores(std::shared_ptr<DNNTensor> c_tensor,
                     std::shared_ptr<DNNTensor> bbox_tensor,
                     std::vector<Detection> &dets,
                     std::vector<EDAnchor> &anchors,
                     int class_num,
                     int tensor_index) {
  auto *shape = c_tensor->properties.validShape.dimensionSize;
  int32_t c_batch_size = shape[0];
  int h_idx, w_idx, c_idx;
  hobot::dnn_node::output_parser::get_tensor_hwc_index(
      c_tensor, &h_idx, &w_idx, &c_idx);

  int32_t c_hnum = shape[h_idx];
  int32_t c_wnum = shape[w_idx];
  int32_t c_cnum = shape[c_idx];
  uint32_t anchor_num_per_pixel = c_cnum / class_num;

  shape = bbox_tensor->properties.validShape.dimensionSize;
  int32_t b_batch_size = shape[0];
  hobot::dnn_node::output_parser::get_tensor_hwc_index(
      bbox_tensor, &h_idx, &w_idx, &c_idx);
  int32_t b_hnum = shape[h_idx];
  int32_t b_wnum = shape[w_idx];
  int32_t b_cnum = shape[c_idx];

  assert(anchor_num_per_pixel == static_cast<uint32_t>(b_cnum / 4));
  assert(c_batch_size == b_batch_size && c_hnum == b_hnum && c_wnum == b_wnum);
  auto box_num = b_batch_size * b_hnum * b_wnum * anchor_num_per_pixel;

  hbSysFlushMem(&(c_tensor->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  hbSysFlushMem(&(bbox_tensor->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

  if (has_dequanti_node_) {
    auto *raw_cls_data = reinterpret_cast<float *>(c_tensor->sysMem[0].virAddr);
    auto *raw_box_data =
        reinterpret_cast<float *>(bbox_tensor->sysMem[0].virAddr);
    for (unsigned int i = 0; i < box_num; i++) {
      uint32_t res_id_cur_anchor = i * class_num;
      int max_id = 0;
      float max_score = raw_cls_data[res_id_cur_anchor];
      if (class_num % 4 != 0) {
        for (int cls = 1; cls < class_num; ++cls) {
          float cls_value = raw_cls_data[res_id_cur_anchor + cls];
          if (cls_value > max_score) {
            max_score = cls_value;
            max_id = cls;
          }
        }
      } else {
        int max_id_0 = 0;
        int max_id_1 = 1;
        int max_id_2 = 2;
        int max_id_3 = 3;
        float max_score_0 = raw_cls_data[res_id_cur_anchor];
        float max_score_1 = raw_cls_data[res_id_cur_anchor + 1];
        float max_score_2 = raw_cls_data[res_id_cur_anchor + 2];
        float max_score_3 = raw_cls_data[res_id_cur_anchor + 3];
        for (int cls = 0; cls < class_num; cls += 4) {
          float cls_value_0 = raw_cls_data[res_id_cur_anchor + cls];
          float cls_value_1 = raw_cls_data[res_id_cur_anchor + cls + 1];
          float cls_value_2 = raw_cls_data[res_id_cur_anchor + cls + 2];
          float cls_value_3 = raw_cls_data[res_id_cur_anchor + cls + 3];
          if (cls_value_0 > max_score_0) {
            max_score_0 = cls_value_0;
            max_id_0 = cls;
          }

          if (cls_value_1 > max_score_1) {
            max_score_1 = cls_value_1;
            max_id_1 = cls + 1;
          }

          if (cls_value_2 > max_score_2) {
            max_score_2 = cls_value_2;
            max_id_2 = cls + 2;
          }

          if (cls_value_3 > max_score_3) {
            max_score_3 = cls_value_3;
            max_id_3 = cls + 3;
          }
        }

        if (max_score_0 > max_score) {
          max_score = max_score_0;
          max_id = max_id_0;
        }

        if (max_score_1 > max_score) {
          max_score = max_score_1;
          max_id = max_id_1;
        }

        if (max_score_2 > max_score) {
          max_score = max_score_2;
          max_id = max_id_2;
        }

        if (max_score_3 > max_score) {
          max_score = max_score_3;
          max_id = max_id_3;
        }
      }
      // box
      if (max_score <= score_threshold_) continue;

      // aligned index
      int start = i * 4;
      float dx = raw_box_data[start];
      float dy = raw_box_data[start + 1];
      float dw = raw_box_data[start + 2];
      float dh = raw_box_data[start + 3];
      float width = anchors[i].w_;
      float height = anchors[i].h_;
      float ctr_x = anchors[i].c_x_;
      float ctr_y = anchors[i].c_y_;

      float pred_ctr_x = dx * width + ctr_x;
      float pred_ctr_y = dy * height + ctr_y;
      float pred_w = std::exp(dw) * width;
      float pred_h = std::exp(dh) * height;

      // python在这里需要对框做clip,  x >= 0 && x <= input_width...
      float xmin = (pred_ctr_x - 0.5f * (pred_w - 1.f));
      float ymin = (pred_ctr_y - 0.5f * (pred_h - 1.f));
      float xmax = (pred_ctr_x + 0.5f * (pred_w - 1.f));
      float ymax = (pred_ctr_y + 0.5f * (pred_h - 1.f));

      Bbox bbox(xmin, ymin, xmax, ymax);
      bbox.xmin = std::max(xmin, 0.f);
      bbox.ymin = std::max(ymin, 0.f);

      dets.push_back(
          Detection(max_id,
                    max_score,
                    bbox,
                    efficient_det_config_.class_names[max_id].c_str()));
    }
  } else {
    auto *raw_cls_data =
        reinterpret_cast<int32_t *>(c_tensor->sysMem[0].virAddr);
    auto *raw_box_data =
        reinterpret_cast<int32_t *>(bbox_tensor->sysMem[0].virAddr);
    for (unsigned int i = 0; i < box_num; i++) {
      // score and cls
      uint32_t res_id_cur_anchor = i * class_num;
      int cls_scale_index = res_id_cur_anchor % 720;
      // cls scales for every box
      auto cls_scales =
          &efficient_det_config_.scales[tensor_index][cls_scale_index];
      // 获取max_id and max_score;
      auto max_score_id =
          MaxScoreID(raw_cls_data + res_id_cur_anchor, cls_scales, class_num);
      float max_score = max_score_id.first;
      int max_id = max_score_id.second;

      // box
      if (max_score <= score_threshold_) continue;

      // stride is 40, if do not have dequanti node
      int start = i * 4 + (4 * i) / 36 * 4;
      int scale_index = (i * 4) % 36;
      // box scales for every box
      auto box_scales =
          &efficient_det_config_.scales[tensor_index + 5][scale_index];
      float dx = raw_box_data[start] * box_scales[0];
      float dy = raw_box_data[start + 1] * box_scales[1];
      float dw = raw_box_data[start + 2] * box_scales[2];
      float dh = raw_box_data[start + 3] * box_scales[3];
      float width = anchors[i].w_;
      float height = anchors[i].h_;
      float ctr_x = anchors[i].c_x_;
      float ctr_y = anchors[i].c_y_;

      float pred_ctr_x = dx * width + ctr_x;
      float pred_ctr_y = dy * height + ctr_y;
      float pred_w = std::exp(dw) * width;
      float pred_h = std::exp(dh) * height;

      // python在这里需要对框做clip,  x >= 0 && x <= input_width...
      float xmin = (pred_ctr_x - 0.5f * (pred_w - 1.f));
      float ymin = (pred_ctr_y - 0.5f * (pred_h - 1.f));
      float xmax = (pred_ctr_x + 0.5f * (pred_w - 1.f));
      float ymax = (pred_ctr_y + 0.5f * (pred_h - 1.f));

      Bbox bbox(xmin, ymin, xmax, ymax);
      bbox.xmin = std::max(xmin, 0.f);
      bbox.ymin = std::max(ymin, 0.f);

      dets.push_back(
          Detection(max_id,
                    max_score,
                    bbox,
                    efficient_det_config_.class_names[max_id].c_str()));
    }
  }
  return 0;
}

int PostProcess(std::vector<std::shared_ptr<DNNTensor>> &tensors,
                Perception &perception) {
  perception.type = Perception::DET;

  int layer_num = efficient_det_config_.feature_strides.size();
  if (!anchor_init_) {
    std::unique_lock<std::mutex> mtx(anchors_mtx);
    if (!anchor_init_) {
      anchors_table_.resize(layer_num);
      for (int i = 0; i < layer_num; i++) {
        int height, width;
        hobot::dnn_node::output_parser::get_tensor_aligned_hw(
            tensors[i], &height, &width);
        GetAnchors(anchors_table_[i], i, height, width);
      }
      anchor_init_ = true;
      mtx.unlock();
    }
  }

  std::vector<Detection> dets;

  for (int i = 0; i < layer_num; i++) {
    std::vector<EDAnchor> &anchors = anchors_table_[i];
    GetBboxAndScores(tensors[i],
                     tensors[layer_num + i],
                     dets,
                     anchors,
                     kEfficientDetClassNum,
                     i);
  }
  yolo5_nms(dets, nms_threshold_, 6000, perception.det, false);
  if (static_cast<int>(perception.det.size()) > nms_top_k_) {
    perception.det.resize(nms_top_k_);
  }

  return 0;
}

}  // namespace parser_efficientdet
}  // namespace dnn_node
}  // namespace hobot
