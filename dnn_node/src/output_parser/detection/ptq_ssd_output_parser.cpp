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

#include "dnn_node/util/output_parser/detection/ptq_ssd_output_parser.h"

#include <queue>

#include "dnn_node/util/output_parser/detection/nms.h"
#include "dnn_node/util/output_parser/utils.h"
#include "rapidjson/document.h"

namespace hobot {
namespace dnn_node {
namespace parser_ssd {

inline float fastExp(float x) {
  union {
    uint32_t i;
    float f;
  } v;
  v.i = (12102203.1616540672f * x + 1064807160.56887296f);
  return v.f;
}

#define SSD_CLASS_NUM_P1 21

/**
 * Config definition for SSD
 */
struct SSDConfig {
  std::vector<float> std;
  std::vector<float> mean;
  std::vector<float> offset;
  std::vector<int> step;
  std::vector<std::pair<float, float>> anchor_size;
  std::vector<std::vector<float>> anchor_ratio;
  int class_num;
  std::vector<std::string> class_names;
};

/**
 * Default ssd config
 * std: [0.1, 0.1, 0.2, 0.2]
 * mean: [0, 0, 0, 0]
 * offset: [0.5, 0.5]
 * step: [8, 16, 32, 64, 100, 300]
 * anchor_size: [[30, 60], [60, 111], [111, 162], [162, 213], [213, 264],
 *              [264,315]]
 * anchor_ratio: [[2, 0.5, 0, 0], [2, 0.5, 3, 1.0 / 3],
 *              [2, 0.5,3, 1.0 / 3], [2, 0.5, 3, 1.0 / 3],
 *              [2, 0.5, 0, 0], [2,0.5, 0, 0]]
 * class_num: 20
 * class_names: ["aeroplane",   "bicycle", "bird",  "boaupdate", "bottle",
     "bus",         "car",     "cat",   "chair",     "cow",
     "diningtable", "dog",     "horse", "motorbike", "person",
     "pottedplant", "sheep",   "sofa",  "train",     "tvmonitor"]
 */
SSDConfig default_ssd_config = {
    {0.1, 0.1, 0.2, 0.2},
    {0, 0, 0, 0},
    {0.5, 0.5},
    {15, 30, 60, 100, 150, 300},
    {{60, -1}, {105, 150}, {150, 195}, {195, 240}, {240, 285}, {285, 300}},
    {{2, 0.5, 0, 0},
     {2, 0.5, 3, 1.0 / 3},
     {2, 0.5, 3, 1.0 / 3},
     {2, 0.5, 3, 1.0 / 3},
     {2, 0.5, 3, 1.0 / 3},
     {2, 0.5, 3, 1.0 / 3}},
    20,
    {"aeroplane",   "bicycle", "bird",  "boaupdate", "bottle",
     "bus",         "car",     "cat",   "chair",     "cow",
     "diningtable", "dog",     "horse", "motorbike", "person",
     "pottedplant", "sheep",   "sofa",  "train",     "tvmonitor"}};

static SSDConfig ssd_config_ = default_ssd_config;
static std::vector<std::vector<Anchor>> anchors_table_;
static std::mutex anchors_table_mutex_;
static float score_threshold_ = 0.25;
static float nms_threshold_ = 0.45;
static bool is_performance_ = true;
static int nms_top_k_ = 200;

/**
 * Post process
 * @param[in] tensor: Model output tensors
 * @param[in] image_tensor: Input image tensor
 * @param[out] perception: Perception output data
 * @return 0 if success
 */
int PostProcess(std::vector<std::shared_ptr<DNNTensor>> &output_tensors,
                Perception &perception);

int SsdAnchors(std::vector<Anchor> &anchors,
               int layer,
               int layer_height,
               int layer_width);

int GetBboxAndScores(std::shared_ptr<DNNTensor> cls_tensor,
                     std::shared_ptr<DNNTensor> bbox_tensor,
                     std::vector<Detection> &dets,
                     std::vector<Anchor> &anchors,
                     int class_num,
                     float cut_off_threshold);

int GetBboxAndScoresScale(std::shared_ptr<DNNTensor> cls_tensor,
                     std::shared_ptr<DNNTensor> bbox_tensor,
                     std::vector<Detection> &dets,
                     std::vector<Anchor> &anchors,
                     int class_num,
                     float cut_off_threshold);

int32_t Parse(
    const std::shared_ptr<hobot::dnn_node::DnnNodeOutput> &node_output,
    std::shared_ptr<DnnParserResult> &result) {
  if (!result) {
    result = std::make_shared<DnnParserResult>();
  }

  int ret = PostProcess(node_output->output_tensors, result->perception);
  std::stringstream ss;
  ss << "PTQSSDPostProcessMethod DoProcess finished, predict result: "
     << result->perception;
  RCLCPP_DEBUG(rclcpp::get_logger("SSDOutputParser"), "%s", ss.str().c_str());
  return ret;
}

int PostProcess(std::vector<std::shared_ptr<DNNTensor>> &tensors,
                Perception &perception) {
  perception.type = Perception::DET;
  int layer_num = ssd_config_.step.size();

  {
    std::lock_guard<std::mutex> lock(anchors_table_mutex_);
    if (anchors_table_.empty()) {
      // Note: note thread safe
      anchors_table_.resize(layer_num);
      for (int i = 0; i < layer_num; i++) {
        int height, width;
        std::vector<Anchor> &anchors = anchors_table_[i];
        hobot::dnn_node::output_parser::get_tensor_hw(
            tensors[i * 2], &height, &width);
        SsdAnchors(anchors, i, height, width);
      }
    }
  }

  std::vector<Detection> dets;
  for (int i = 0; i < layer_num; i++) {
    std::vector<Anchor> &anchors = anchors_table_[i];
    if (tensors[0]->properties.quantiType == NONE) {
      GetBboxAndScores(tensors[i * 2 + 1],
                     tensors[i * 2],
                     dets,
                     anchors,
                     ssd_config_.class_num + 1,
                     0.0001);
    } else if (tensors[0]->properties.quantiType == SCALE) {
      GetBboxAndScoresScale(tensors[i * 2 + 1],
                     tensors[i * 2],
                     dets,
                     anchors,
                     ssd_config_.class_num + 1,
                     0.0001);
    }
  }
  nms(dets, nms_threshold_, nms_top_k_, perception.det, false);
  return 0;
}

int SsdAnchors(std::vector<Anchor> &anchors,
               int layer,
               int layer_height,
               int layer_width) {
  int step = ssd_config_.step[layer];
  float min_size = ssd_config_.anchor_size[layer].first;
  float max_size = ssd_config_.anchor_size[layer].second;
  auto &anchor_ratio = ssd_config_.anchor_ratio[layer];
  for (int i = 0; i < layer_height; i++) {
    for (int j = 0; j < layer_width; j++) {
      float cy = (i + ssd_config_.offset[0]) * step;
      float cx = (j + ssd_config_.offset[1]) * step;
      anchors.emplace_back(Anchor(cx, cy, min_size, min_size));
      if (max_size > 0) {
        anchors.emplace_back(Anchor(cx,
                                    cy,
                                    std::sqrt(max_size * min_size),
                                    std::sqrt(max_size * min_size)));
      }
      for (int k = 0; k < 4; k++) {
        if (anchor_ratio[k] == 0) continue;
        float sr = std::sqrt(anchor_ratio[k]);
        float w = min_size * sr;
        float h = min_size / sr;
        anchors.emplace_back(Anchor(cx, cy, w, h));
      }
    }
  }
  return 0;
}

int GetBboxAndScores(std::shared_ptr<DNNTensor> cls_tensor,
                     std::shared_ptr<DNNTensor> bbox_tensor,
                     std::vector<Detection> &dets,
                     std::vector<Anchor> &anchors,
                     int class_num,
                     float cut_off_threshold) {
  int *shape = cls_tensor->properties.validShape.dimensionSize;
  int32_t c_batch_size = shape[0];
  int h_idx, w_idx, c_idx;
  hobot::dnn_node::output_parser::get_tensor_hwc_index(
      cls_tensor, &h_idx, &w_idx, &c_idx);

  int32_t c_hnum = shape[h_idx];
  int32_t c_wnum = shape[w_idx];
  int32_t c_cnum = shape[c_idx];
  int32_t anchor_num_per_pixel = c_cnum / class_num;

  shape = bbox_tensor->properties.validShape.dimensionSize;
  int32_t b_batch_size = shape[0];
  hobot::dnn_node::output_parser::get_tensor_hwc_index(
      cls_tensor, &h_idx, &w_idx, &c_idx);

  int32_t b_hnum = shape[h_idx];
  int32_t b_wnum = shape[w_idx];
  int32_t b_cnum = shape[c_idx];

  RCLCPP_DEBUG(rclcpp::get_logger("SSDOutputParser"),
               "PostProcess c_wnum:%d c_hnum:%d c_cnum:%d b_wnum:%d b_hnum:%d "
               "b_cnum: %d",
               c_wnum,
               c_hnum,
               c_cnum,
               b_wnum,
               b_hnum,
               b_cnum);

  assert(anchor_num_per_pixel == b_cnum / 4);
  assert(c_batch_size == b_batch_size && c_hnum == b_hnum && c_wnum == b_wnum);
  auto box_num = b_batch_size * b_hnum * b_wnum * anchor_num_per_pixel;

  hbSysFlushMem(&(cls_tensor->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  auto *raw_cls_data = reinterpret_cast<float *>(cls_tensor->sysMem[0].virAddr);

  hbSysFlushMem(&(bbox_tensor->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  auto *raw_box_data =
      reinterpret_cast<float *>(bbox_tensor->sysMem[0].virAddr);

  for (int i = 0; i < box_num; i++) {
    uint32_t res_id_cur_anchor = i * class_num;
    // get softmax sum
    double sum = 0;
    int max_id = 0;
    // TODO(@d-robotics.cc): fastExp only affect the final score value
    // confirm whether it affects the accuracy
    double background_score;
    if (is_performance_) {
      background_score = fastExp(raw_cls_data[res_id_cur_anchor]);
    } else {
      background_score = std::exp(raw_cls_data[res_id_cur_anchor]);
    }

    double max_score = 0;
    for (int cls = 0; cls < class_num; ++cls) {
      float cls_score;
      if (is_performance_) {
        cls_score = fastExp(raw_cls_data[res_id_cur_anchor + cls]);
      } else {
        cls_score = std::exp(raw_cls_data[res_id_cur_anchor + cls]);
      }
      sum += cls_score;
      /* scores should be larger than background score, or else will not be
      selected */
      if (cls != 0 && cls_score > max_score && cls_score > background_score) {
        max_id = cls - 1;
        max_score = cls_score;
      }
    }
    // get softmax score
    max_score = max_score / sum;

    if (max_score <= score_threshold_) {
      continue;
    }

    int start = i * 4;
    float dx = raw_box_data[start];
    float dy = raw_box_data[start + 1];
    float dw = raw_box_data[start + 2];
    float dh = raw_box_data[start + 3];

    auto x_min = (anchors[i].cx - anchors[i].w / 2);
    auto y_min = (anchors[i].cy - anchors[i].h / 2);
    auto x_max = (anchors[i].cx + anchors[i].w / 2);
    auto y_max = (anchors[i].cy + anchors[i].h / 2);

    auto prior_w = x_max - x_min;
    auto prior_h = y_max - y_min;
    auto prior_center_x = (x_max + x_min) / 2;
    auto prior_center_y = (y_max + y_min) / 2;
    auto decode_x = ssd_config_.std[0] * dx * prior_w + prior_center_x;
    auto decode_y = ssd_config_.std[1] * dy * prior_h + prior_center_y;
    auto decode_w = std::exp(ssd_config_.std[2] * dw) * prior_w;
    auto decode_h = std::exp(ssd_config_.std[3] * dh) * prior_h;

    auto xmin = (decode_x - decode_w * 0.5);
    auto ymin = (decode_y - decode_h * 0.5);
    auto xmax = (decode_x + decode_w * 0.5);
    auto ymax = (decode_y + decode_h * 0.5);

    xmin = std::max(xmin, 0.0);
    ymin = std::max(ymin, 0.0);

    if (xmax <= 0 || ymax <= 0) continue;
    if (xmin > xmax || ymin > ymax) continue;

    Bbox bbox(xmin, ymin, xmax, ymax);
    dets.emplace_back(static_cast<int>(max_id),
                      max_score,
                      bbox,
                      ssd_config_.class_names[max_id].c_str());
  }
  return 0;
}

int GetBboxAndScoresScale(std::shared_ptr<DNNTensor> cls_tensor,
                     std::shared_ptr<DNNTensor> bbox_tensor,
                     std::vector<Detection> &dets,
                     std::vector<Anchor> &anchors,
                     int class_num,
                     float cut_off_threshold) {
  int *shape = cls_tensor->properties.validShape.dimensionSize;
  int32_t c_batch_size = shape[0];
  int h_idx, w_idx, c_idx;
  hobot::dnn_node::output_parser::get_tensor_hwc_index(
      cls_tensor, &h_idx, &w_idx, &c_idx);

  int32_t c_hnum = shape[h_idx];
  int32_t c_wnum = shape[w_idx];
  int32_t c_cnum = shape[c_idx];
  int32_t anchor_num_per_pixel = c_cnum / class_num;

  shape = bbox_tensor->properties.validShape.dimensionSize;
  int32_t b_batch_size = shape[0];
  hobot::dnn_node::output_parser::get_tensor_hwc_index(
      cls_tensor, &h_idx, &w_idx, &c_idx);

  int32_t b_hnum = shape[h_idx];
  int32_t b_wnum = shape[w_idx];
  int32_t b_cnum = shape[c_idx];

  RCLCPP_DEBUG(rclcpp::get_logger("SSDOutputParser"),
               "PostProcess c_wnum:%d c_hnum:%d c_cnum:%d b_wnum:%d b_hnum:%d "
               "b_cnum: %d anchor_num: %d",
               c_wnum,
               c_hnum,
               c_cnum,
               b_wnum,
               b_hnum,
               b_cnum,
               anchor_num_per_pixel);


  assert(anchor_num_per_pixel == b_cnum / 4);
  assert(c_batch_size == b_batch_size && c_hnum == b_hnum && c_wnum == b_wnum);
  auto box_num = b_batch_size * b_hnum * b_wnum * anchor_num_per_pixel;

  hbSysFlushMem(&(cls_tensor->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  auto *cls_data = reinterpret_cast<int32_t *>(cls_tensor->sysMem[0].virAddr);
  auto *cls_scale_data = reinterpret_cast<float *>(cls_tensor->properties.scale.scaleData);
  int32_t cls_aligned = cls_tensor->properties.alignedShape.dimensionSize[c_idx];

  hbSysFlushMem(&(bbox_tensor->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  auto *box_data =
      reinterpret_cast<int32_t *>(bbox_tensor->sysMem[0].virAddr);
  auto *box_scale_data = reinterpret_cast<float *>(bbox_tensor->properties.scale.scaleData);
  int32_t box_aligned = bbox_tensor->properties.alignedShape.dimensionSize[c_idx];
  for (int h = 0; h < b_hnum; ++h) {
    for (int w = 0; w < b_wnum; ++w) {
      for (int i = 0; i < anchor_num_per_pixel; ++i) {
        auto cur_cls_data = cls_data + i * class_num;
        auto cur_cls_scale = cls_scale_data + i * class_num;

        double background_score;
        if (is_performance_) {
          background_score = fastExp(cur_cls_data[0] * cur_cls_scale[0]);
        } else {
          background_score = std::exp(cur_cls_data[0] * cur_cls_scale[0]);
        }
        double sum = 0;
        int max_id = 0;
        double max_score = 0;
        for (int cls = 0; cls < class_num; ++cls) {
          float cls_score;
          if (is_performance_) {
            cls_score = fastExp(cur_cls_data[cls] * cur_cls_scale[cls]);
          } else {
            cls_score = std::exp(cur_cls_data[cls] * cur_cls_scale[cls]);
          }
          sum += cls_score;
          /* scores should be larger than background score, or else will not be
          selected */
          if (cls != 0 && cls_score > max_score && cls_score > background_score) {
            max_id = cls - 1;
            max_score = cls_score;
          }
        }
        // get softmax score
        max_score = max_score / sum;
        
        if (max_score < score_threshold_) {
          continue;
        }

        auto cur_box_data = box_data + i * 4;
        auto cur_box_scale = box_scale_data + i * 4;
        float dx = cur_box_data[0] * box_scale_data[0];
        float dy = cur_box_data[1] * box_scale_data[1];
        float dw = cur_box_data[2] * box_scale_data[2];
        float dh = cur_box_data[3] * box_scale_data[3];

        int anchor_id = h * b_wnum * anchor_num_per_pixel + w * anchor_num_per_pixel + i;
        auto x_min = (anchors[anchor_id].cx - anchors[anchor_id].w / 2);
        auto y_min = (anchors[anchor_id].cy - anchors[anchor_id].h / 2);
        auto x_max = (anchors[anchor_id].cx + anchors[anchor_id].w / 2);
        auto y_max = (anchors[anchor_id].cy + anchors[anchor_id].h / 2);

        auto prior_w = x_max - x_min;
        auto prior_h = y_max - y_min;
        auto prior_center_x = (x_max + x_min) / 2;
        auto prior_center_y = (y_max + y_min) / 2;
        auto decode_x = ssd_config_.std[0] * dx * prior_w + prior_center_x;
        auto decode_y = ssd_config_.std[1] * dy * prior_h + prior_center_y;
        auto decode_w = std::exp(ssd_config_.std[2] * dw) * prior_w;
        auto decode_h = std::exp(ssd_config_.std[3] * dh) * prior_h;

        auto xmin = (decode_x - decode_w * 0.5);
        auto ymin = (decode_y - decode_h * 0.5);
        auto xmax = (decode_x + decode_w * 0.5);
        auto ymax = (decode_y + decode_h * 0.5);

        xmin = std::max(xmin, 0.0);
        ymin = std::max(ymin, 0.0);

        if (xmax <= 0 || ymax <= 0) continue;
        if (xmin > xmax || ymin > ymax) continue;

        Bbox bbox(xmin, ymin, xmax, ymax);
        dets.emplace_back(static_cast<int>(max_id),
                          max_score,
                          bbox,
                          ssd_config_.class_names[max_id].c_str());
      }
      cls_data += cls_aligned;
      box_data += box_aligned;
    }
  }
  return 0;
}

}  // namespace parser_ssd
}  // namespace dnn_node
}  // namespace hobot
