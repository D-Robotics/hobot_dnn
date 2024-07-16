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

#include "dnn_node/util/output_parser/detection/fasterrcnn_output_parser.h"

#include "dnn_node/util/output_parser/utils.h"
namespace hobot {
namespace dnn_node {
namespace parser_fasterrcnn {

using cpu_op_rcnn_post_process_bbox_float_type_t = struct {
  float left;
  float top;
  float right;
  float bottom;
  float score;
  float class_label;
};

using detection_post_process_bbox_with_pad_type_t = struct {
  int16_t left;
  int16_t top;
  int16_t right;
  int16_t bottom;
  int8_t score;
  uint8_t class_label;
  int16_t padding[3];
};

inline float SigMoid(const float &input) {
  return 1 / (1 + std::exp(-1 * input));
}

inline float GetFloatByInt(int32_t value, uint32_t shift) {
  return (static_cast<float>(value)) / (static_cast<float>(1 << shift));
}

int ParseTensorRect(const std::shared_ptr<DNNTensor> &output_tensor,
                    std::shared_ptr<Filter2DResult> &output,
                    int32_t branch_idx);

int ParseTensorKps(const std::shared_ptr<DNNTensor> &output_tensor,
                   const std::shared_ptr<FasterRcnnKpsParserPara> &parser_para_,
                   const std::shared_ptr<Filter2DResult> &output_body_rect,
                   std::shared_ptr<LandmarksResult> &output_body_kps);

int32_t Parse(
    const std::shared_ptr<hobot::dnn_node::DnnNodeOutput> &node_output,
    const std::shared_ptr<FasterRcnnKpsParserPara> &parser_para_,
    const std::vector<int32_t> &box_outputs_index,
    int32_t kps_output_index,
    int32_t body_box_output_index,
    std::vector<std::shared_ptr<Filter2DResult>> &outputs,
    std::shared_ptr<LandmarksResult> &output_body_kps) {
  for (size_t i = 0; i < node_output->output_tensors.size(); i++) {
    hbSysFlushMem(&(node_output->output_tensors[i]->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  }
  outputs.resize(node_output->output_tensors.size());
  for (const auto &idx : box_outputs_index) {
    if (idx >= static_cast<int32_t>(node_output->output_tensors.size())) {
      return -1;
    }
    if (ParseTensorRect(
            node_output->output_tensors.at(idx), outputs[idx], idx) < 0) {
      return -1;
    }
  }
  if (kps_output_index > 0 &&
      kps_output_index < static_cast<int32_t>(node_output->output_tensors.size()) &&
      body_box_output_index > 0 &&
      body_box_output_index < static_cast<int32_t>(node_output->output_tensors.size())) {
    if (ParseTensorKps(node_output->output_tensors.at(kps_output_index),
                       parser_para_,
                       outputs.at(body_box_output_index),
                       output_body_kps) < 0) {
      return -1;
    }
  }

  return 0;
}

int ParseTensorRect(const std::shared_ptr<DNNTensor> &output_tensor,
                    std::shared_ptr<Filter2DResult> &output,
                    int32_t branch_idx) {
  if (!output) {
    output = std::make_shared<Filter2DResult>();
  }

  auto *detection_result =
      reinterpret_cast<uint8_t *>(output_tensor->sysMem[0].virAddr);
  auto tensor_properties_type = output_tensor->properties.tensorType;

  if (HB_DNN_TENSOR_TYPE_F32 == tensor_properties_type) {
    size_t item_size = sizeof(cpu_op_rcnn_post_process_bbox_float_type_t);
    float output_byte_size = *reinterpret_cast<float *>(detection_result);
    uint32_t box_num = output_byte_size / item_size;
    auto *p_box =
        reinterpret_cast<cpu_op_rcnn_post_process_bbox_float_type_t *>(
            reinterpret_cast<uintptr_t>(detection_result) + item_size);
    for (uint32_t i = 0; i < box_num; i++) {
      PerceptionRect roi{};
      roi.left = p_box[i].left;
      roi.top = p_box[i].top;
      roi.right = p_box[i].right;
      roi.bottom = p_box[i].bottom;
      roi.conf = p_box[i].score;
      roi.type = p_box[i].class_label;
      roi.branch = branch_idx;
      output->boxes.push_back(roi);
    }
  } else {
    size_t item_size = sizeof(detection_post_process_bbox_with_pad_type_t);
    uint16_t output_byte_size = *reinterpret_cast<uint16_t *>(detection_result);
    uint16_t box_num = output_byte_size / item_size;
    auto *p_box =
        reinterpret_cast<detection_post_process_bbox_with_pad_type_t *>(
            reinterpret_cast<uintptr_t>(detection_result) + item_size);
    for (uint32_t i = 0; i < box_num; i++) {
      PerceptionRect roi{};
      roi.left = p_box[i].left;
      roi.top = p_box[i].top;
      roi.right = p_box[i].right;
      roi.bottom = p_box[i].bottom;
      roi.conf = p_box[i].score;
      roi.type = p_box[i].class_label;
      roi.branch = branch_idx;
      output->boxes.push_back(roi);
    }
  }
  return 0;
}

int ParseTensorKps(const std::shared_ptr<DNNTensor> &output_tensor,
                   const std::shared_ptr<FasterRcnnKpsParserPara> &parser_para_,
                   const std::shared_ptr<Filter2DResult> &output_body_rect,
                   std::shared_ptr<LandmarksResult> &output_body_kps) {
  RCLCPP_INFO(rclcpp::get_logger("fasterrcnn_parser"),
              "FasterRcnnKpsOutputParser parse start");
  if (!parser_para_) {
    RCLCPP_ERROR(rclcpp::get_logger("fasterrcnn_parser"),
                 "FasterRcnnKpsOutputParser para is not set");
    return -1;
  }
  if (!output_tensor || !output_body_rect) {
    return -1;
  }
  if (!output_body_rect) {
    return -1;
  }
  RCLCPP_INFO(rclcpp::get_logger("fasterrcnn_parser"),
              "out box size: %zu",
              output_body_rect->boxes.size());
  size_t body_box_num = output_body_rect->boxes.size();

  if (!output_body_kps) {
    output_body_kps = std::make_shared<LandmarksResult>();
    output_body_kps->Reset();
  }

  if (body_box_num < 1 || !output_body_rect) {
    return 0;
  }

  int feature_size = parser_para_->aligned_kps_dim.at(1) *
                     parser_para_->aligned_kps_dim.at(2) *
                     parser_para_->aligned_kps_dim.at(3);
  int h_stride =
      parser_para_->aligned_kps_dim.at(2) * parser_para_->aligned_kps_dim.at(3);
  int w_stride = parser_para_->aligned_kps_dim.at(3);

  int32_t *kps_feature =
      reinterpret_cast<int32_t *>(output_tensor->sysMem[0].virAddr);
  for (size_t box_id = 0; box_id < body_box_num; ++box_id) {
    const auto &body_box = output_body_rect->boxes[box_id];
    float x1 = body_box.left;
    float y1 = body_box.top;
    float x2 = body_box.right;
    float y2 = body_box.bottom;
    float w = x2 - x1 + 1;
    float h = y2 - y1 + 1;

    float scale_x = parser_para_->kps_feat_width_ / w;
    float scale_y = parser_para_->kps_feat_height_ / h;

    float pos_distance =
        parser_para_->kps_pos_distance_ * parser_para_->kps_feat_width_;

    auto *mxnet_out_for_one_point_begin = kps_feature + feature_size * box_id;

    Landmarks skeleton;
    skeleton.resize(parser_para_->kps_points_number_);

    for (int kps_id = 0; kps_id < parser_para_->kps_points_number_; ++kps_id) {
      // find the best position
      int max_w = 0;
      int max_h = 0;
      int max_score_before_shift = mxnet_out_for_one_point_begin[kps_id];
      int32_t *mxnet_out_for_one_point = nullptr;
      for (int hh = 0; hh < parser_para_->kps_feat_height_; ++hh) {
        for (int ww = 0; ww < parser_para_->kps_feat_width_; ++ww) {
          mxnet_out_for_one_point =
              mxnet_out_for_one_point_begin + hh * h_stride + ww * w_stride;
          if (mxnet_out_for_one_point[kps_id] > max_score_before_shift) {
            max_w = ww;
            max_h = hh;
            max_score_before_shift = mxnet_out_for_one_point[kps_id];
          }
        }
      }

      float max_score = GetFloatByInt(max_score_before_shift,
                                      parser_para_->kps_shifts_[kps_id]);

      // get delta
      mxnet_out_for_one_point =
          mxnet_out_for_one_point_begin + max_h * h_stride + max_w * w_stride;
      const auto x_delta =
          mxnet_out_for_one_point[2 * kps_id +
                                  parser_para_->kps_points_number_];
      const auto x_shift =
          parser_para_
              ->kps_shifts_[2 * kps_id + parser_para_->kps_points_number_];
      float fp_delta_x = GetFloatByInt(x_delta, x_shift) * pos_distance;

      const auto y_delta =
          mxnet_out_for_one_point[2 * kps_id +
                                  parser_para_->kps_points_number_ + 1];
      const auto y_shift =
          parser_para_
              ->kps_shifts_[2 * kps_id + parser_para_->kps_points_number_ + 1];
      float fp_delta_y = GetFloatByInt(y_delta, y_shift) * pos_distance;

      Point point;
      point.x =
          (max_w + fp_delta_x + 0.46875 + parser_para_->kps_anchor_param_) /
              scale_x +
          x1;
      point.y =
          (max_h + fp_delta_y + 0.46875 + parser_para_->kps_anchor_param_) /
              scale_y +
          y1;
      point.score = SigMoid(max_score);
      skeleton[kps_id] = point;
    }
    output_body_kps->values.emplace_back(skeleton);
  }

  RCLCPP_INFO(rclcpp::get_logger("fasterrcnn_parser"),
              "Kps size: %zu",
              output_body_kps->values.size());
  RCLCPP_INFO(rclcpp::get_logger("fasterrcnn_parser"),
              "FasterRcnnKpsOutputParser parse done");
  return 0;
}

}  // namespace parser_fasterrcnn
}  // namespace dnn_node
}  // namespace hobot
