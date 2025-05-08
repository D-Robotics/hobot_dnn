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

#include <arm_neon.h>
#include <opencv2/opencv.hpp>

#include "dnn_node/util/output_parser/segmentation/ptq_stdc_output_parser.h"

#include <queue>

#include "rclcpp/rclcpp.hpp"

#include "dnn_node/util/output_parser/utils.h"

namespace hobot {
namespace dnn_node {
namespace parser_stdc {

int num_classes_ = 19;

static void Dequantize(float *output,
                       int32_t *input,
                       float *input_scale,
                       int32_t *input_shape,
                       int32_t *input_aligned_shape) {
  int32_t channel = input_shape[1];
  int32_t height = input_shape[2];
  // Here width is a multiple of 4, and neon can be used to accelerate
  // calculations
  int32_t width = input_shape[3];

  for (int32_t c = 0; c < channel; c++) {
    float32x4_t scale = vdupq_n_f32(input_scale[c]);
    for (int32_t h = 0; h < height; h++) {
      for (int32_t w = 0; w < width; w += 4) {
        int32x4_t input_data_tmp =
            vld1q_s32(&input[h * input_aligned_shape[3] + w]);
        float32x4_t input_data = vcvtq_f32_s32(input_data_tmp);
        float32x4_t ouput_data = vmulq_f32(input_data, scale);
        vst1q_f32(&output[h * width + w], ouput_data);
      }
    }
    input += input_aligned_shape[2] * input_aligned_shape[3];
    output += height * width;
  }
}

int32_t Parse(
    const std::shared_ptr<hobot::dnn_node::DnnNodeOutput>& node_output,
    const int resized_img_h,
    const int resized_img_w,
    const int model_h,
    const int model_w,
    std::shared_ptr<DnnParserResult>& result) {
  if (!result) {
    result = std::make_shared<DnnParserResult>();
  }
  if (node_output->output_tensors.empty()) {
    RCLCPP_ERROR(rclcpp::get_logger("StdcOutputParser"),
                 "output_tensors is empty");
    return -1;
  }
  int ret = PostProcess(node_output->output_tensors, 
                        resized_img_h,
                        resized_img_w,
                        model_h,
                        model_w,
                        result->perception);

  if (ret != 0) {
    RCLCPP_INFO(rclcpp::get_logger("StdcOutputParser"),
                "postprocess return error, code = %d",
                ret);
  }
  std::stringstream ss;
  ss << "StdcOutputParser parse finished, predict result: "
     << result->perception;
  RCLCPP_DEBUG(rclcpp::get_logger("StdcOutputParser"), "%s", ss.str().c_str());
  return ret;
}

int PostProcess(std::vector<std::shared_ptr<DNNTensor>>& tensors,
                const int resized_img_h,
                const int resized_img_w,
                const int model_h,
                const int model_w,
                Perception& perception) {
  perception.type = Perception::SEG;
  hbSysFlushMem(&(tensors[0]->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

  // get shape
  int h_index, w_index, c_index;
  hobot::dnn_node::output_parser::get_tensor_hwc_index(
      tensors[0], &h_index, &w_index, &c_index);
  int seg_height = tensors[0]->properties.validShape.dimensionSize[h_index];
  int seg_width = tensors[0]->properties.validShape.dimensionSize[w_index];
  int channel = tensors[0]->properties.validShape.dimensionSize[c_index];


  RCLCPP_DEBUG(rclcpp::get_logger("StdcOutputParser"),
               "PostProcess width: %d height: %d channel: %d",
               seg_width,
               seg_height,
               channel);

  float valid_h_ratio = static_cast<float>(resized_img_h) / static_cast<float>(model_h);
  float valid_w_ratio = static_cast<float>(resized_img_w) / static_cast<float>(model_w);

  int valid_h = static_cast<int>(valid_h_ratio * seg_height);
  int valid_w = static_cast<int>(valid_w_ratio * seg_width);

  perception.seg.data.resize(valid_h * valid_w);
  perception.seg.seg.resize(valid_h * valid_w);

  perception.seg.valid_h = valid_h;
  perception.seg.valid_w = valid_w;
  perception.seg.height = static_cast<int>(model_h * valid_h_ratio);
  perception.seg.width = static_cast<int>(model_w * valid_w_ratio);
  perception.seg.channel = channel;
  perception.seg.num_classes = num_classes_;

  int32_t *feat_data =
      reinterpret_cast<int32_t *>(tensors[0]->sysMem[0].virAddr);
  int32_t *feat_valid_shape = tensors[0]->properties.validShape.dimensionSize;
  int32_t *feat_aligned_shape =
      tensors[0]->properties.alignedShape.dimensionSize;
  float *feat_scale = tensors[0]->properties.scale.scaleData;

  std::vector<float> feat(channel * seg_height * seg_width);
  {
    Dequantize(
        feat.data(), feat_data, feat_scale, feat_valid_shape, feat_aligned_shape);
  }

  cv::Mat seg_u8(valid_h, valid_w, CV_8UC1, perception.seg.seg.data());
  cv::Mat seg_f32(valid_h, valid_w, CV_32FC1, perception.seg.data.data());

  cv::Mat ori_seg_u8(seg_height, seg_width, CV_8UC1);
  cv::Mat ori_seg_f32(seg_height, seg_width, CV_32FC1);

  for (int h = 0; h < seg_height; ++h) {
    for (int w = 0; w < seg_width; ++w) {
      float top_score = -1000000.0f;
      int top_index = 0;
      for (int c = 0; c < channel; c++) {
        float score = feat[c * seg_width * seg_height + h * seg_width + w];
        if (score > top_score) {
          top_score = score;
          top_index = c;
        }
      }
      ori_seg_u8.at<uint8_t>(h * seg_width + w) = top_index;
      ori_seg_f32.at<float>(h * seg_width + w) = static_cast<float>(top_index);
    }
  }
  cv::resize(ori_seg_u8, seg_u8, cv::Size(valid_w, valid_h), cv::INTER_LINEAR);
  cv::resize(ori_seg_f32, seg_f32, cv::Size(valid_w, valid_h), cv::INTER_LINEAR);
  return 0;
}

int LoadConfig(const rapidjson::Document &document) {

  return 0;
}


}  // namespace parser_stdc
}  // namespace dnn_node
}  // namespace hobot
