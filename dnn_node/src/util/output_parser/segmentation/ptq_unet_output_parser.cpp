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

#include "dnn_node/util/output_parser/segmentation/ptq_unet_output_parser.h"

#include <queue>

#include "rclcpp/rclcpp.hpp"

#include "dnn_node/util/output_parser/utils.h"

namespace hobot {
namespace dnn_node {
namespace parser_unet {

int num_classes_ = 19;

int32_t Parse(
    const std::shared_ptr<hobot::dnn_node::DnnNodeOutput>& node_output,
    int img_h,
    int img_w,
    int model_h,
    int model_w,
    std::shared_ptr<DnnParserResult>& result) {
  if (!result) {
    result = std::make_shared<DnnParserResult>();
  }
  if (node_output->output_tensors.empty()) {
    RCLCPP_ERROR(rclcpp::get_logger("ClassficationOutputParser"),
                 "output_tensors is empty");
    return -1;
  }

  int ret = PostProcess(node_output->output_tensors, 
                        img_h,
                        img_w,
                        model_h,
                        model_w,
                        result->perception);

  if (ret != 0) {
    RCLCPP_INFO(rclcpp::get_logger("UnetOutputParser"),
                "postprocess return error, code = %d",
                ret);
  }
  std::stringstream ss;
  ss << "UnetOutputParser parse finished, predict result: "
     << result->perception;
  RCLCPP_DEBUG(rclcpp::get_logger("UnetOutputParser"), "%s", ss.str().c_str());
  return ret;
}

int PostProcess(std::vector<std::shared_ptr<DNNTensor>>& tensors,
                int img_h,
                int img_w,
                int model_h,
                int model_w,
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


  RCLCPP_DEBUG(rclcpp::get_logger("UnetOutputParser"),
               "PostProcess width: %d height: %d channel: %d",
               seg_width,
               seg_height,
               channel);

  float resize_ratio_h = static_cast<float>(img_h) / static_cast<float>(model_h);
  float resize_ratio_w = static_cast<float>(img_w) / static_cast<float>(model_w);
  float dst_ratio = std::max(resize_ratio_w, resize_ratio_h);
  // 根据长边的缩放系数计算出输出长宽上有效的比例
  float valid_h_ratio = resize_ratio_h < resize_ratio_w? 
    (static_cast<float>(img_h) / dst_ratio) / model_h: 1.0;
  float valid_w_ratio = resize_ratio_w < resize_ratio_h? 
    (static_cast<float>(img_w) / dst_ratio) / model_w: 1.0;

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

  if (tensors[0]->properties.tensorType == HB_DNN_TENSOR_TYPE_F32) {
    float* data = reinterpret_cast<float*>(tensors[0]->sysMem[0].virAddr);

    for (int h = 0; h < valid_h; ++h) {
      for (int w = 0; w < valid_w; ++w) {
        float top_score = -1000000.0f;
        int top_index = 0;
        float* c_data = data + (seg_width * h + w) * channel;
        for (int c = 0; c < channel; c++) {
          if (c_data[c] > top_score) {
            top_score = c_data[c];
            top_index = c;
          }
        }
        perception.seg.seg[h * valid_w + w] = top_index;
        perception.seg.data[h * valid_w + w] = static_cast<float>(top_index);
      }
    }
  } else if (tensors[0]->properties.tensorType == HB_DNN_TENSOR_TYPE_S8) {
    int8_t *data = reinterpret_cast<int8_t *>(tensors[0]->sysMem[0].virAddr);

    // parsing output data
    for (int h = 0; h < valid_h; ++h) {
      for (int w = 0; w < valid_w; ++w) {
        // the number of channel is 1, mean is category
        auto top_index = reinterpret_cast<int8_t *>(data + (h * seg_width + w))[0];
        perception.seg.seg[h * valid_w + w] = top_index;
        perception.seg.data[h * valid_w + w] = static_cast<float>(top_index);
      }
    }
  }

  return 0;
}

}  // namespace parser_unet
}  // namespace dnn_node
}  // namespace hobot
