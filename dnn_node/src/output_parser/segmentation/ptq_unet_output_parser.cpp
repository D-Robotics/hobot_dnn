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

#include "dnn_node/util/output_parser/segmentation/ptq_unet_output_parser.h"

#include <queue>

#include "rapidjson/document.h"
#include "rclcpp/rclcpp.hpp"

#include "dnn_node/util/output_parser/utils.h"

namespace hobot {
namespace dnn_node {
namespace parser_unet {

struct UNetConfig {
  int class_num;
};

UNetConfig default_unet_config = {19};
UNetConfig unet_config_ = default_unet_config;

int InitClassNum(const int &class_num) {
  if(class_num > 0){
    unet_config_.class_num = class_num;
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("UNet_parse"),
                 "class_num = %d is not allowed, only support class_num > 0",
                 class_num);
    return -1;
  }
  return 0;
}


int LoadConfig(const rapidjson::Document &document) {
  if (document.HasMember("class_num")){
    int class_num = document["class_num"].GetInt();
    if (InitClassNum(class_num) < 0) {
      return -1;
    }
  }
  return 0;
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
    RCLCPP_ERROR(rclcpp::get_logger("ClassficationOutputParser"),
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
    RCLCPP_INFO(rclcpp::get_logger("UnetOutputParser"),
                "postprocess return error, code = %d",
                ret);
  }
  return ret;
}

int PostProcess(std::vector<std::shared_ptr<DNNTensor>>& tensors,
                const int resized_img_h,
                const int resized_img_w,
                const int model_h,
                const int model_w,
                Perception& perception) {
  perception.type = Perception::SEG;
  tensors[0]->CACHE_INVALIDATE();

  // get shape
  int h_index, w_index, c_index;
  hobot::dnn_node::output_parser::get_tensor_hwc_index(
      tensors[0], &h_index, &w_index, &c_index);
  int seg_height = tensors[0]->properties.validShape.dimensionSize[h_index];
  int seg_width = tensors[0]->properties.validShape.dimensionSize[w_index];
  int channel = tensors[0]->properties.validShape.dimensionSize[c_index];
  int stride = tensors[0]->properties.stride[w_index] / tensors[0]->properties.stride[c_index];

  int aligned_channel = tensors[0]->properties.stride[c_index - 1] / tensors[0]->properties.stride[c_index];

  RCLCPP_INFO(rclcpp::get_logger("UnetOutputParser"),
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
  perception.seg.num_classes = unet_config_.class_num;

  #ifdef BPU_LIBDNN
    auto tensorlayout = tensors[0]->properties.tensorLayout;
  #endif
  #ifdef BPU_UCP
    auto tensorlayout = hobot::dnn_node::output_parser::get_tensor_layout(tensors[0]);
  #endif

  if (tensors[0]->properties.tensorType == HB_DNN_TENSOR_TYPE_F32) {
    float *data = tensors[0]->GetTensorData<float>();
    if (tensorlayout == HB_DNN_LAYOUT_NHWC) {
      for (int h = 0; h < valid_h; ++h) {
        for (int w = 0; w < valid_w; ++w) {
          float top_score = -1000000.0f;
          int top_index = 0;
          float* c_data = data + (seg_width * h + w) * stride;
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
    } else if (tensorlayout == HB_DNN_LAYOUT_NCHW) {
      for (int h = 0; h < valid_h; ++h) {
        for (int w = 0; w < valid_w; ++w) {
          float top_score = -1e9;
          int top_index = 0;

          for (int c = 0; c < channel; ++c) {
            // N=0, assume single batch
            // Offset = c * H * W + h * W + w
            float score = data[c * valid_h * valid_w + h * valid_w + w];
            if (score > top_score) {
              top_score = score;
              top_index = c;
            }
          }
          perception.seg.seg[h * valid_w + w] = top_index;
          perception.seg.data[h * valid_w + w] = static_cast<float>(top_index);
        }
      }
    }
  } else if (tensors[0]->properties.tensorType == HB_DNN_TENSOR_TYPE_S8) {
    int8_t *data = tensors[0]->GetTensorData<int8_t>();

    // parsing output data
    for (int h = 0; h < valid_h; ++h) {
      for (int w = 0; w < valid_w; ++w) {
        // the number of channel is 1, mean is category
        auto top_index = reinterpret_cast<int8_t *>(data + (h * seg_width + w))[0];
        perception.seg.seg[h * valid_w + w] = top_index;
        perception.seg.data[h * valid_w + w] = static_cast<float>(top_index);
      }
    }
  } else if (tensors[0]->properties.tensorType == HB_DNN_TENSOR_TYPE_S32) {
    int32_t *data = tensors[0]->GetTensorData<int32_t>();

    for (int h = 0; h < valid_h; ++h) {
      for (int w = 0; w < valid_w; ++w) {
        float top_score = -1000000.0f;
        int top_index = 0;
        int32_t* c_data = data + (seg_width * h + w) * aligned_channel;
        for (int c = 0; c < channel; c++) {
          float score = static_cast<float>(c_data[c] * tensors[0]->properties.scale.scaleData[c]);
          if (c_data[c] > top_score) {
            top_score = c_data[c];
            top_index = c;
          }
        }
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
