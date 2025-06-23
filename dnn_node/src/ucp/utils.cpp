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

#include "dnn_node/util/output_parser/utils.h"

namespace hobot {
namespace dnn_node {
namespace output_parser {

int get_tensor_layout(std::shared_ptr<DNNTensor> tensor) {
  if (tensor->properties.quantizeAxis == HB_DNN_LAYOUT_NHWC) {
    return HB_DNN_LAYOUT_NHWC;
  } else if (tensor->properties.quantizeAxis == HB_DNN_LAYOUT_NCHW) {
    return HB_DNN_LAYOUT_NCHW;
  } else if (tensor->properties.quantizeAxis == HB_DNN_LAYOUT_NONE) {
    if (tensor->properties.validShape.dimensionSize[3] <= tensor->properties.validShape.dimensionSize[1]) {
      return HB_DNN_LAYOUT_NHWC;
    } else {
      return HB_DNN_LAYOUT_NCHW;
    }
  }
  return -1;
}

int get_tensor_hwc_index(std::shared_ptr<DNNTensor> tensor,
                         int *h_index,
                         int *w_index,
                         int *c_index) {
  if (tensor->properties.quantizeAxis == HB_DNN_LAYOUT_NHWC) {
    *h_index = 1;
    *w_index = 2;
    *c_index = 3;
  } else if (tensor->properties.quantizeAxis == HB_DNN_LAYOUT_NCHW) {
    *c_index = 1;
    *h_index = 2;
    *w_index = 3;
  } else if (tensor->properties.quantizeAxis == HB_DNN_LAYOUT_NONE) {
    auto tensorlayout = get_tensor_layout(tensor);
    TensorUtils::GetTensorHWCIndex(
          tensorlayout, h_index, w_index, c_index);
  } else {
    return -1;
  }
  return 0;
}

int get_tensor_hw(std::shared_ptr<DNNTensor> tensor, int *height, int *width) {
  int h_index = 0;
  int w_index = 0;
  int c_index = 0;
  get_tensor_hwc_index(tensor, &h_index, &w_index, &c_index);
  *height = tensor->properties.validShape.dimensionSize[h_index];
  *width = tensor->properties.validShape.dimensionSize[w_index];
  return 0;
}

void seg_background_adjust(int8_t *seg,
                           float *data,
                           int cur_id,
                           int background_id,
                           bool have_background) {
  if (have_background) {
    if (cur_id == 0) {
      *seg = static_cast<int8_t>(background_id);
      *data = static_cast<float>(background_id);
    } else if (cur_id == background_id) {
      *seg = 0;
      *data = 0.;
    } else {
      *seg = static_cast<int8_t>(cur_id);
      *data = static_cast<float>(cur_id);
    }
  } else {
    *seg = static_cast<int8_t>(cur_id + 1);
    *data = static_cast<float>(cur_id + 1);
  }
}

int32_t TensorUtils::GetTensorValidHWC(hbDNNTensorProperties *properties,
                                       int *valid_h,
                                       int *valid_w,
                                       int *valid_c) {
  int h_index, w_index, c_index;
  TensorUtils::GetTensorHWCIndex(
      properties->quantizeAxis, &h_index, &w_index, &c_index);
  if (valid_h) {
    *valid_h = properties->validShape.dimensionSize[h_index];
  }
  if (valid_w) {
    *valid_w = properties->validShape.dimensionSize[w_index];
  }
  if (valid_c) {
    *valid_c = properties->validShape.dimensionSize[c_index];
  }
  return 0;
}

int32_t TensorUtils::GetTensorHWCIndex(int32_t tensor_layout,
                                       int *h_index,
                                       int *w_index,
                                       int *c_index) {
  if (tensor_layout == HB_DNN_LAYOUT_NHWC) {
    *h_index = 1;
    *w_index = 2;
    *c_index = 3;
  } else if (tensor_layout == HB_DNN_LAYOUT_NCHW) {
    *c_index = 1;
    *h_index = 2;
    *w_index = 3;
  } else {
    return -1;
  }
  return 0;
}

void Utils::GetRoiScale(float &scale_h,
                        float &scale_w,
                        hbDNNRoi &roi,
                        hbDNNTensorProperties &properties) {
  int roi_width = roi.right - roi.left;
  int roi_height = roi.bottom - roi.top;
  int dst_w = properties.validShape.dimensionSize[3];  // NCHW
  int dst_h = properties.validShape.dimensionSize[2];
  int step_w = ((roi_width - 1) * 256 + (dst_w - 1) / 2) / (dst_w - 1);
  int step_h = ((roi_height - 1) * 256 + (dst_h - 1) / 2) / (dst_h - 1);
  scale_w = static_cast<float>(step_w) / 256.0f;
  scale_h = static_cast<float>(step_h) / 256.0f;
}

}  // namespace output_parser
}  // namespace dnn_node
}  // namespace hobot
