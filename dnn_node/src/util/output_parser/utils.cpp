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

#include "dnn_node/util/output_parser/utils.h"

namespace hobot {
namespace dnn_node {
namespace output_parser {

int get_tensor_hwc_index(std::shared_ptr<DNNTensor> tensor,
                         int *h_index,
                         int *w_index,
                         int *c_index) {
  if (tensor->properties.tensorLayout == HB_DNN_LAYOUT_NHWC) {
    *h_index = 1;
    *w_index = 2;
    *c_index = 3;
  } else if (tensor->properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
    *c_index = 1;
    *h_index = 2;
    *w_index = 3;
  } else {
    return -1;
  }
  return 0;
}

int get_tensor_hw(std::shared_ptr<DNNTensor> tensor, int *height, int *width) {
  int h_index = 0;
  int w_index = 0;
  if (tensor->properties.tensorLayout == HB_DNN_LAYOUT_NHWC) {
    h_index = 1;
    w_index = 2;
  } else if (tensor->properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
    h_index = 2;
    w_index = 3;
  } else {
    return -1;
  }
  *height = tensor->properties.validShape.dimensionSize[h_index];
  *width = tensor->properties.validShape.dimensionSize[w_index];
  return 0;
}

int get_tensor_aligned_hw(std::shared_ptr<DNNTensor> tensor,
                          int *height,
                          int *width) {
  int h_index = 0;
  int w_index = 0;
  if (tensor->properties.tensorLayout == HB_DNN_LAYOUT_NHWC) {
    h_index = 1;
    w_index = 2;
  } else if (tensor->properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
    h_index = 2;
    w_index = 3;
  } else {
    return -1;
  }
  *height = tensor->properties.alignedShape.dimensionSize[h_index];
  *width = tensor->properties.alignedShape.dimensionSize[w_index];
  return 0;
}

int32_t TensorUtils::GetTensorValidHWC(hbDNNTensorProperties *properties,
                                       int *valid_h,
                                       int *valid_w,
                                       int *valid_c) {
  int h_index, w_index, c_index;
  TensorUtils::GetTensorHWCIndex(
      properties->tensorLayout, &h_index, &w_index, &c_index);
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
    // LOGE << "Unexpected layout:" << tensor_layout;
    return -1;
  }
  return 0;
}

void TensorUtils::GetTensorScale(hbDNNTensorProperties const &properties,
                                 std::vector<float> &scales) {
  if (properties.quantiType == SHIFT) {
    for (int i = 0; i < properties.shift.shiftLen; i++) {
      scales.push_back(1.0f /
                       static_cast<float>(1 << properties.shift.shiftData[i]));
    }
  } else {
    for (int i = 0; i < properties.scale.scaleLen; i++) {
      scales.push_back(properties.scale.scaleData[i]);
    }
  }
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
