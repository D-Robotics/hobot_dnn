// Copyright (c) [2024] [Horizon Robotics].
//
// You can use this software according to the terms and conditions of
// the Apache v2.0.
// You may obtain a copy of Apache v2.0. at:
//
//     http: //www.apache.org/licenses/LICENSE-2.0
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See Apache v2.0 for more details.

#include "easy_dnn/input_process.h"

#include <iostream>

#include "rclcpp/rclcpp.hpp"

#include "easy_dnn/common.h"
#include "easy_dnn/data_structure.h"

#define ALIGNED_16(w) \
  ((static_cast<uint32_t>(w) + 15U) & (~15U))

namespace hobot {
namespace easy_dnn {

int32_t CropProcessor::Process(std::shared_ptr<DNNTensor>& tensor,
                               std::shared_ptr<CropConfig>& crop_config,
                               std::shared_ptr<DNNInput>& input) {
  if (tensor->properties.tensorType > HB_DNN_IMG_TYPE_NV12_SEPARATE) {
    RCLCPP_ERROR(rclcpp::get_logger("dnn"), "CropProcessor only support Y, NV12 and NV12_SEPARATE!");
    return -1;
  }
  // nv12 and nv12_separate model is compatible
  // nv12 model can give separate tensor data
  if (tensor->properties.tensorType == HB_DNN_IMG_TYPE_NV12) {
    tensor->properties.tensorType = HB_DNN_IMG_TYPE_NV12_SEPARATE;
  }
  auto const pyramid_input{std::dynamic_pointer_cast<NV12PyramidInput>(input)};
  if (pyramid_input == nullptr) {
    return -1;
  }
  if (pyramid_input->y_stride != pyramid_input->uv_stride) {
    RCLCPP_ERROR(rclcpp::get_logger("dnn"), "Y stride must equal to uv stride!!!");
    return -1;
  }

  if (crop_config->x % 2 != 0 ||
      crop_config->y % 2 != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("dnn"), "x,y expected even, but got x: %d, y: %d", crop_config->x, crop_config->y);
    return -1;
  }
  if (crop_config->x >= pyramid_input->width ||
      crop_config->y >= pyramid_input->height) {
    std::stringstream ss;
    ss << "crop postion x,y out of bound, x:" << crop_config->x
         << ", y:" << crop_config->y
         << ", input data width: " << pyramid_input->width
         << ", height: " << pyramid_input->height << "\n";
    RCLCPP_ERROR(rclcpp::get_logger("dnn"), "%s", ss.str().c_str());
    return -1;
  }
  if (crop_config->x + crop_config->width > pyramid_input->width ||
      crop_config->y + crop_config->height > pyramid_input->height) {
    std::stringstream ss;
    ss << "crop size out of bound, x + width = "
         << crop_config->x + crop_config->width
         << ", y + height = " << crop_config->y + crop_config->height
         << ", input data width: " << pyramid_input->width
         << ", height: " << pyramid_input->height << "\n";
    RCLCPP_ERROR(rclcpp::get_logger("dnn"), "%s", ss.str().c_str());
    return -1;
  }
  // BPU address is aligned to 16
  if (crop_config->x % 16 != 0) {
    crop_config->x = static_cast<int32_t>(static_cast<uint32_t>(crop_config->x) &
                                        (~15U));
    std::stringstream ss;
    ss << "Crop description x position must be aligned to 16, adjust it in "
            "crop processor. The adjusted description is: "
         << crop_config << "\n";
    RCLCPP_ERROR(rclcpp::get_logger("dnn"), "%s", ss.str().c_str());
  }
  auto& y{tensor->sysMem[0]};
  auto& properties{tensor->properties};
  // for NV12 Tensor, the layout is always NCHW
  constexpr int32_t h_index{2};
  constexpr int32_t w_index{3};

  int32_t valid_width{0};
  int32_t valid_height{0};
  int32_t const aligned_width{pyramid_input->y_stride};
  if (crop_config->height != 0) {
    valid_height = crop_config->height;
  } else {
    valid_height = pyramid_input->height - crop_config->y;
  }
  if (crop_config->width != 0) {
    valid_width = crop_config->width;
  } else {
    valid_width = pyramid_input->width - crop_config->x;
  }

  properties.validShape.dimensionSize[h_index] = valid_height;
  properties.validShape.dimensionSize[w_index] = valid_width;
  properties.alignedShape.dimensionSize[h_index] = valid_height;
  properties.alignedShape.dimensionSize[w_index] = aligned_width;

  int32_t const y_offset{crop_config->y * pyramid_input->y_stride + crop_config->x};
  y.phyAddr = pyramid_input->y_phy_addr + static_cast<uint64_t>(y_offset);
  y.virAddr = reinterpret_cast<uint8_t*>(pyramid_input->y_vir_addr) + y_offset;

  if (y.phyAddr % 16U != 0U) {
    RCLCPP_ERROR(rclcpp::get_logger("dnn"), "y.phyAddr error");
    return -1;
  }

  // Boundary case: aligned mem can be out of memory
  // but the outer area will not be used, it is ok
  //          --------------------
  //          |                  |
  //          |      -------     |
  //          |      |     |     |
  //          |      |     |     |
  //          --------------------
  y.memSize = ALIGNED_16(aligned_width * valid_height);
  if (tensor->properties.tensorType == HB_DNN_IMG_TYPE_NV12_SEPARATE) {
    auto& uv{tensor->sysMem[1]};
    int32_t const uv_offset{
        crop_config->y / 2 * pyramid_input->uv_stride + crop_config->x};
    uv.phyAddr = pyramid_input->uv_phy_addr + static_cast<uint64_t>(uv_offset);
    uv.virAddr =
        reinterpret_cast<uint8_t*>(pyramid_input->uv_vir_addr) + uv_offset;
    uv.memSize = ALIGNED_16(aligned_width * valid_height / 2U);
  }
  return 0;
}
}  // namespace easy_dnn
}  // namespace hobot
