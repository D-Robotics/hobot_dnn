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

#ifndef EASY_DNN_IMAGE_UTILS_H
#define EASY_DNN_IMAGE_UTILS_H

#include <memory>
#include <string>
#include <vector>

#include "dnn_node/dnn_node_data.h"

#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#define LOGE_AND_RETURN_IF_NULL(ptr)                   \
  if (ptr == nullptr) {                                \
    std::cerr << #ptr " is null pointer" << std::endl; \
    return;                                            \
  }

using hobot::dnn_node::DNNTensor;
using hobot::dnn_node::NV12PyramidInput;

#define M_PI_F 3.141592653f

#define ALIGNED_2E(w, alignment) \
  ((static_cast<uint32_t>(w) + (alignment - 1U)) & (~(alignment - 1U)))
#define ALIGN_4(w) ALIGNED_2E(w, 4U)
#define ALIGN_8(w) ALIGNED_2E(w, 8U)
#define ALIGN_16(w) ALIGNED_2E(w, 16U)
#define ALIGN_64(w) ALIGNED_2E(w, 64U)

static cv::Scalar colors[] = {
    cv::Scalar(255, 0, 0),    // red
    cv::Scalar(255, 255, 0),  // yellow
    cv::Scalar(0, 255, 0),    // green
    cv::Scalar(0, 0, 255),    // blue
};

enum class ImageType {
  BGR = 0,
  NV12 = 1
};

class ImageUtils {
 public:
  static std::shared_ptr<NV12PyramidInput> GetNV12Pyramid(
      const std::string &image_file, ImageType image_type,
      int scaled_img_height, int scaled_img_width);

  static std::shared_ptr<NV12PyramidInput> GetNV12Pyramid(
      const std::string &image_file,
      int scaled_img_height,
      int scaled_img_width,
      int &original_img_height,
      int &original_img_width);

  static std::shared_ptr<NV12PyramidInput> GetNV12Pyramid(
      const cv::Mat &image,
      int scaled_img_height,
      int scaled_img_width);

  // 输入图片size小于scale size（模型输入size）：将输入图片padding到左上区域
  // 输入图片size大于scale size（模型输入size）：crop输入图片左上区域
  static std::shared_ptr<NV12PyramidInput> GetNV12PyramidFromNV12Img(
      const char* in_img_data,
      int in_img_height,
      int in_img_width,
      int scaled_img_height,
      int scaled_img_width);

  static int32_t BGRToNv12(cv::Mat &bgr_mat, cv::Mat &img_nv12);

  static void GetNV12Tensor(std::string &image_file,
                            std::shared_ptr<DNNTensor> &tensor);
};

#endif  // EASY_DNN_IMAGE_UTILS_H
