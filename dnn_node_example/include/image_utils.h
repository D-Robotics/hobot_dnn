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

#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <memory>
#include <string>
#include <vector>

#include "ai_msgs/msg/perception_targets.hpp"
#include "dnn_node/dnn_node_data.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

using hobot::dnn_node::DNNTensor;
using hobot::dnn_node::NV12PyramidInput;

#define ALIGNED_2E(w, alignment) \
  ((static_cast<uint32_t>(w) + (alignment - 1U)) & (~(alignment - 1U)))
#define ALIGN_4(w) ALIGNED_2E(w, 4U)
#define ALIGN_8(w) ALIGNED_2E(w, 8U)
#define ALIGN_16(w) ALIGNED_2E(w, 16U)
#define ALIGN_64(w) ALIGNED_2E(w, 64U)

static uint8_t bgr_putpalette[] = {
    0,   0 ,  0  , 244, 35 , 232, 70 , 70 , 70 , 102, 102, 156, 190, 153, 153, 
    153, 153, 153, 250, 170, 30 , 220, 220, 0  , 107, 142, 35 , 152, 251, 152, 
    0  , 130, 180, 220, 20 , 60 , 255, 0  , 0  , 0  , 0  , 142, 0  , 0  , 70 , 
    0  , 60 , 100, 0  , 80 , 100, 0  , 0  , 230, 119, 11 , 32 , 216, 191, 69 , 
    50 , 33 , 199, 108, 59 , 247, 249, 96 , 97 , 97 , 234, 195, 239, 202, 156, 
    81 , 177, 90 , 180, 100, 245, 251, 146, 184, 245, 26 , 209, 56 , 20 , 144, 
    210, 56 , 241, 19 , 75 , 171, 144, 17 , 198, 216, 105, 125, 108, 212, 181, 
    75 , 189, 225, 137, 152, 226, 210, 107, 81 , 130, 189, 63 , 4  , 31 , 139, 
    106, 202, 255, 184, 64 , 56 , 200, 69 , 31 , 62 , 129, 13 , 19 , 235, 0  , 
    255, 129, 8  , 238, 24 , 80 , 176, 115, 54 , 232, 100, 164, 13 , 192, 234, 
    48 , 140, 176, 178, 145, 83 , 115, 225, 250, 18 , 6  , 98 , 34 , 156, 78 , 
    74 , 120, 22 , 185, 5  , 159, 111, 133, 243, 170, 252, 118, 23 , 29 , 143, 
    237, 6  , 163, 104, 231, 87 , 18 , 15 , 185, 45 , 152, 178, 147, 116, 56 , 
    28 , 197, 148, 134, 46 , 205, 243, 200, 47 , 5  , 233, 70 , 224, 88 , 0  , 
    237, 82 , 6  , 180, 104, 75 , 80 , 91 , 20 , 95 , 225, 61 , 91 , 37 , 187, 
    129, 183, 114, 246, 21 , 181, 26 , 90 , 201, 218, 8  , 81 , 97 , 14 , 208, 
    51 , 172, 247
};

static std::vector<cv::Scalar> colors{
    cv::Scalar(255, 0, 0),    // red
    cv::Scalar(255, 255, 0),  // yellow
    cv::Scalar(0, 255, 0),    // green
    cv::Scalar(0, 0, 255),    // blue
};

enum class ImageType { BGR = 0, NV12 = 1, BIN = 2 };

class ImageUtils {
 public:

  static int Render(
      const std::shared_ptr<hobot::dnn_node::NV12PyramidInput> &pyramid,
      const ai_msgs::msg::PerceptionTargets::UniquePtr &perception,
      const int img_h,
      const int img_w);
};

#endif  // IMAGE_UTILS_H
