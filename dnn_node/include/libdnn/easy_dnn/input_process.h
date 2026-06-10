// Copyright (c) [2024] [D-Robotics].
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

#ifndef _EASY_DNN_INPUT_PROCESS_CROP_H_
#define _EASY_DNN_INPUT_PROCESS_CROP_H_

#include <memory>
#include <mutex>
#include <ostream>
#include <string>

#include "easy_dnn/common.h"
#include "easy_dnn/data_structure.h"

namespace hobot {
namespace easy_dnn {

/**
 *            image_width, image_width_stride
 *      ┌---------------------------------------┐
 *      |                               |       |
 *      |                               |       |
 *      |       (x,y)----------┐        |       |
 *      |       |              |        |       |
 *      |       |              |        |       |   image_height
 *      |       |              |height  |       |
 *      |       |              |        |       |
 *      |       |              |        |       |
 *      |       └--------------┘        |       |
 *      |             width             |       |
 *      |                               |       |
 *      |                               |       |
 *      └-------------------------------┘-------┘
 */

class CropConfig {
 public:
  CropConfig(int32_t x = 0,
                  int32_t y = 0,
                  int32_t width = 0,
                  int32_t height = 0)
      : x(x),
        y(y),
        width(width),
        height(height) {}

  int32_t x;
  int32_t y;
  int32_t width;   // 0 means crop to the right edge [x,...]
  int32_t height;  // 0 means crop to the bottom edge [y, ...]

};

class CropProcessor {
 public:
  int32_t Process(std::shared_ptr<DNNTensor>& tensor,
                  std::shared_ptr<CropConfig>& crop_config,
                  std::shared_ptr<DNNInput>& input);
};

}  // namespace easy_dnn
}  // namespace hobot

#endif  // _EASY_DNN_INPUT_PROCESS_CROP_H_
