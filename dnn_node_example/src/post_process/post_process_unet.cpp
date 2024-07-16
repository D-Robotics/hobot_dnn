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

#include "include/post_process/post_process_unet.h"

#include <unistd.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "include/dnn_example_node.h"
#include "include/image_utils.h"

namespace hobot {
namespace dnn_node {
namespace parser_unet {

int RenderUnet(
    const std::shared_ptr<DnnNodeOutput> &node_output, Parsing &seg) {
  static uint8_t bgr_putpalette[] = {
      128, 64,  128, 244, 35,  232, 70,  70,  70,  102, 102, 156, 190, 153, 153,
      153, 153, 153, 250, 170, 30,  220, 220, 0,   107, 142, 35,  152, 251, 152,
      0,   130, 180, 220, 20,  60,  255, 0,   0,   0,   0,   142, 0,   0,   70,
      0,   60,  100, 0,   80,  100, 0,   0,   230, 119, 11,  32};

  auto dnnExampleOutput =
      std::dynamic_pointer_cast<DnnExampleOutput>(node_output);
  auto pyramid = dnnExampleOutput->pyramid;
  if (!pyramid) {
    RCLCPP_ERROR(rclcpp::get_logger("UnetPostProcess"), "Invalid pyramid!");
    return -1;
  }
  int parsing_width = seg.width;
  int parsing_height = seg.height;

  char *y_img = reinterpret_cast<char *>(pyramid->y_vir_addr);
  char *uv_img = reinterpret_cast<char *>(pyramid->uv_vir_addr);
  auto height = pyramid->height;
  auto width = pyramid->width;
  auto img_y_size = height * width;
  auto img_uv_size = img_y_size / 2;
  char *buf = new char[img_y_size + img_uv_size];
  memcpy(buf, y_img, img_y_size);
  memcpy(buf + img_y_size, uv_img, img_uv_size);
  cv::Mat nv12(height * 3 / 2, width, CV_8UC1, buf);
  cv::Mat mat;                               //  get bgr mat from pyramid
  cv::cvtColor(nv12, mat, CV_YUV2BGR_NV12);  //  nv12 to bgr
  delete[] buf;

  cv::Mat parsing_img(parsing_height, parsing_width, CV_8UC3);
  uint8_t *parsing_img_ptr = parsing_img.ptr<uint8_t>();

  for (int h = 0; h < parsing_height; ++h) {
    for (int w = 0; w < parsing_width; ++w) {
      auto id = seg.seg[h * parsing_width + w];
      *parsing_img_ptr++ = bgr_putpalette[id * 3];
      *parsing_img_ptr++ = bgr_putpalette[id * 3 + 1];
      *parsing_img_ptr++ = bgr_putpalette[id * 3 + 2];
    }
  }

  // resize parsing image
  cv::resize(parsing_img, parsing_img, mat.size(), 0, 0);

  // alpha blending
  float alpha_f = 0.5;
  cv::Mat dst;
  addWeighted(mat, alpha_f, parsing_img, 1 - alpha_f, 0.0, dst);
  mat = std::move(dst);
  std::string saving_path = "render_unet_" + dnnExampleOutput->msg_header->frame_id + "_" +
                            std::to_string(dnnExampleOutput->msg_header->stamp.sec) + "_" +
                            std::to_string(dnnExampleOutput->msg_header->stamp.sec) +
                            ".jpeg";

  RCLCPP_INFO(rclcpp::get_logger("UnetPostProcess"),
              "Draw result to file: %s",
              saving_path.c_str());
  cv::imwrite(saving_path, mat);
  return 0;
}

}
}
}
