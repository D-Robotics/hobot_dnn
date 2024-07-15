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

#include "include/image_utils.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "dnn/hb_sys.h"

int ImageUtils::Render(
    const std::shared_ptr<hobot::dnn_node::NV12PyramidInput> &pyramid,
    const ai_msgs::msg::PerceptionTargets::UniquePtr &ai_msg) {
  if (!pyramid || !ai_msg) return -1;

  char *y_img = reinterpret_cast<char *>(pyramid->y_vir_addr);
  char *uv_img = reinterpret_cast<char *>(pyramid->uv_vir_addr);
  auto height = pyramid->height;
  auto width = pyramid->y_stride;
  auto img_y_size = height * width;
  auto img_uv_size = img_y_size / 2;
  char *buf = new char[img_y_size + img_uv_size];
  memcpy(buf, y_img, img_y_size);
  memcpy(buf + img_y_size, uv_img, img_uv_size);
  cv::Mat nv12(height * 3 / 2, width, CV_8UC1, buf);
  cv::Mat mat;
  cv::cvtColor(nv12, mat, CV_YUV2BGR_NV12);
  delete[] buf;

  RCLCPP_INFO(rclcpp::get_logger("ImageUtils"),
              "target size: %d",
              ai_msg->targets.size());
  bool hasRois = false;
  for (size_t idx = 0; idx < ai_msg->targets.size(); idx++) {
    const auto &target = ai_msg->targets.at(idx);
    if (target.rois.empty()) {
      continue;
    }
    hasRois = true;
    RCLCPP_INFO(rclcpp::get_logger("ImageUtils"),
                "target type: %s, rois.size: %d",
                target.type.c_str(),
                target.rois.size());
    auto &color = colors[idx % colors.size()];
    for (const auto &roi : target.rois) {
      RCLCPP_INFO(
          rclcpp::get_logger("ImageUtils"),
          "roi.type: %s, x_offset: %d y_offset: %d width: %d height: %d",
          roi.type.c_str(),
          roi.rect.x_offset,
          roi.rect.y_offset,
          roi.rect.width,
          roi.rect.height);
      cv::rectangle(mat,
                    cv::Point(roi.rect.x_offset, roi.rect.y_offset),
                    cv::Point(roi.rect.x_offset + roi.rect.width,
                              roi.rect.y_offset + roi.rect.height),
                    color,
                    3);
      std::string roi_type = target.type;
      if (!roi.type.empty()) {
        roi_type = roi.type;
      }
      if (!roi_type.empty()) {
        cv::putText(mat,
                    roi_type,
                    cv::Point2f(roi.rect.x_offset, roi.rect.y_offset - 10),
                    cv::HersheyFonts::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1.5);
      }
    }

    for (const auto &lmk : target.points) {
      for (const auto &pt : lmk.point) {
        cv::circle(mat, cv::Point(pt.x, pt.y), 3, color, 3);
      }
    }
  }

  if (!hasRois) {
    RCLCPP_WARN(rclcpp::get_logger("ImageUtils"),
                "Frame has no roi, skip the rendering");
    return 0;
  }
  std::string saving_path = "render_" + ai_msg->header.frame_id + "_" +
                            std::to_string(ai_msg->header.stamp.sec) + "_" +
                            std::to_string(ai_msg->header.stamp.nanosec) +
                            ".jpeg";
  RCLCPP_WARN(rclcpp::get_logger("ImageUtils"),
              "Draw result to file: %s",
              saving_path.c_str());
  cv::imwrite(saving_path, mat);
  return 0;
}
