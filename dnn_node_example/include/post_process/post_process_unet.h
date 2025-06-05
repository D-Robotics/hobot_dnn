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

#include <builtin_interfaces/msg/time.hpp>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <string>

#include "rapidjson/document.h"
#include "opencv2/core/core.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/matx.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "dnn_node/util/output_parser/segmentation/ptq_unet_output_parser.h"
#include "ai_msgs/msg/perception_targets.hpp"

#ifndef POST_PROCESS_UNET_H
#define POST_PROCESS_UNET_H

namespace hobot {
namespace dnn_node {
namespace parser_unet {

int LoadConfig(const rapidjson::Document &document);

ai_msgs::msg::PerceptionTargets::UniquePtr PostProcess(
    const std::shared_ptr<hobot::dnn_node::DnnNodeOutput>& outputs,
    int img_w,
    int img_h,
    int model_w,
    int model_h,
    bool dump_render_img);

int RenderSeg(const std::shared_ptr<hobot::dnn_node::DnnNodeOutput>& node_output,
                Parsing& seg);

}
}
}
#endif  // POST_PROCESS_UNET_H
