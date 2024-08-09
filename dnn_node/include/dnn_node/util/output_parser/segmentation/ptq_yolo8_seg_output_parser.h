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

#ifndef _SEGMENTATION_PTQ_YOLO8_SEG_OUTPUT_PARSER_H_
#define _SEGMENTATION_PTQ_YOLO8_SEG_OUTPUT_PARSER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "rapidjson/document.h"

#include "dnn/hb_dnn_ext.h"
#include "dnn_node/dnn_node_data.h"
#include "dnn_node/util/output_parser/perception_common.h"

using hobot::dnn_node::output_parser::Bbox;
using hobot::dnn_node::output_parser::YOLOSeg;
using hobot::dnn_node::output_parser::DnnParserResult;
using hobot::dnn_node::output_parser::Perception;

namespace hobot {
namespace dnn_node {
namespace parser_yolov8_seg {
int LoadConfig(const rapidjson::Document &document);

int32_t Parse(
    const std::shared_ptr<hobot::dnn_node::DnnNodeOutput> &node_output,
    const int resized_img_h,
    const int resized_img_w,
    const int model_h,
    const int model_w,
    std::shared_ptr<DnnParserResult> &output);
}  // namespace parser_yolov8_seg
}  // namespace dnn_node
}  // namespace hobot
#endif  // _SEGMENTATION_PTQ_YOLO8_SEG_OUTPUT_PARSER_H_
