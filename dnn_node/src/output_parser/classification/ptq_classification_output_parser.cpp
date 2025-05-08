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

#include "dnn_node/util/output_parser/classification/ptq_classification_output_parser.h"

#include <queue>
#include <string>
#include <fstream>

#include "rclcpp/rclcpp.hpp"

namespace hobot {
namespace dnn_node {
namespace parser_mobilenetv2 {

int PostProcess(std::shared_ptr<DNNTensor> &tensors, Perception &perception);

void GetTopkResult(std::shared_ptr<DNNTensor> &tensor,
                   std::vector<Classification> &top_k_cls);

const char *GetClsName(int id);

size_t top_k_ = 1;
std::vector<std::string> class_names_;

int InitClassNames(const std::string &cls_name_file) {
  std::ifstream fi(cls_name_file);
  if (fi) {
    std::string line;
    while (std::getline(fi, line)) {
      class_names_.push_back(line);
    }
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("ClassficationOutputParser"),
                 "can not open cls name file: %s",
                 cls_name_file.c_str());
    return -1;
  }
  return 0;
}

int LoadConfig(const rapidjson::Document &document) {
  if (document.HasMember("cls_names_list")) {
    std::string cls_name_file = document["cls_names_list"].GetString();
    if (InitClassNames(cls_name_file) < 0) {
      RCLCPP_ERROR(rclcpp::get_logger("example"),
                  "Load classification file [%s] fail",
                  cls_name_file.data());
      return -1;
    }
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("example"),
                "classification file is not set");
    return -1;
  }
  return 0;
}

int32_t Parse(
    const std::shared_ptr<hobot::dnn_node::DnnNodeOutput> &node_output,
    std::shared_ptr<DnnParserResult> &result) {
  if (!result) {
    result = std::make_shared<DnnParserResult>();
  }
  if (node_output->output_tensors.empty()) {
    RCLCPP_ERROR(rclcpp::get_logger("ClassficationOutputParser"),
                 "output_tensors is empty");
    return -1;
  }
  int ret = PostProcess(node_output->output_tensors.at(0), result->perception);
  if (ret != 0) {
    RCLCPP_INFO(rclcpp::get_logger("ClassficationOutputParser"),
                "postprocess return error, code = %d",
                ret);
  }
  std::stringstream ss;
  ss << "ClassficationOutputParser parse finished, predict result: "
     << result->perception;
  RCLCPP_DEBUG(
      rclcpp::get_logger("ClassficationOutputParser"), "%s", ss.str().c_str());
  return ret;
}

int PostProcess(std::shared_ptr<DNNTensor> &output_tensors,
                Perception &perception) {
  perception.type = Perception::CLS;
  GetTopkResult(output_tensors, perception.cls);
  return 0;
}

void GetTopkResult(std::shared_ptr<DNNTensor> &tensor,
                   std::vector<Classification> &top_k_cls) {
  tensor->CACHE_INVALIDATE();

  std::priority_queue<Classification,
                      std::vector<Classification>,
                      std::greater<Classification>>
      queue;
  int *shape = tensor->properties.validShape.dimensionSize;
  RCLCPP_DEBUG(rclcpp::get_logger("ClassficationOutputParser"),
               "PostProcess shape[1]: %d shape[2]: %d shape[3]: %d",
               shape[1],
               shape[2],
               shape[3]);
  for (auto i = 0; i < shape[1] * shape[2] * shape[3]; i++) {
    float score = tensor->GetTensorData<float>()[i];

    queue.push(Classification(i, score, GetClsName(i)));
    if (queue.size() > top_k_) {
      queue.pop();
    }
  }
  while (!queue.empty()) {
    top_k_cls.emplace_back(queue.top());
    queue.pop();
  }
  std::reverse(top_k_cls.begin(), top_k_cls.end());
}

const char *GetClsName(int id) {
  if (!class_names_.empty()) {
    return class_names_[id].c_str();
  }
  return nullptr;
}

}  // namespace parser_mobilenetv2
}  // namespace dnn_node
}  // namespace hobot
