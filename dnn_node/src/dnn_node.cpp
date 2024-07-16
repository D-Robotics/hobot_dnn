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

#include "dnn_node/dnn_node.h"

#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dnn_node/dnn_node_impl.h"

namespace hobot {
namespace dnn_node {

DnnNode::DnnNode(const std::string &node_name, const NodeOptions &options)
    : rclcpp::Node(node_name, options) {
  dnn_node_para_ptr_ = std::make_shared<DnnNodePara>();
  dnn_node_impl_ = std::make_shared<DnnNodeImpl>(dnn_node_para_ptr_);
}

DnnNode::DnnNode(const std::string &node_name,
                 const std::string &namespace_,
                 const NodeOptions &options)
    : rclcpp::Node(node_name, namespace_, options) {
  dnn_node_para_ptr_ = std::make_shared<DnnNodePara>();
  dnn_node_impl_ = std::make_shared<DnnNodeImpl>(dnn_node_para_ptr_);
}

DnnNode::~DnnNode() {}

int DnnNode::Init() {
  RCLCPP_INFO(rclcpp::get_logger("dnn"), "Node init.");

  int ret = 0;
  // 1. set model info in node para
  ret = SetNodePara();
  if (ret != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("dnn"), "Set node para failed!");
    return ret;
  }

  // check node para
  if (ModelTaskType::InvalidType == dnn_node_para_ptr_->model_task_type) {
    RCLCPP_ERROR(rclcpp::get_logger("dnn"), "Invalid model task type");
    return -1;
  }

  // 校验bpu_core_ids参数
  if (!dnn_node_para_ptr_->bpu_core_ids.empty()) {
    if (static_cast<int>(dnn_node_para_ptr_->bpu_core_ids.size()) !=
        dnn_node_para_ptr_->task_num) {
      RCLCPP_ERROR(rclcpp::get_logger("dnn"),
                   "DnnNodePara of bpu_core_ids size %zu should be zero or "
                   "equal with task_num %d",
                   dnn_node_para_ptr_->bpu_core_ids.size(),
                   dnn_node_para_ptr_->task_num);
      return -1;
    }
    for (const auto &bpu_core_id : dnn_node_para_ptr_->bpu_core_ids) {
      if (bpu_core_id < HB_BPU_CORE_ANY ||
          bpu_core_id > HB_BPU_CORE_1) {
        RCLCPP_ERROR(rclcpp::get_logger("dnn"),
                     "Invalid bpu_core_id %d, which should be [%d, %d]",
                     static_cast<int>(bpu_core_id),
                     static_cast<int>(HB_BPU_CORE_ANY),
                     static_cast<int>(HB_BPU_CORE_1));
        return -1;
      }
    }
  }

  // 2. model init
  ret = dnn_node_impl_->ModelInit();
  if (ret != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("dnn"), "Model init failed!");
    return ret;
  }

  // 3. task init
  ret = dnn_node_impl_->TaskInit();
  if (ret != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("dnn"), "Task init failed!");
    return ret;
  }

  return ret;
}

int DnnNode::PostProcess(const std::shared_ptr<DnnNodeOutput> &output) {
  if (output) {
    RCLCPP_INFO(rclcpp::get_logger("dnn"), "Post process in dnn node");
  }
  return 0;
}

Model *DnnNode::GetModel() { return dnn_node_impl_->GetModel(); }

int DnnNode::GetModelInputSize(int32_t input_index, int &w, int &h) {
  return dnn_node_impl_->GetModelInputSize(input_index, w, h);
}

int DnnNode::Run(std::vector<std::shared_ptr<DNNInput>> &dnn_inputs,
                 const std::shared_ptr<DnnNodeOutput> &output,
                 const std::shared_ptr<std::vector<hbDNNRoi>> rois,
                 const bool is_sync_mode,
                 const int alloctask_timeout_ms,
                 const int infer_timeout_ms) {
  std::vector<std::shared_ptr<DNNTensor>> tensor_inputs;
  InputType input_type = InputType::DNN_INPUT;
  return dnn_node_impl_->Run(
      dnn_inputs,
      tensor_inputs,
      input_type,
      output,
      std::bind(&DnnNode::PostProcess, this, std::placeholders::_1),
      rois,
      is_sync_mode,
      alloctask_timeout_ms,
      infer_timeout_ms);
}

int DnnNode::Run(std::vector<std::shared_ptr<DNNTensor>> &tensor_inputs,
                 const std::shared_ptr<DnnNodeOutput> &output,
                 const bool is_sync_mode,
                 const int alloctask_timeout_ms,
                 const int infer_timeout_ms) {
  std::vector<std::shared_ptr<DNNInput>> dnn_inputs;
  InputType input_type = InputType::DNN_TENSOR;
  return dnn_node_impl_->Run(
      dnn_inputs,
      tensor_inputs,
      input_type,
      output,
      std::bind(&DnnNode::PostProcess, this, std::placeholders::_1),
      nullptr,
      is_sync_mode,
      alloctask_timeout_ms,
      infer_timeout_ms);
}

}  // namespace dnn_node
}  // namespace hobot
