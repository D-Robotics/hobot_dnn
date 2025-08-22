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

#include "easy_dnn/task.h"

namespace hobot {
namespace easy_dnn {

Task::Task()
    : model_(nullptr),
      task_status_(TaskStatus::ALLOCATED),
      task_handle_(nullptr) {
  HB_UCP_INITIALIZE_SCHED_PARAM(&ctrl_param_);
}

int32_t Task::SetCtrlParam(hbUCPSchedParam &ctrl_param) {
  ctrl_param_ = ctrl_param;
  return HB_DNN_SUCCESS;
}

std::shared_ptr<DNNTensor> Task::AllocateTensor(
    hbDNNTensorProperties const &tensor_properties) {

    auto tensor = new DNNTensor;

    // 获取模型输出尺寸
    auto const out_aligned_size = tensor_properties.alignedByteSize;

    hbUCPSysMem *mem = new hbUCPSysMem;
    hbUCPMallocCached(mem, out_aligned_size, 0);

    tensor->properties = tensor_properties;
    tensor->sysMem = *mem;
    
    return std::shared_ptr<DNNTensor>(tensor,
                                      [mem](DNNTensor *tensor) {
                                        hbUCPFree(&(tensor->sysMem));
                                        delete tensor;
                                        delete mem;
                                      });
}

void Task::SetStatus(TaskStatus const status) {
  std::lock_guard<std::mutex> const lk{task_status_mutex_};
  if ((task_status_ == TaskStatus::TERMINATED) &&
      (status != TaskStatus::ALLOCATED)) {
    RCLCPP_WARN(rclcpp::get_logger("dnn"), "Task has been terminated, current stage set status failed.");
    return;
  }
  if ((task_status_ == TaskStatus::ALLOCATED) &&
      (status == TaskStatus::TERMINATED)) {
    RCLCPP_WARN(rclcpp::get_logger("dnn"), 
            "Task has been reset as TaskStatus::ALLOCATED, does not need to "
            "set TaskStatus::TERMINATED");
    return;
  }
  task_status_ = status;
}

int32_t Task::SetModel(Model *model) {

  if (model_ != nullptr) {
    RCLCPP_WARN(rclcpp::get_logger("dnn"), "Model already been set before!");
  }
  model_ = dynamic_cast<Model *>(model);

  if (model_ == nullptr) {
    RCLCPP_ERROR(rclcpp::get_logger("dnn"), "Task set model failed!");
    return HB_DNN_INVALID_ARGUMENT;
  }

  return HB_DNN_SUCCESS;
}

}  // namespace easy_dnn
}  // namespace hobot
