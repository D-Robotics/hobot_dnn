// Copyright (c) [2024] [Horizon Robotics].
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
  HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&ctrl_param_);
}

int32_t Task::SetCtrlParam(hbDNNInferCtrlParam &ctrl_param) {
  ctrl_param_ = ctrl_param;
  return HB_DNN_SUCCESS;
}

std::shared_ptr<DNNTensor> Task::AllocateTensor(
    hbDNNTensorProperties const &tensor_properties) {

    auto tensor = new DNNTensor;

    // 获取模型输出尺寸
    int out_aligned_size = 4;
    for (int j = 0; j < tensor_properties.alignedShape.numDimensions; j++) {
        out_aligned_size *= tensor_properties.alignedShape.dimensionSize[j];
    }

    hbSysMem *mem = new hbSysMem;
    hbSysAllocCachedMem(mem, out_aligned_size);
    memset(mem->virAddr, 0, out_aligned_size);

    tensor->properties = tensor_properties;
    tensor->sysMem[0] = *mem;
    
    return std::shared_ptr<DNNTensor>(tensor,
                                      [mem](DNNTensor *tensor) {
                                        hbSysFreeMem(&(tensor->sysMem[0]));
                                        delete tensor;
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

void Task::Reset() {
  input_dnn_tensors_.clear();
  output_dnn_tensors_.clear();
}

}  // namespace easy_dnn
}  // namespace hobot
