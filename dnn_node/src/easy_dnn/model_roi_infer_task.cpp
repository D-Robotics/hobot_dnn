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

#include "easy_dnn/model_roi_infer_task.h"

namespace hobot {
namespace easy_dnn {

ModelRoiInferTask::ModelRoiInferTask() : Task::Task() {}

int32_t ModelRoiInferTask::SetModel(Model *model) {
  int ret = Task::SetModel(model);
  if (ret != HB_DNN_SUCCESS) {
    return ret;
  }

  int32_t const input_count{model_->GetInputCount()};
  input_tensors_.resize(static_cast<size_t>(input_count));
  input_dnn_tensors_.resize(static_cast<size_t>(input_count));

  int32_t const output_count{model->GetOutputCount()};
  output_tensors_.resize(static_cast<size_t>(output_count));
  output_dnn_tensors_.resize(static_cast<size_t>(output_count));
  real_mem_size_.resize(static_cast<size_t>(output_count), 0);

  return HB_DNN_SUCCESS;
}

int32_t ModelRoiInferTask::SetInputRois(std::vector<hbDNNRoi> &rois) {
  rois_ = rois;
  int32_t const input_count{model_->GetInputCount()};
  size_t const roi_num{rois.size()};
  size_t const total_input_count{roi_num * static_cast<uint32_t>(input_count)};

  inputs_.resize(total_input_count);
  input_tensors_.resize(total_input_count);
  input_dnn_tensors_.resize(total_input_count);
  return HB_DNN_SUCCESS;
}

int32_t ModelRoiInferTask::SetInputs(
    std::vector<std::shared_ptr<DNNInput>> &inputs) {
  if (inputs.size() != inputs_.size()) {
    RCLCPP_ERROR(rclcpp::get_logger("dnn"), 
              "Inpus size [%zu] is not match setting size [%zu]", inputs.size(), inputs_.size());
    return HB_DNN_INVALID_ARGUMENT;
  }

  size_t const input_size{inputs_.size()};
  for (size_t i{0U}; i < input_size; ++i) {
    if (!inputs[i]) {
      RCLCPP_ERROR(rclcpp::get_logger("dnn"), 
                "Set Inputs[%zu] failed", i);
      return HB_DNN_INVALID_ARGUMENT;
    }
    inputs_[i] = inputs[i];
  }
  
  input_dnn_tensors_.resize(inputs_.size());
  return HB_DNN_SUCCESS;
}

int32_t ModelRoiInferTask::SetInputTensors(
    std::vector<std::shared_ptr<DNNTensor>> &input_tensors) {
  size_t const input_size{input_tensors.size()};
  if (input_size != input_tensors_.size()) {
    RCLCPP_ERROR(rclcpp::get_logger("dnn"), 
          "input_size[%zu] is not equal to input_tensors_ size[%zu]", input_size, input_tensors_.size());
    return HB_DNN_API_USE_ERROR;
  }

  for (size_t i{0U}; i < input_size; ++i) {
    if (input_tensors[i] == nullptr) {
      RCLCPP_ERROR(rclcpp::get_logger("dnn"), 
          "input_tensors [%zu] is null", i);
      return HB_DNN_INVALID_ARGUMENT;
    }
    input_tensors_[i] = input_tensors[i];
    input_dnn_tensors_[i] = static_cast<hbDNNTensor>(*(input_tensors[i]));
  }
  return HB_DNN_SUCCESS;
}

int32_t ModelRoiInferTask::GetOutputTensors(
    std::vector<std::shared_ptr<DNNTensor>> &output_tensors) {
  output_tensors = output_tensors_;
  return HB_DNN_SUCCESS;
}

int32_t ModelRoiInferTask::GetOutputTensors(
    std::vector<std::vector<std::shared_ptr<DNNTensor>>> &output_tensors) {
  output_tensors = roi_output_tensors_;
  return HB_DNN_SUCCESS;
}

int32_t ModelRoiInferTask::ProcessInput() {
  int32_t process_count{0};
  int32_t const rois_size{static_cast<int32_t>(rois_.size())};
  int32_t const input_count{model_->GetInputCount()};
  for (int32_t i{0}; i < rois_size; i++) {
    for (int32_t j{0}; j < input_count; j++) {
      int32_t const k{i * input_count + j};
      if (inputs_[k] != nullptr) {

        if (input_tensors_[k] == nullptr) {
          
          model_->GetInputTensorProperties(input_dnn_tensors_[k].properties, j);
          input_tensors_[k] = std::shared_ptr<DNNTensor>(
              static_cast<DNNTensor *>(&input_dnn_tensors_[k]),
              [](DNNTensor *const tensor) {});
        }

        std::shared_ptr<CropConfig> input_conf = std::make_shared<CropConfig>();
        std::shared_ptr<CropProcessor> input_processor = std::make_shared<CropProcessor>();
        int ret = input_processor->Process(
                input_tensors_[k], input_conf, inputs_[i]);
        if (ret != HB_DNN_SUCCESS) {
          RCLCPP_ERROR(rclcpp::get_logger("dnn"), 
            "Input process failed, roi: %d, ret[%d]", i, HB_DNN_RUN_TASK_FAILED);
          return HB_DNN_RUN_TASK_FAILED;
        }

        process_count++;
      } else {
        RCLCPP_ERROR(rclcpp::get_logger("dnn"), 
            "DNNInput must be set for roi:{%d},branch{%d}", i, j);
        return HB_DNN_API_USE_ERROR;
      }
    }
  }
  SetStatus(TaskStatus::INPUT_PROCESS_DONE);
  return HB_DNN_SUCCESS;
}

int32_t ModelRoiInferTask::RunInfer() {
  
  int ret = PrepareInferInputOutput();
  if (ret != 0) {
    return ret;
  }

  ctrl_param_.more = false;
  auto *output_ptr{output_dnn_tensors_.data()};

  {
    std::unique_lock<std::mutex> const lk{release_mtx_};
    RETURN_IF_FAILED(hbDNNRoiInfer(&task_handle_,
                                   &output_ptr,
                                   input_dnn_tensors_.data(),
                                   rois_.data(),
                                   static_cast<int32_t>(rois_.size()),
                                   model_->GetDNNHandle(),
                                   &ctrl_param_));
  }
  SetStatus(TaskStatus::INFERRING);
  return HB_DNN_SUCCESS;
}

int32_t ModelRoiInferTask::WaitInferDone(int32_t timeout) {
  int32_t code{HB_DNN_SUCCESS};
  {
    std::unique_lock<std::mutex> const lk{release_mtx_};
    code = hbDNNWaitTaskDone(task_handle_, timeout);
  }

  if (code == HB_DNN_SUCCESS) {
    SetStatus(TaskStatus::INFERENCE_DONE);
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("dnn"), 
          "ModelRoiInferTask Infer TimeOut %d", timeout);
    SetStatus(TaskStatus::INFERENCE_TIMEOUT);
  }
  // release dnn task resource immediately, will not cost much time
  hbDNNReleaseTask(task_handle_);
  task_handle_ = nullptr;
  return code;
}

int32_t ModelRoiInferTask::PrepareInferInputOutput() {
  // check input tensors
  int32_t roi_num{static_cast<int32_t>(rois_.size())};

  auto const alloc_tensor{[this, &roi_num](
                              int32_t const i,
                              hbDNNTensorProperties properties) -> int32_t {
    properties.alignedShape.dimensionSize[0] *= roi_num;
    properties.validShape.dimensionSize[0] *= roi_num;
    properties.alignedByteSize *= roi_num;
    real_mem_size_[i] = properties.alignedByteSize;
    RCLCPP_DEBUG(rclcpp::get_logger("dnn"), 
        "Alloc output tensor for branch {%d} internal, alloc mem size {%d}.",
        i,
        real_mem_size_[i]);

    auto const output_tensor{AllocateTensor(properties)};
    if (output_tensor.get() == nullptr) {
      RCLCPP_ERROR(rclcpp::get_logger("dnn"), 
        "Allocate tensor failed, output branch: %d", i);
      return HB_DNN_OUT_OF_MEMORY;
    }
    output_tensors_[i] = output_tensor;
    output_dnn_tensors_[i] = static_cast<hbDNNTensor>(*output_tensor);
    return HB_DNN_SUCCESS;
  }};

  int32_t ret{HB_DNN_SUCCESS};
  int32_t const output_count{model_->GetOutputCount()};

  for (int32_t i{0}; i < output_count; ++i) {
    hbDNNTensorProperties properties;
    ret = model_->GetOutputTensorProperties(properties, i);
    if (ret != HB_DNN_SUCCESS) {
      RCLCPP_ERROR(rclcpp::get_logger("dnn"), 
        "GetOutputTensorProperties for index {%d} internal failed!", i);
      return ret;
    }

    int32_t const need_size{properties.alignedByteSize * roi_num};
    if ((output_tensors_[i] != nullptr)) {
      if (real_mem_size_[i] > need_size) {
        RCLCPP_DEBUG(rclcpp::get_logger("dnn"), 
            "Use EasyDNN internal tensor for branch {%d}, the real mem size is "
            "{%d}, need mem size is {%d}.",
            i,
            real_mem_size_[i],
            need_size);
        properties.alignedShape.dimensionSize[0] *= roi_num;
        properties.validShape.dimensionSize[0] *= roi_num;
        properties.alignedByteSize *= roi_num;
        output_dnn_tensors_[i].properties = properties;
        output_tensors_[i]->properties = properties;
      } else if (real_mem_size_[i] < need_size) {
        RCLCPP_DEBUG  (rclcpp::get_logger("dnn"), 
            "Alloc internal tensor for branch {%d}, current internal tensor real "
            "mem size {%d} is not enough, need mem size is {%d}.",
            i,
            real_mem_size_[i],
            need_size);
        RETURN_IF_FAILED(alloc_tensor(i, properties));
      } 
    } else if (output_tensors_[i] == nullptr) {
      RCLCPP_DEBUG  (rclcpp::get_logger("dnn"), 
            "Alloc internal tensor for branch {%d}, alloc mem size {%d}.",
            i,
            need_size);
      RETURN_IF_FAILED(alloc_tensor(i, properties));
    } 
  }

  roi_output_tensors_.resize(static_cast<size_t>(roi_num));
  for (int32_t i{0}; i < roi_num; ++i) {
    roi_output_tensors_[i].resize(static_cast<size_t>(output_count));
    for (int32_t j{0}; j < output_count; ++j) {
      // Manager by pool to avoid small mem allocation (low
      //   priority)
      auto const slice{std::make_shared<DNNTensorSlice>()};
      // split tensor by offset
      slice->tensor = output_tensors_[j];
      auto &properties{slice->properties};
      // assign properties
      properties = output_tensors_[j]->properties;
      // update dimension
      properties.validShape.dimensionSize[0] /= roi_num;
      properties.alignedShape.dimensionSize[0] /= roi_num;
      // update size
      properties.alignedByteSize /= roi_num;
      auto &sys_mem{slice->sysMem[0]};
      sys_mem = output_tensors_[j]->sysMem[0];
      sys_mem.memSize = static_cast<uint32_t>(properties.alignedByteSize);
      uint32_t const offset{
          static_cast<uint32_t>(properties.alignedByteSize * i)};
      sys_mem.phyAddr += offset;
      sys_mem.virAddr = reinterpret_cast<uint8_t *>(sys_mem.virAddr) + offset;
      roi_output_tensors_[i][j] = slice;
    }
  }
  return HB_DNN_SUCCESS;
}

void ModelRoiInferTask::Reset() {
  Task::Reset();
  rois_.clear();
  inputs_.clear();
  input_tensors_.clear();
  roi_output_tensors_.clear();
  output_tensors_.clear();
  real_mem_size_.clear();
}

}  // namespace easy_dnn
}  // namespace hobot
