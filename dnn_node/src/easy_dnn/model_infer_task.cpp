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

#include "easy_dnn/model_infer_task.h"

#include <algorithm>
#include <iostream>

#include "dnn/hb_dnn.h"
#include "dnn/hb_dnn_status.h"

#include "easy_dnn/common.h"
#include "easy_dnn/data_structure.h"
#include "easy_dnn/input_process.h"

namespace hobot {
namespace easy_dnn {

ModelInferTask::ModelInferTask() : Task::Task() {}

int32_t ModelInferTask::SetModel(Model *model) {
  int ret = Task::SetModel(model);
  if (ret != HB_DNN_SUCCESS) {
    return ret;
  }

  int32_t const input_count{model->GetInputCount()};
  input_tensors_.resize(static_cast<size_t>(input_count));
  input_dnn_tensors_.resize(static_cast<size_t>(input_count));
  for (int32_t i{0}; i < input_count; i++) {
    int32_t input_source{-1};
    model->GetInputSource(input_source, i);
    if (input_source == HB_DNN_INPUT_FROM_RESIZER) {
      RCLCPP_ERROR(rclcpp::get_logger("dnn"), "This is a resizer model, please use ModelRoiInferTask instead of ModelInferTask!");
      return HB_DNN_API_USE_ERROR;
    }
  }

  inputs_.resize(static_cast<size_t>(model_->GetBatchInputCount()));

  int32_t const output_count{model->GetOutputCount()};
  output_tensors_.resize(static_cast<size_t>(output_count));
  output_dnn_tensors_.resize(static_cast<size_t>(output_count));

  return HB_DNN_SUCCESS;
}

int32_t ModelInferTask::SetInputs(
    std::vector<std::shared_ptr<DNNInput>> &inputs) {

  auto const batch_input_count{model_->GetBatchInputCount()};
  // for pyramid batch inputs, should set separate inputs to processor
  // other type batch_input_count == input_count, do not support batch separate
  if ((inputs.size() != static_cast<size_t>(batch_input_count))) {
    RCLCPP_ERROR(rclcpp::get_logger("dnn"), 
      "Here inputs size != batch_input_count. Pyramid batch model must set "
      "separate inputs to process, such as: model input is batch n, you should "
      "prepare n inputs to process.");
  }

  for (int32_t i{0}; i < batch_input_count; ++i) {
    if (inputs[i] == nullptr) {
      RCLCPP_ERROR(rclcpp::get_logger("dnn"), 
        "inputs[%d] is null", i);
      return HB_DNN_INVALID_ARGUMENT;
    }
    inputs_[i] = inputs[i];
  }
  input_tensors_.resize(static_cast<size_t>(batch_input_count));
  input_dnn_tensors_.resize(static_cast<size_t>(batch_input_count));
  return HB_DNN_SUCCESS;
}

int32_t ModelInferTask::SetInputTensors(
    std::vector<std::shared_ptr<DNNTensor>> &input_tensors) {
  int32_t const input_size{static_cast<int32_t>(input_tensors.size())};
  input_tensors_ = input_tensors;
  input_dnn_tensors_.resize(static_cast<size_t>(input_size));
  for (int32_t i{0}; i < input_size; ++i) {
    if (input_tensors[i] == nullptr) {
      RCLCPP_ERROR(rclcpp::get_logger("dnn"), 
        "input_tensors[%d] is null", i);
      return HB_DNN_INVALID_ARGUMENT;
    }
    input_dnn_tensors_[i] = static_cast<hbDNNTensor>(*(input_tensors[i]));
  }
  return HB_DNN_SUCCESS;
}

int32_t ModelInferTask::GetOutputTensors(
    std::vector<std::shared_ptr<DNNTensor>> &output_tensors) {
  output_tensors = output_tensors_;
  return HB_DNN_SUCCESS;
}

void ModelInferTask::Reset() {
  Task::Reset();
  inputs_.clear();
  input_tensors_.clear();
  output_tensors_.clear();
}

int32_t ModelInferTask::ProcessInput() {
  // if model is pyramid batch input, broadcast processor
  // and do batch separate infer
  auto const input_tensor_size{1};
  auto const model_input_count{model_->GetInputCount()};
  auto const batch_size{1};
  if (model_input_count != input_tensor_size) {
    int32_t index{input_tensor_size - 1};
    for (int32_t i{model_input_count - 1}; i >= 0; i--) {
      for (int32_t j{0}; j < batch_size; j++) {
        index--;
      }
    }
  }

  int32_t process_count{0};

  // pyramid batch model's inputs_ must be separate
  for (size_t i{0U}; i < inputs_.size(); i++) {
    if (inputs_[i] != nullptr) {

      if (input_tensors_[i] == nullptr) {
        model_->GetInputTensorProperties(input_dnn_tensors_[i].properties,
                                         static_cast<int32_t>(i) / batch_size);
        // DNNInput only support seperate address
        input_dnn_tensors_[i].properties.validShape.dimensionSize[0] =
            1;
        input_dnn_tensors_[i].properties.alignedShape.dimensionSize[0] =
            1;
        input_tensors_[i] = std::shared_ptr<DNNTensor>(
            static_cast<DNNTensor *>(&input_dnn_tensors_[i]),
            [](DNNTensor const *const tensor) {});
      }

      std::shared_ptr<CropConfig> input_conf = std::make_shared<CropConfig>();
      std::shared_ptr<CropProcessor> input_processor = std::make_shared<CropProcessor>();

      hbDNNTensorProperties tensor_properties;
      hbDNNGetInputTensorProperties(&tensor_properties, model_->GetDNNHandle(), i);
      if (tensor_properties.tensorLayout == HB_DNN_LAYOUT_NHWC) {
        input_conf->width = tensor_properties.validShape.dimensionSize[2];
        input_conf->height = tensor_properties.validShape.dimensionSize[1];
      } else if (tensor_properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
        input_conf->width = tensor_properties.validShape.dimensionSize[3];
        input_conf->height = tensor_properties.validShape.dimensionSize[2];
      }

      int ret = input_processor->Process(
              input_tensors_[i], input_conf, inputs_[i]);
      if (ret != HB_DNN_SUCCESS) {
        RCLCPP_ERROR(rclcpp::get_logger("dnn"), 
          "Input process failed, input branch: %zu, ret[%d]", i, HB_DNN_RUN_TASK_FAILED);
        return HB_DNN_RUN_TASK_FAILED;
      }
      process_count++;
    } else {
      RCLCPP_ERROR(rclcpp::get_logger("dnn"), 
          "DNNInput must be set for branch:{%zu}", i);
      return HB_DNN_API_USE_ERROR;
    }
  }
  SetStatus(TaskStatus::INPUT_PROCESS_DONE);
  return HB_DNN_SUCCESS;
}

int32_t ModelInferTask::RunInfer() {

  RETURN_IF_FAILED(PrepareInferInputOutput());
  ctrl_param_.more = false;
  auto *output_ptr{output_dnn_tensors_.data()};

  {
    std::unique_lock<std::mutex> const lk{release_mtx_};
    RETURN_IF_FAILED(hbDNNInfer(&task_handle_,
                                &output_ptr,
                                input_dnn_tensors_.data(),
                                model_->GetDNNHandle(),
                                &ctrl_param_));
  }
  SetStatus(TaskStatus::INFERRING);

  return HB_DNN_SUCCESS;
}

int32_t ModelInferTask::WaitInferDone(int32_t timeout) {

  int32_t code{0};
  {
    std::unique_lock<std::mutex> const lk{release_mtx_};
    code = hbDNNWaitTaskDone(task_handle_, timeout);
  }

  if (code == HB_DNN_SUCCESS) {
    SetStatus(TaskStatus::INFERENCE_DONE);
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("dnn"), 
          "ModelInferTask Infer TimeOut %d", timeout);
    SetStatus(TaskStatus::INFERENCE_TIMEOUT);
  }

  // release dnn task resource immediately, will not cost much time
  hbDNNReleaseTask(task_handle_);
  task_handle_ = nullptr;

  return code;
}

int32_t ModelInferTask::PrepareInferInputOutput() {
  // checking input tensors
  for (size_t i{0U}; i < input_dnn_tensors_.size(); i++) {
    if (input_dnn_tensors_[i].sysMem[0].virAddr == nullptr) {
      RCLCPP_ERROR(rclcpp::get_logger("dnn"), 
              "Input is not processed or tensor is not set for branch: %zu", i);
      return HB_DNN_API_USE_ERROR;
    }
  }

  // checking output tensors, allocate if not set
  auto const output_count{model_->GetOutputCount()};
  output_dnn_tensors_.resize(static_cast<size_t>(output_count));

  for (int32_t i{0}; i < output_count; i++) {
    if (output_tensors_[i] == nullptr) {
      model_->GetOutputTensorProperties(
          output_dnn_tensors_[i].properties, i);

      // output tensor alloc pad mem
      auto const output_tensor{AllocateTensor(
          output_dnn_tensors_[i].properties)};
      if (output_tensor.get() == nullptr) {
        RCLCPP_ERROR(rclcpp::get_logger("dnn"), 
              "Allocate tensor failed, output branch: %d", i);
        return HB_DNN_API_USE_ERROR;
      }
      output_tensors_[i] = output_tensor;
      output_dnn_tensors_[i] = *output_tensor;
    }
  }
  return HB_DNN_SUCCESS;
}

}  // namespace easy_dnn
}  // namespace hobot
