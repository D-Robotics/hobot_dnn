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

  for (int i = 0; i < input_count; i++) {
    hbDNNTensorProperties properties;
    model_->GetInputTensorProperties(properties, i);
    int dim0 = properties.validShape.dimensionSize[0];
    int dim1 = properties.validShape.dimensionSize[1];
    int dim2 = properties.validShape.dimensionSize[2];
    int dim3 = properties.validShape.dimensionSize[3];
    int tensorType = properties.tensorType;
    if (dim0 == 1 && dim1 == 4 && dim2 == 0 && dim3 == 0 
          && tensorType == HB_DNN_TENSOR_TYPE_S32) {
      roi_index_ = i;
      break;
    }
  }
  if (roi_index_ == -1) {
    return HB_DNN_INVALID_MODEL;
  }

  return HB_DNN_SUCCESS;
}

int32_t ModelRoiInferTask::SetInputRois(std::vector<hbDNNRoi> &rois) {
  auto roi_size = rois.size();
  roi_mem_.resize(roi_size);
  for (auto i = 0; i < roi_size; ++i) {
    int32_t mem_size = 4 * sizeof(int32_t);
    hbUCPMallocCached(&roi_mem_[i], mem_size, 0);
    int32_t *roi_data = reinterpret_cast<int32_t *>(roi_mem_[i].virAddr);
    // The order of filling in the corner points of roi tensor is left, top, right, bottom
    roi_data[0] = rois[i].left;
    roi_data[1] = rois[i].top;
    roi_data[2] = rois[i].right;
    roi_data[3] = rois[i].bottom;
    // make sure cahced mem data is flushed to DDR before inference
    hbUCPMemFlush(&roi_mem_[i], HB_SYS_MEM_CACHE_CLEAN);
  }

  return HB_DNN_SUCCESS;
}

int32_t ModelRoiInferTask::SetInputs(
    std::vector<std::shared_ptr<DNNInput>> &inputs) {
  auto roi_size = roi_mem_.size();
  if (inputs.size() != roi_size) {
    RCLCPP_ERROR(rclcpp::get_logger("dnn"), 
              "Input Size [%zu] Not Equal to Roi Size [%zu]!", inputs.size(), roi_size);
    return HB_DNN_INVALID_ARGUMENT;
  }
  for (int32_t i{0}; i < inputs.size(); ++i) {
    if (!inputs[i]) {
      RCLCPP_ERROR(rclcpp::get_logger("dnn"), 
                "Set Inputs[%zu] failed", i);
      return HB_DNN_INVALID_ARGUMENT;
    }
    inputs_.push_back(inputs[i]);
  }
  return HB_DNN_SUCCESS;
}

int32_t ModelRoiInferTask::SetInputTensors(
    std::vector<std::shared_ptr<DNNTensor>> &input_tensors) {
  size_t const input_size{input_tensors.size()};
  int32_t const input_count{model_->GetInputCount()};

  if (input_size != input_count - 1) {
    RCLCPP_ERROR(rclcpp::get_logger("dnn"), 
          "input tensor size[%zu] is not equal to model input size[%zu]", input_size, input_count - 1);
    return HB_DNN_API_USE_ERROR;
  }

  auto roi_size = roi_mem_.size();
  for (size_t n{0U}; n < roi_size; ++n) {
    for (size_t i{0U}; i < input_size; ++i) {
      if (input_tensors[i] == nullptr) {
        RCLCPP_ERROR(rclcpp::get_logger("dnn"), 
            "input_tensors [%zu] is null", i);
        return HB_DNN_INVALID_ARGUMENT;
      }
      roi_input_tensors_[n][i] = input_tensors[i];
      roi_input_dnn_tensors_[n][i] = static_cast<hbDNNTensor>(*(input_tensors[i]));
    }
  }
  return HB_DNN_SUCCESS;
}

int32_t ModelRoiInferTask::GetOutputTensors(
    std::vector<std::shared_ptr<DNNTensor>> &output_tensors) {
  for (int32_t n{0}; n < roi_output_tensors_.size(); n++) {
    for (int32_t i{0}; i < roi_output_tensors_[n].size(); i++) {
      output_tensors.push_back(roi_output_tensors_[n][i]);
    }
  }
  return HB_DNN_SUCCESS;
}

int32_t ModelRoiInferTask::GetOutputTensors(
    std::vector<std::vector<std::shared_ptr<DNNTensor>>> &output_tensors) {
  output_tensors = roi_output_tensors_;
  return HB_DNN_SUCCESS;
}

int32_t ModelRoiInferTask::ProcessInput() {
  // checking output tensors, allocate if not set
  int32_t roi_num{static_cast<int32_t>(roi_mem_.size())};
  int32_t const input_count{model_->GetInputCount()};

  roi_input_dnn_tensors_.resize(static_cast<size_t>(roi_num));
  roi_input_tensors_.resize(static_cast<size_t>(roi_num));

  for (size_t i{0U}; i < inputs_.size(); i ++) {
    roi_input_dnn_tensors_[i].resize(static_cast<size_t>(input_count));
    roi_input_tensors_[i].resize(static_cast<size_t>(input_count));

    if (inputs_[i] == nullptr) {
      RCLCPP_ERROR(rclcpp::get_logger("dnn"), 
      "DNNInput must be set for branch:{%zu}", i);
      return HB_DNN_INVALID_ARGUMENT;
    }

    auto pyramid_input = std::dynamic_pointer_cast<NV12PyramidInput>(inputs_[i]);

    for (size_t j{0U}; j < 2; j++) {
      if (roi_input_tensors_[i][j] == nullptr) {
        model_->GetInputTensorProperties(roi_input_dnn_tensors_[i][j].properties,
                                        static_cast<int32_t>(j));

        roi_input_tensors_[i][j] = std::shared_ptr<DNNTensor>(
            static_cast<DNNTensor *>(&roi_input_dnn_tensors_[i][j]),
            [](DNNTensor const *const tensor) {});
                
        roi_input_tensors_[i][j]->properties.validShape.dimensionSize[1] = pyramid_input->height;
        roi_input_tensors_[i][j]->properties.validShape.dimensionSize[2] = pyramid_input->width;
        if (j == 1) {
          // uv input
          roi_input_tensors_[i][j]->properties.validShape.dimensionSize[1] /= 2;
          roi_input_tensors_[i][j]->properties.validShape.dimensionSize[2] /= 2;
        }
        
        // RDK S600 need ALIGN_64
        roi_input_tensors_[i][j]->properties.stride[1] =
              ALIGN_32(roi_input_tensors_[i][j]->properties.stride[2] *
              roi_input_tensors_[i][j]->properties.validShape.dimensionSize[2]);
        roi_input_tensors_[i][j]->properties.stride[0] =
              roi_input_tensors_[i][j]->properties.stride[1] *
              roi_input_tensors_[i][j]->properties.validShape.dimensionSize[1];
      }
    }

    std::shared_ptr<FillProcessor> input_processor = std::make_shared<FillProcessor>();
    int ret = input_processor->Process(
            roi_input_tensors_[i][0], roi_input_tensors_[i][1], inputs_[i]);
    if (ret != HB_DNN_SUCCESS) {
      RCLCPP_ERROR(rclcpp::get_logger("dnn"), 
        "Input process failed, input branch: %zu, ret[%d]", i, HB_DNN_RUN_TASK_FAILED);
      return HB_DNN_RUN_TASK_FAILED;
    }

    model_->GetInputTensorProperties(roi_input_dnn_tensors_[i][roi_index_].properties, roi_index_);
    roi_input_dnn_tensors_[i][roi_index_].sysMem = roi_mem_[i];
    roi_input_dnn_tensors_[i][roi_index_].sysMem.memSize = 16;
  }
  SetStatus(TaskStatus::INPUT_PROCESS_DONE);
  return HB_DNN_SUCCESS;
}

int32_t ModelRoiInferTask::RunInfer() {
  
  int ret = PrepareInferInputOutput();
  if (ret != 0) {
    return ret;
  }

  {
    std::unique_lock<std::mutex> const lk{release_mtx_};    
    int32_t roi_num{static_cast<int32_t>(roi_mem_.size())};
    for (int32_t n{0}; n < roi_num; n++) {
      hbDNNInferV2(&task_handle_,
                roi_output_dnn_tensors_[n].data(),
                roi_input_dnn_tensors_[n].data(),
                model_->GetDNNHandle());
    }
    hbUCPSubmitTask(task_handle_, &ctrl_param_);
  }
  SetStatus(TaskStatus::INFERRING);
  return HB_DNN_SUCCESS;
}

int32_t ModelRoiInferTask::WaitInferDone(int32_t timeout) {
  int32_t code{HB_DNN_SUCCESS};
  {
    std::unique_lock<std::mutex> const lk{release_mtx_};
    code = hbUCPWaitTaskDone(task_handle_, timeout);
  }

  if (code == HB_DNN_SUCCESS) {
    SetStatus(TaskStatus::INFERENCE_DONE);
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("dnn"), 
          "ModelRoiInferTask Infer TimeOut %d", timeout);
    SetStatus(TaskStatus::INFERENCE_TIMEOUT);
  }
  // release dnn task resource immediately, will not cost much time
  hbUCPReleaseTask(task_handle_);
  task_handle_ = nullptr;
  return code;
}

int32_t ModelRoiInferTask::PrepareInferInputOutput() {
  // checking output tensors, allocate if not set
  int32_t roi_num{static_cast<int32_t>(roi_mem_.size())};
  auto const output_count{model_->GetOutputCount()};

  roi_output_dnn_tensors_.resize(static_cast<size_t>(roi_num));
  roi_output_tensors_.resize(static_cast<size_t>(roi_num));

  for (int32_t n{0}; n < roi_num; n++) {
    roi_output_dnn_tensors_[n].resize(static_cast<size_t>(output_count));
    roi_output_tensors_[n].resize(static_cast<size_t>(output_count));
    for (int32_t i{0}; i < output_count; i++) {
      if (roi_output_tensors_[n][i] == nullptr) {
        model_->GetOutputTensorProperties(
            roi_output_dnn_tensors_[n][i].properties, i);

        // output tensor alloc pad mem
        auto const output_tensor{AllocateTensor(
            roi_output_dnn_tensors_[n][i].properties)};
        if (output_tensor.get() == nullptr) {
          RCLCPP_ERROR(rclcpp::get_logger("dnn"), 
                "Allocate tensor failed, output branch: %d", i);
          return HB_DNN_OUT_OF_MEMORY;
        }
        roi_output_tensors_[n][i] = output_tensor;
        roi_output_dnn_tensors_[n][i] = *output_tensor;
      }  
    }
  }
  return HB_DNN_SUCCESS;
}

ModelRoiInferTask::~ModelRoiInferTask() {
  for (auto &roi_mem: roi_mem_) {
    hbUCPFree(&roi_mem);
  }
  inputs_.clear();
  roi_input_tensors_.clear();
  roi_input_dnn_tensors_.clear();
  roi_output_dnn_tensors_.clear();
  roi_output_tensors_.clear();
}

}  // namespace easy_dnn
}  // namespace hobot
