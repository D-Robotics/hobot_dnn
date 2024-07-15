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

#include "easy_dnn/model.h"

#include <algorithm>
#include <fstream>
#include <memory>
#include <sstream>

#include "dnn/hb_dnn_status.h"

namespace hobot {
namespace easy_dnn {

Model::Model(hbDNNHandle_t const dnn_handle,
                     const char *const model_name)
    : dnn_handle_(dnn_handle),
      model_name_(model_name) {
  hbDNNGetInputCount(&input_count_, dnn_handle_);
  hbDNNGetOutputCount(&output_count_, dnn_handle_);

  hbDNNTensorProperties tensor_properties;
  hbDNNGetInputTensorProperties(&tensor_properties, dnn_handle_, 0);
  // here batch size need update dnn
  batch_size_ = tensor_properties.alignedShape.dimensionSize[0];
  batch_input_count_ = input_count_ * batch_size_;

}

hbDNNHandle_t Model::GetDNNHandle() { return dnn_handle_; }

std::string Model::GetName() { return model_name_; }

int32_t Model::GetInputCount() { return input_count_; }

int32_t Model::GetBatchInputCount() { return batch_input_count_; }

int32_t Model::GetInputTensorProperties(
    hbDNNTensorProperties &tensor_properties, int32_t input_index) {
  RETURN_IF_FAILED(hbDNNGetInputTensorProperties(
      &tensor_properties, dnn_handle_, input_index));
  return HB_DNN_SUCCESS;
}

int32_t Model::GetOutputCount() { return output_count_; }

int32_t Model::GetOutputName(std::string &output_name,
                                 int32_t output_index) {
  const char *name{nullptr};
  RETURN_IF_FAILED(hbDNNGetOutputName(&name, dnn_handle_, output_index));
  output_name = name;
  return HB_DNN_SUCCESS;
}

int32_t Model::GetInputSource(int32_t &input_source, int32_t input_index) {
  int32_t source{0};
  RETURN_IF_FAILED(hbDNNGetInputSource(&source, dnn_handle_, input_index));
  input_source = source;
  return HB_DNN_SUCCESS;
}

int32_t Model::GetOutputTensorProperties(
    hbDNNTensorProperties &tensor_properties, int32_t output_index) {
  return hbDNNGetOutputTensorProperties(
      &tensor_properties, dnn_handle_, output_index);
}

int Model::PrintModelInfo(std::stringstream &ss) {
  ss << "\nModel Info:\n"
    << "name: " << model_name_ << ".\n";

  std::vector<std::string> types = {"input", "output"};
  for (auto type: types) {
    ss << "[" + type << "]\n";
    int count = 0;
    if (type == "input") {
      count = input_count_;
    } else if (type == "output"){
      count = output_count_;
    }

    for (int i = 0; i < count; i++){
      hbDNNTensorProperties tensor_properties;
      if (type == "input") {
        hbDNNGetInputTensorProperties(&tensor_properties, dnn_handle_, i);
      } else if (type == "output"){
        hbDNNGetOutputTensorProperties(&tensor_properties, dnn_handle_, i);
      }

      ss << " - (" << i << ") ";
      if (tensor_properties.tensorLayout == HB_DNN_LAYOUT_NHWC) {
        ss << "Layout: NHWC";
      } else if (tensor_properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
        ss << "Layout: NCHW";
      } else if (tensor_properties.tensorLayout == HB_DNN_LAYOUT_NONE) {
        ss << "Layout: NONE";
      }
      ss << ", Shape: [";
      for (int j = 0; j < 4; j++) {
        ss << tensor_properties.validShape.dimensionSize[j];
        if (j != 3) {
          ss << ", ";
        }
      }
      ss << "], Type: ";
      switch (tensor_properties.tensorType)
      { 
        case HB_DNN_IMG_TYPE_Y: ss << "HB_DNN_IMG_TYPE_Y"; break;
        case HB_DNN_IMG_TYPE_NV12: ss << "HB_DNN_IMG_TYPE_NV12"; break;
        case HB_DNN_IMG_TYPE_NV12_SEPARATE: ss << "HB_DNN_IMG_TYPE_NV12_SEPARATE"; break;
        case HB_DNN_IMG_TYPE_YUV444: ss << "HB_DNN_IMG_TYPE_YUV444"; break;
        case HB_DNN_IMG_TYPE_RGB: ss << "HB_DNN_IMG_TYPE_RGB"; break;
        case HB_DNN_IMG_TYPE_BGR: ss << "HB_DNN_IMG_TYPE_BGR"; break;
        case HB_DNN_TENSOR_TYPE_S4: ss << "HB_DNN_TENSOR_TYPE_S4"; break;
        case HB_DNN_TENSOR_TYPE_U4: ss << "HB_DNN_TENSOR_TYPE_U4"; break;
        case HB_DNN_TENSOR_TYPE_S8: ss << "HB_DNN_TENSOR_TYPE_S8"; break;
        case HB_DNN_TENSOR_TYPE_U8: ss << "HB_DNN_TENSOR_TYPE_U8"; break;
        case HB_DNN_TENSOR_TYPE_F16: ss << "HB_DNN_TENSOR_TYPE_F16"; break;
        case HB_DNN_TENSOR_TYPE_S16: ss << "HB_DNN_TENSOR_TYPE_S16"; break;
        case HB_DNN_TENSOR_TYPE_U16: ss << "HB_DNN_TENSOR_TYPE_U16"; break;
        case HB_DNN_TENSOR_TYPE_F32: ss << "HB_DNN_TENSOR_TYPE_F32"; break;
        case HB_DNN_TENSOR_TYPE_S32: ss << "HB_DNN_TENSOR_TYPE_S32"; break;
        case HB_DNN_TENSOR_TYPE_U32: ss << "HB_DNN_TENSOR_TYPE_U32"; break;
        case HB_DNN_TENSOR_TYPE_F64: ss << "HB_DNN_TENSOR_TYPE_F64"; break;
        case HB_DNN_TENSOR_TYPE_S64: ss << "HB_DNN_TENSOR_TYPE_S64"; break;
        case HB_DNN_TENSOR_TYPE_U64: ss << "HB_DNN_TENSOR_TYPE_U64"; break;
        case HB_DNN_TENSOR_TYPE_MAX: ss << "HB_DNN_TENSOR_TYPE_MAX"; break;
        default: ss << "ERROR TYPE!"; break;
      }
      ss << ".\n";
    }
  }
  return 0;
}


Model::~Model() {
  dnn_handle_ = nullptr;
}

}  // namespace easy_dnn
}  // namespace hobot
