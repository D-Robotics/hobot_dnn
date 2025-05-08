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

#ifndef _ROI_INFER_TASK_H_
#define _ROI_INFER_TASK_H_

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#include "dnn/hb_dnn.h"
#include "dnn/hb_dnn_status.h"

#include "easy_dnn/common.h"
#include "easy_dnn/data_structure.h"
#include "easy_dnn/input_process.h"
#include "easy_dnn/model.h"
#include "easy_dnn/task.h"

namespace hobot {
namespace easy_dnn {

class Model;
class DNNInput;
class DNNTensor;
class Task;

/**
 * A slice tensor ref to a tensor by offset & length
 */
class DNNTensorSlice : public DNNTensor {
 public:
  void Reset() override { tensor.reset(); }

  // a ref just to hold the original tensor (smart pointer)
  std::shared_ptr<DNNTensor> tensor;
};

class ModelRoiInferTask : public Task {
 public:

  ModelRoiInferTask();

  int32_t SetModel(Model *model);

  int32_t ProcessInput() override;
  
  int32_t RunInfer() override;

  int32_t WaitInferDone(int32_t timeout) override;

  void Reset();

  /**
   * Set input rois (non-required)
   * @param[in] rois
   * @return 0 if success, return defined error code otherwise
   */
  int32_t SetInputRois(std::vector<hbDNNRoi> &rois);

  /**
   * Set inputs for all roi (non-required)
   * @param[in] inputs
   * @return 0 if success, return defined error code otherwise
   */
  int32_t SetInputs(std::vector<std::shared_ptr<DNNInput>> &inputs);

  /**
   * Set input tensors for all roi (non-required)
   * @param[in] input_tensors
   * @return 0 if success, return defined error code otherwise
   */
  int32_t SetInputTensors(
      std::vector<std::shared_ptr<DNNTensor>> &input_tensors);

  /**
   * Get all output tensors, batch layout
   * @param[out] output_tensors, the size equal to model output count
   * @return 0 if success, return defined error code otherwise
   */
  int32_t GetOutputTensors(
      std::vector<std::shared_ptr<DNNTensor>> &output_tensors);

  /**
   * Get all output tensors,
   * @param[out] output_tensors, the size equal to roi count and
   *    each element's size is model output count
   * @return 0 if success, return defined error code otherwise
   */
  int32_t GetOutputTensors(std::vector<std::vector<std::shared_ptr<DNNTensor>>>
                               &output_tensors);

  /**
   * Prepare infer input tensor output tensor,
   * @return 0 if success, return defined error code otherwise
   */
  int32_t PrepareInferInputOutput();

 private:

  std::vector<hbDNNRoi> rois_;
  std::vector<std::shared_ptr<DNNInput>> inputs_;
  std::vector<std::shared_ptr<DNNTensor>> input_tensors_;
  // batch layout roi{0-m}_output{0}...
  //  roi{0-m}_output{n}
  //  the size equals to modelOutputCount
  
  std::vector<std::shared_ptr<DNNTensor>> output_tensors_;
  // roi{0}_output{0} ... roi{0}_output{n} ...
  //  roi{m}_output{0} ... roi{m}_output{n}
  //  the size equals to roiNum
  std::vector<std::vector<std::shared_ptr<DNNTensor>>> roi_output_tensors_;
  // record the real mem size for internal output tensor
  std::vector<int32_t> real_mem_size_;
};
}  // namespace easy_dnn
}  // namespace hobot

#endif  // _ROI_INFER_TASK_H_
