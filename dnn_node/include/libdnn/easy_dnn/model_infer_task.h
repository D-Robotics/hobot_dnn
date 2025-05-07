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

#ifndef _INFER_TASK_H_
#define _INFER_TASK_H_
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <iterator>
#include <list>
#include <memory>
#include <numeric>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "easy_dnn/model.h"
#include "easy_dnn/task.h"

namespace hobot {
namespace easy_dnn {

class Task;
class Model;
class DNNInput;
class DNNTensor;

class ModelInferTask : public Task {
 public:

  ModelInferTask();

  int32_t SetModel(Model *model);

  int32_t ProcessInput() override;
  
  int32_t RunInfer() override;

  int32_t WaitInferDone(int32_t timeout) override;

  void Reset();

  /**
   *  Set all inputs
   * @param[in] inputs
   * @return 0 if success, return defined error code otherwise
   */
  int32_t SetInputs(std::vector<std::shared_ptr<DNNInput>> &inputs);

  /**
   * Set all input tensors
   * @param[in] input_tensors
   * @return 0 if success, return defined error code otherwise
   */
  int32_t SetInputTensors(
      std::vector<std::shared_ptr<DNNTensor>> &input_tensors);

  /**
   * Get all output tensors
   * @param[out] output_tensors
   * @return 0 if success, return defined error code otherwise
   */
  int32_t GetOutputTensors(
      std::vector<std::shared_ptr<DNNTensor>> &output_tensors);
  
  /**
   * Prepare infer input tensor output tensor,
   * @return 0 if success, return defined error code otherwise
   */
  int32_t PrepareInferInputOutput();

 private:
  std::vector<std::shared_ptr<DNNInput>> inputs_;
  std::vector<std::shared_ptr<DNNTensor>> input_tensors_;
  std::vector<std::shared_ptr<DNNTensor>> output_tensors_;
};
}  // namespace easy_dnn
}  // namespace hobot

#endif  // _INFER_TASK_H_
