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

#ifndef _EASY_DNN_TASK_H_
#define _EASY_DNN_TASK_H_

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

#include "rclcpp/rclcpp.hpp"

#include "dnn/hb_dnn.h"
#include "dnn/hb_dnn_status.h"

#include "easy_dnn/data_structure.h"
#include "easy_dnn/model.h"

namespace hobot {
namespace easy_dnn {

class Model;
class DNNInput;
class DNNTensor;

enum class TaskStatus : int32_t {
  ALLOCATED,
  INPUT_PROCESS_DONE,
  INFERRING,
  INFERENCE_TIMEOUT,
  INFERENCE_DONE,
  OUTPUT_PARSE_DONE,
  TERMINATED
};

class Task {
  public:

    Task();

    virtual ~Task() = default; 

    /**
     * Set task control param, include set core id, priority etc.
     * @param[in] ctrl_param
     */
    int32_t SetCtrlParam(hbDNNInferCtrlParam &ctrl_param);

    std::shared_ptr<DNNTensor> AllocateTensor(
                hbDNNTensorProperties const &tensor_properties);

    /**
     * Process input according to input description information
     * @return 0 if success, return defined error code otherwise
     */
    virtual int32_t ProcessInput() = 0;

    /**
     * Run task
     * @return 0 if success, return defined error code otherwise
     */
    virtual int32_t RunInfer() = 0;

    /**
     * Wait task infer done
     * @param[in] timeout, timeout of ms
     * @return 0 if success, return defined error code otherwise
     */
    virtual int32_t WaitInferDone(int32_t timeout) = 0;

    void Reset();

    /**
     * Set model
     * @param[in] model
     * @return 0 if success, return defined error code otherwise
     */
    int32_t SetModel(Model *model);

    void SetStatus(TaskStatus const status);

    Model *model_;
    TaskStatus task_status_;
    hbDNNTaskHandle_t task_handle_;
    hbDNNInferCtrlParam ctrl_param_;
    std::vector<hbDNNTensor> input_dnn_tensors_;
    std::vector<hbDNNTensor> output_dnn_tensors_;
    std::mutex release_mtx_;
    std::mutex task_status_mutex_;

};

}  // namespace easy_dnn
}  // namespace hobot

#endif  // _EASY_DNN_TASK_H_
