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

#ifndef _EASY_DNN_MODEL_IMPL_H_
#define _EASY_DNN_MODEL_IMPL_H_

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#include "dnn/hb_dnn.h"
#include "dnn/hb_dnn_ext.h"

#include "easy_dnn/common.h"
#include "easy_dnn/data_structure.h"

namespace hobot {
namespace easy_dnn {

class InputProcessor;

class Model {
 public:

  Model(hbDNNHandle_t const dnn_handle,
            const char *const model_name);

  ~Model();

  Model() = delete;
  // Model(const Model &) = delete;
  // Model &operator=(const Model &) = delete;
  // Model(Model &&) = delete;
  // Model &operator=(Model &&) = delete;

  /**
   * Get model handle
   * @return model handle
   */
  hbDNNHandle_t GetDNNHandle();

  /**
   * Get model name
   * @return model name
   */
  std::string GetName();

  /**
   * Get input count
   * @return input count
   */
  int32_t GetInputCount();

  /**
   * Get batch input count
   * @return batch * input count
   */
  int32_t GetBatchInputCount();

  /**
   * Get input source
   * @param[out] input_source
   * @param[in] input_index
   * @return 0 if success, return defined error code otherwise
   */
  int32_t GetInputSource(int32_t& input_source,
                                 int32_t input_index);

  /**
   * Get input tensor properties
   * @param[out] tensor_properties
   * @param[in] input_index
   * @return 0 if success, return defined error code otherwise
   */
  int32_t GetInputTensorProperties(
      hbDNNTensorProperties& tensor_properties, int32_t input_index);

  /**
   * Get output count
   * @return output count
   */
  int32_t GetOutputCount();

  /**
   * Get output name
   * @param[out] output_name
   * @param[in] output_index
   * @return 0 if success, return defined error code otherwise
   */
  int32_t GetOutputName(std::string& output_name,
                                int32_t output_index);

  /**
   * Get output tensor properties
   * @param[out] tensor_properties
   * @param[in] output_index
   * @return 0 if success, return defined error code otherwise
   */
  int32_t GetOutputTensorProperties(
      hbDNNTensorProperties& tensor_properties, int32_t output_index);

  int PrintModelInfo(std::stringstream &ss);

 private:
  hbDNNHandle_t dnn_handle_;
  std::string model_name_;
  std::vector<std::shared_ptr<InputProcessor>> input_processors_;
  int32_t batch_size_{0};
  int32_t batch_input_count_{0};
  int32_t input_count_{0};
  int32_t output_count_{0};

};

}  // namespace easy_dnn
}  // namespace hobot

#endif  // _EASY_DNN_MODEL_IMPL_H_
