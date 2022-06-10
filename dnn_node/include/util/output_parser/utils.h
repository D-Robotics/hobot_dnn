// Copyright (c) 2022，Horizon Robotics.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef _OUTPUT_PARSER_UTILS_H_
#define _OUTPUT_PARSER_UTILS_H_

#include <string>
#include <vector>
#include <memory>

#include "dnn_node/dnn_node_data.h"

using hobot::dnn_node::DNNTensor;

/**
 *
 * @param[in] tensor
 * @param[out] h_index
 * @param[out] w_index
 * @param[out] c_index
 * @return 0f if success
 */
int get_tensor_hwc_index(std::shared_ptr<DNNTensor> tensor,
                         int *h_index,
                         int *w_index,
                         int *c_index);

/**
 *
 * @param[in] tensor
 * @param[out] h_index
 * @param[out] w_index
 * @param[out] c_index
 * @return 0f if success
 */
int get_tensor_hw(std::shared_ptr<DNNTensor> tensor, int *height, int *width);


/**
 *
 * @param tensor
 * @param height
 * @param width
 * @return
 */
int get_tensor_aligned_hw(std::shared_ptr<DNNTensor> tensor,
                          int *height, int *width);

#endif  // _OUTPUT_PARSER_UTILS_H_
