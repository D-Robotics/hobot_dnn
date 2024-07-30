// Copyright (c) 2024，D-Robotics.
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

#ifndef _INPUT_INPUT_DATA_H_
#define _INPUT_INPUT_DATA_H_

#include <memory>
#include <ostream>
#include <string>

#include "rclcpp/rclcpp.hpp"

#include "dnn_node/dnn_node.h"

using hobot::dnn_node::NV12PyramidInput;
typedef std::shared_ptr<hobot::dnn_node::NV12PyramidInput> NV12PyramidInputPtr;


#endif  // _INPUT_INPUT_DATA_H_
