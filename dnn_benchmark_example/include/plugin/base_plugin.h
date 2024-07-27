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

#ifndef _PLUGIN_BASE_PLUGIN_H_
#define _PLUGIN_BASE_PLUGIN_H_

#include <memory>
#include <string>
#include <utility>

#include "rclcpp/rclcpp.hpp"

#include "input/input_data.h"

class BasePlugin {
 public:
  BasePlugin() = default;
  virtual int Init(std::string config_file, std::string config_string);

  virtual int Start() = 0;

  virtual int Stop() = 0;

  virtual ~BasePlugin() = default;

 protected:
  virtual int LoadConfig(std::string &config_string)
  {
    RCLCPP_INFO(rclcpp::get_logger("example"),
        " %s ", config_string.c_str());
    return 0;
  }

 private:
  int LoadConfigFile(std::string &config_file);
};

#endif  // _PLUGIN_BASE_PLUGIN_H_
