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

#include <fstream>
#include <sstream>
#include "plugin/base_plugin.h"
#include "rclcpp/rclcpp.hpp"

int BasePlugin::Init(std::string config_file, std::string config_string) {
  if (!config_file.empty())
  {
    int ret_code = this->LoadConfigFile(config_file);
    if (ret_code != 0)
    {
      return ret_code;
    }
  }

  if (!config_string.empty())
  {
    int ret_code = this->LoadConfig(config_string);
    if (ret_code != 0)
    {
      return ret_code;
    }
  }

  return 0;
}

int BasePlugin::LoadConfigFile(std::string& config_file)
{
  std::ifstream ifs(config_file.c_str());
  if (!ifs)
  {
    RCLCPP_ERROR(rclcpp::get_logger("example"),
      "Open config file %s failed! ", config_file.c_str());
    return -1;
  }

  std::stringstream buffer;
  buffer << ifs.rdbuf();
  std::string contents(buffer.str());
  return this->LoadConfig(contents);
}
