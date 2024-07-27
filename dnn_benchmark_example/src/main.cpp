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

#include <string>
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "include/workflow.h"

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    RCLCPP_WARN(rclcpp::get_logger("example"),
                "This is dnn benchmark example!");

    auto node = std::make_shared<Workflow>("dnn_benchamark_node");

    auto ret_code = node->WorkflowInit();
    if (0 != ret_code)
    {
        RCLCPP_WARN(rclcpp::get_logger("example"),
                "Workflow init failed!");
        return -1;
    }
    node->WorkflowStart();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
