# Copyright (c) 2024，D-Robotics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.5)
project(dnn_benchmark_example)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(rclcpp REQUIRED)
find_package(dnn_node REQUIRED)
find_package(std_msgs REQUIRED)

# x3|rdkultra|x86
set(PREFIX_PATH x3)
set(SYS_ROOT ${CMAKE_SYSROOT})

if(PLATFORM_X3)
  message("build platform X3")
  add_definitions(-DPLATFORM_X3)
  set(PREFIX_PATH x3)
elseif(PLATFORM_Rdkultra)
  message("build platform Rdkultra")
  add_definitions(-DPLATFORM_Rdkultra)
  set(PREFIX_PATH rdkultra)
elseif(PLATFORM_X5)
  message("build platform X5")
  add_definitions(-DPLATFORM_X5)
  set(PREFIX_PATH x5)
elseif(PLATFORM_X86)
  message("build platform x86")
  add_definitions(-DPLATFORM_X86)
  set(PREFIX_PATH x86)
  set(SYS_ROOT ${THIRD_PARTY})
else()
  if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    message("invalid platform, build platform X3 default")
    add_definitions(-DPLATFORM_X3)
    set(PREFIX_PATH x3)
  elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
    message("build platform X86")
    add_definitions(-DPLATFORM_X86)
    set(PREFIX_PATH x86)
    set(SYS_ROOT ${THIRD_PARTY})
  endif()
endif()

message("PREFIX_PATH is " ${PREFIX_PATH})
message("SYS_ROOT is " ${SYS_ROOT})

include_directories(include
  ${PROJECT_SOURCE_DIR}
)
include_directories(
  ${SYS_ROOT}/usr/include/
  ${SYS_ROOT}/usr/hobot/include/
  ${SYS_ROOT}/usr/include/opencv/
)

link_directories(
  ${SYS_ROOT}/usr/lib/hbbpu/
  ${SYS_ROOT}/usr/lib/hbmedia/
  ${SYS_ROOT}/usr/lib/
  ${SYS_ROOT}/usr/hobot/lib/
)

add_executable(${PROJECT_NAME}
  src/main.cpp
  src/workflow.cpp
  src/input/data_iterator.cpp
  src/input/image_list_data_iterator.cpp
  src/plugin/base_plugin.cpp
  src/plugin/input_plugin.cpp
  src/utils/image_utils.cpp
  src/utils/utils.cpp
)

ament_target_dependencies(
  ${PROJECT_NAME}
  rclcpp
  dnn_node
  std_msgs
)


# Install executables
install(
  TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  ${PROJECT_SOURCE_DIR}/config/
  DESTINATION lib/${PROJECT_NAME}/config/
)

ament_package()
