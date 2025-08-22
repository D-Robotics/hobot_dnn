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
project(dnn_node)

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
find_package(std_msgs REQUIRED)
find_package(OpenCV 4 REQUIRED)

# libdnn|ucp
set(PREFIX_PATH libdnn)
set(SYS_ROOT ${CMAKE_SYSROOT})

if(PLATFORM_X3)
  message("build platform X3")
  add_definitions(-DPLATFORM_X3)
  add_definitions(-DBPU_LIBDNN)
  set(PREFIX_PATH libdnn)
elseif(PLATFORM_Rdkultra)
  message("build platform Rdkultra")
  add_definitions(-DPLATFORM_Rdkultra)
  add_definitions(-DBPU_LIBDNN)
  set(PREFIX_PATH libdnn)
elseif(PLATFORM_X5)
  message("build platform X5")
  add_definitions(-DPLATFORM_X5)
  add_definitions(-DBPU_LIBDNN)
  set(PREFIX_PATH libdnn)
elseif(PLATFORM_S100)
  message("build platform S100")
  add_definitions(-DPLATFORM_S100)
  add_definitions(-DBPU_UCP)
  set(PREFIX_PATH ucp)
elseif(PLATFORM_X86)
  message("build platform x86")
  add_definitions(-DPLATFORM_X86)
  set(SYS_ROOT ${THIRD_PARTY})
  add_definitions(-DBPU_LIBDNN)
  set(PREFIX_PATH libdnn)
else()
  if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    message("invalid platform, build platform X3 default")
    add_definitions(-DPLATFORM_X3)
    set(PLATFORM_X3 ON)
    add_definitions(-DBPU_LIBDNN)
    set(PREFIX_PATH libdnn)
  elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
    message("build platform X86")
    add_definitions(-DPLATFORM_X86)
    set(PLATFORM_X86 ON)
    set(SYS_ROOT ${THIRD_PARTY})
  endif()
endif()

message("PREFIX_PATH is " ${PREFIX_PATH})
message("SYS_ROOT is " ${SYS_ROOT})

include_directories(
  include/${PREFIX_PATH}/
  include/util/
  ${PROJECT_SOURCE_DIR}
)
include_directories(
        ${SYS_ROOT}/usr/include
        ${SYS_ROOT}/usr/include/hobot/
	      ${SYS_ROOT}/usr/hobot/include/
        ${SYS_ROOT}/usr/include/opencv/
)

link_directories(
  ${SYS_ROOT}/usr/lib/hbbpu/
  ${SYS_ROOT}/usr/lib/
  ${SYS_ROOT}/usr/hobot/lib/
)
 
if(PLATFORM_X3)
    # Build dnn_node library
  add_library(${PROJECT_NAME} SHARED
    src/dnn_node.cpp
    src/${PREFIX_PATH}/dnn_node_impl.cpp
    src/${PREFIX_PATH}/image_proc.cpp
    src/${PREFIX_PATH}/utils.cpp
    src/${PREFIX_PATH}/easy_dnn/input_process.cpp
    src/${PREFIX_PATH}/easy_dnn/model.cpp
    src/${PREFIX_PATH}/easy_dnn/model_infer_task.cpp
    src/${PREFIX_PATH}/easy_dnn/model_roi_infer_task.cpp
    src/${PREFIX_PATH}/easy_dnn/task.cpp
    src/threads/threadpool.cpp
    src/output_parser/detection/nms.cpp
    src/output_parser/detection/ptq_yolo3_darknet_output_parser.cpp
    src/output_parser/detection/ptq_yolo2_output_parser.cpp
    src/output_parser/detection/ptq_yolo5_output_parser.cpp
    src/output_parser/detection/ptq_yolo8_output_parser.cpp
    src/output_parser/detection/ptq_yolo10_output_parser.cpp
    src/output_parser/detection/fasterrcnn_output_parser.cpp
    src/output_parser/detection/ptq_efficientdet_output_parser.cpp
    src/output_parser/detection/ptq_ssd_output_parser.cpp
    src/output_parser/detection/fcos_output_parser.cpp
    src/output_parser/classification/ptq_classification_output_parser.cpp
    src/output_parser/segmentation/ptq_unet_output_parser.cpp
    src/output_parser/segmentation/ptq_yolo8_seg_output_parser.cpp
    src/output_parser/segmentation/ptq_stdc_output_parser.cpp
  )

  target_link_libraries(${PROJECT_NAME}
    alog
    dnn
    cnn_intf
    ion
    hbrt_bernoulli_aarch64
  )
elseif(PLATFORM_Rdkultra)
  add_library(${PROJECT_NAME} SHARED
    src/dnn_node.cpp
    src/${PREFIX_PATH}/dnn_node_impl.cpp
    src/${PREFIX_PATH}/image_proc.cpp
    src/${PREFIX_PATH}/utils.cpp
    src/${PREFIX_PATH}/easy_dnn/input_process.cpp
    src/${PREFIX_PATH}/easy_dnn/model.cpp
    src/${PREFIX_PATH}/easy_dnn/model_infer_task.cpp
    src/${PREFIX_PATH}/easy_dnn/model_roi_infer_task.cpp
    src/${PREFIX_PATH}/easy_dnn/task.cpp
    src/threads/threadpool.cpp
    src/output_parser/detection/nms.cpp
    src/output_parser/detection/ptq_yolo3_darknet_output_parser.cpp
    src/output_parser/detection/ptq_yolo2_output_parser.cpp
    src/output_parser/detection/ptq_yolov5x_output_parser.cpp
    src/output_parser/detection/ptq_yolo8_output_parser.cpp
    src/output_parser/detection/ptq_yolo10_output_parser.cpp
    src/output_parser/detection/fasterrcnn_output_parser.cpp
    src/output_parser/detection/ptq_efficientdet_output_parser.cpp
    src/output_parser/detection/ptq_ssd_output_parser.cpp
    src/output_parser/detection/fcos_output_parser.cpp
    src/output_parser/classification/ptq_classification_output_parser.cpp
    src/output_parser/segmentation/ptq_unet_output_parser.cpp
    src/output_parser/segmentation/ptq_yolo8_seg_output_parser.cpp
    src/output_parser/segmentation/ptq_stdc_output_parser.cpp
  )
  target_link_libraries(${PROJECT_NAME}
    alog
    dnn
    cnn_intf
    ion
    hbrt_bayes_aarch64
  )
elseif(PLATFORM_X5)
  add_library(${PROJECT_NAME} SHARED
    src/dnn_node.cpp
    src/${PREFIX_PATH}/dnn_node_impl.cpp
    src/${PREFIX_PATH}/image_proc.cpp
    src/${PREFIX_PATH}/utils.cpp
    src/${PREFIX_PATH}/easy_dnn/input_process.cpp
    src/${PREFIX_PATH}/easy_dnn/model.cpp
    src/${PREFIX_PATH}/easy_dnn/model_infer_task.cpp
    src/${PREFIX_PATH}/easy_dnn/model_roi_infer_task.cpp
    src/${PREFIX_PATH}/easy_dnn/task.cpp
    src/threads/threadpool.cpp
    src/output_parser/detection/nms.cpp
    src/output_parser/detection/ptq_yolo3_darknet_output_parser.cpp
    src/output_parser/detection/ptq_yolo2_output_parser.cpp
    src/output_parser/detection/ptq_yolov5x_output_parser.cpp
    src/output_parser/detection/ptq_yolo8_output_parser.cpp
    src/output_parser/detection/ptq_yolo10_output_parser.cpp
    src/output_parser/detection/fasterrcnn_output_parser.cpp
    # src/output_parser/detection/ptq_efficientdet_output_parser.cpp
    src/output_parser/detection/ptq_ssd_output_parser.cpp
    src/output_parser/detection/fcos_output_parser.cpp
    src/output_parser/classification/ptq_classification_output_parser.cpp
    src/output_parser/segmentation/ptq_unet_output_parser.cpp
    src/output_parser/segmentation/ptq_yolo8_seg_output_parser.cpp
    src/output_parser/segmentation/ptq_stdc_output_parser.cpp
  )
  target_link_libraries(${PROJECT_NAME}
    alog
    gflags
    dnn
    cnn_intf
    hbrt_bayes_aarch64
  )
elseif(PLATFORM_S100)
  add_library(${PROJECT_NAME} SHARED
    src/dnn_node.cpp
    src/${PREFIX_PATH}/dnn_node_impl.cpp
    src/${PREFIX_PATH}/image_proc.cpp
    src/${PREFIX_PATH}/utils.cpp
    src/${PREFIX_PATH}/easy_dnn/input_process.cpp
    src/${PREFIX_PATH}/easy_dnn/model.cpp
    src/${PREFIX_PATH}/easy_dnn/model_infer_task.cpp
    src/${PREFIX_PATH}/easy_dnn/model_roi_infer_task.cpp
    src/${PREFIX_PATH}/easy_dnn/task.cpp
    src/output_parser/detection/nms.cpp
    src/threads/threadpool.cpp
    src/output_parser/detection/ptq_yolo3_darknet_output_parser.cpp
    src/output_parser/detection/ptq_yolo2_output_parser.cpp
    src/output_parser/detection/ptq_yolov5x_output_parser.cpp
    src/output_parser/detection/ptq_yolo8_output_parser.cpp
    src/output_parser/detection/ptq_yolo10_output_parser.cpp
    # src/output_parser/detection/fasterrcnn_output_parser.cpp
    # src/output_parser/detection/ptq_efficientdet_output_parser.cpp
    # src/output_parser/detection/ptq_ssd_output_parser.cpp
    # src/output_parser/detection/fcos_output_parser.cpp
    src/output_parser/classification/ptq_classification_output_parser.cpp
    src/output_parser/segmentation/ptq_unet_output_parser.cpp
    src/output_parser/segmentation/ptq_yolo8_seg_output_parser.cpp
    # src/output_parser/segmentation/ptq_stdc_output_parser.cpp
  )
  target_link_libraries(${PROJECT_NAME}
    dnn
    hbucp
    pthread
    rt
    dl
  )
elseif(PLATFORM_X86)
  add_library(${PROJECT_NAME} SHARED
    src/dnn_node.cpp
    src/${PREFIX_PATH}/dnn_node_impl.cpp
    src/${PREFIX_PATH}/image_proc.cpp
    src/${PREFIX_PATH}/utils.cpp
    src/${PREFIX_PATH}/easy_dnn/input_process.cpp
    src/${PREFIX_PATH}/easy_dnn/model.cpp
    src/${PREFIX_PATH}/easy_dnn/model_infer_task.cpp
    src/${PREFIX_PATH}/easy_dnn/model_roi_infer_task.cpp
    src/${PREFIX_PATH}/easy_dnn/task.cpp
    src/output_parser/detection/nms.cpp
    src/threads/threadpool.cpp
    src/output_parser/detection/ptq_yolo3_darknet_output_parser.cpp
    src/output_parser/detection/ptq_yolo2_output_parser.cpp
    src/output_parser/detection/ptq_yolo8_output_parser.cpp
    src/output_parser/detection/ptq_yolo10_output_parser.cpp
    src/output_parser/detection/fasterrcnn_output_parser.cpp
    src/output_parser/detection/ptq_ssd_output_parser.cpp
    src/output_parser/detection/fcos_output_parser.cpp
    src/output_parser/classification/ptq_classification_output_parser.cpp
    src/output_parser/segmentation/ptq_unet_output_parser.cpp
    src/output_parser/segmentation/ptq_yolo8_seg_output_parser.cpp
  )

  target_link_libraries(${PROJECT_NAME}
    dnn
    hbdk_sim_x86
  )
endif()


ament_target_dependencies(
  ${PROJECT_NAME}
  rclcpp
  std_msgs
  OpenCV
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# Install libraries
install(TARGETS dnn_node
  DESTINATION lib/)

# Install include files
install(
  FILES include/dnn_node.h include/${PREFIX_PATH}/dnn_node/dnn_node_data.h
  DESTINATION include/dnn_node/
)

install(
  FILES include/util/image_proc.h
  DESTINATION include/dnn_node/util/
)

install(
  DIRECTORY include/dnn_node/util/output_parser/
  DESTINATION include/dnn_node/util/output_parser/
)

install(
  DIRECTORY include/${PREFIX_PATH}/easy_dnn/
  DESTINATION include/easy_dnn/
)

# Causes the visibility macros to use dllexport rather than dllimport,
# which is appropriate when building the dll but not consuming it.
target_compile_definitions(${PROJECT_NAME}
  PRIVATE "RCLCPP_BUILDING_LIBRARY")

install(
  TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# specific order: dependents before dependencies
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_targets(${PROJECT_NAME})

# Build example
# add_subdirectory(example)

if(BUILD_TESTING)
  # Build gtest
  add_subdirectory(test)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
