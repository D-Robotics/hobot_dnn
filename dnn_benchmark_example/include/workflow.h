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

#ifndef _WORKFLOW_H_
#define _WORKFLOW_H_

#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "rapidjson/writer.h"
#include "rclcpp/rclcpp.hpp"

#include "include/input/input_data.h"
#include "include/plugin/base_plugin.h"
#include "include/utils/image_utils.h"
#include "include/utils/pc_queue.h"
#include "include/utils/utils.h"
#define EMPTY ""

using rclcpp::NodeOptions;

using hobot::dnn_node::DNNInput;
using hobot::dnn_node::DnnNode;
using hobot::dnn_node::DnnNodeOutput;
using hobot::dnn_node::ModelTaskType;

struct FasterRcnnOutput : public DnnNodeOutput
{
  uint64_t Predict_Start_time;
  uint64_t Predict_duration;
};

class Workflow : public DnnNode
{
 public:
  Workflow(const std::string node_name = "dnn_benchamark_node",
                 const NodeOptions &options = NodeOptions());
  ~Workflow() override;

  int WorkflowInit();

  int WorkflowStart();

  int FeedWorkflow(NV12PyramidInputPtr pyramid);

  void WorkflowRun();

  int Stop();

 protected:
  int SetNodePara() override;
  int PostProcess(const std::shared_ptr<DnnNodeOutput> &outputs)
      override;

 private:
  bool stop_ = false;
  std::shared_ptr<std::thread> thread_;

  int is_sync_mode_ = 1;

  PCQueue<NV12PyramidInputPtr> pc_queue_;
  #ifdef PLATFORM_X3
  std::string model_file_name_ = "config/X3/multitask_body_kps_960x544.hbm";
  std::string model_name_ = "";
  #endif
  #ifdef PLATFORM_Rdkultra
  std::string model_file_name_ = "config/Rdkultra/mobilenetv1_224x224_nv12_pyramid.bin";
  std::string model_name_ = "";
  #endif
  #ifdef PLATFORM_X5
  std::string model_file_name_ = "config/X5/mobilenetv1_224x224_nv12.bin";
  std::string model_name_ = "";
  #endif
  #ifdef PLATFORM_X86
  std::string model_file_name_ = "config/X3/mobilenetv1_224x224_nv12_pyramid.bin";
  std::string model_name_ = "";
  #endif
  ModelTaskType model_task_type_ = ModelTaskType::ModelInferType;

  int model_input_width_ = -1;
  int model_input_height_ = -1;
  int32_t model_output_count_ = 1;
  // box output index is 1
  // const int32_t box_output_index_ = 1;
  // kps output index is 2
  const int32_t output_index_ = 1;

  int show_fps_log = 1;
  int show_latency_log = 1;
  std::string config_file = "config/hobot_benchmark_config.json";
  int statistic_cycle = 500;
  uint64_t frame_cnt = 0;
  std::mutex frame_cnt_mtx;

  int Predict(std::vector<std::shared_ptr<DNNInput>> &inputs,
    const std::shared_ptr<std::vector<hbDNNRoi>> rois,
    std::shared_ptr<DnnNodeOutput> dnn_output);
};

#endif  // _WORKFLOW_H_
