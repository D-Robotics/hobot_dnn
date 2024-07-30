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

#include <iostream>
#include <ostream>

#include "include/workflow.h"
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "rapidjson/writer.h"
#include "include/plugin/input_plugin.h"

Workflow::Workflow(const std::string node_name,
                               const NodeOptions &options) :
                               DnnNode(node_name, options)
{
  this->declare_parameter<int>("show_fps_log", show_fps_log);
  this->declare_parameter<int>("show_latency_log", show_latency_log);
  this->declare_parameter<int>("statistic_cycle", statistic_cycle);
  this->declare_parameter<int>("is_sync_mode", is_sync_mode_);
  this->declare_parameter<std::string>("config_file", config_file);
  this->declare_parameter<std::string>("model_file_name", model_file_name_);
  this->declare_parameter<std::string>("model_name", model_name_);

  this->get_parameter<int>("show_fps_log", show_fps_log);
  this->get_parameter<int>("show_latency_log", show_latency_log);
  this->get_parameter<int>("statistic_cycle", statistic_cycle);
  this->get_parameter<int>("is_sync_mode", is_sync_mode_);
  this->get_parameter<std::string>("config_file", config_file);
  this->get_parameter<std::string>("model_file_name", model_file_name_);
  this->get_parameter<std::string>("model_name", model_name_);

  std::stringstream ss;
  ss << "Parameter:"
     << "\n show_fps_log: " << show_fps_log
     << "\n show_latency_log: " << show_latency_log
     << "\n statistic_cycle: " << statistic_cycle
     << "\n is_sync_mode_: " << is_sync_mode_
     << "\n config_file: " << config_file
     << "\n model_file_name: " << model_file_name_
     << "\n model_name: " << model_name_;
  RCLCPP_WARN(rclcpp::get_logger("example"), "%s", ss.str().c_str());

  if (Init() != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("example"),
      "Init failed!");
  }

  if (GetModelInputSize(0, model_input_width_, model_input_height_) < 0) {
    RCLCPP_ERROR(rclcpp::get_logger("example"),
      "Get model input size fail!");
  } else {
    RCLCPP_INFO(rclcpp::get_logger("example"),
    "The model input width is %d and height is %d",
    model_input_width_, model_input_height_);
  }
}

Workflow::~Workflow() {}

int Workflow::WorkflowInit()
{
  RCLCPP_INFO(rclcpp::get_logger("example"), "this is WorkflowInit!!!");
// Parsing config
  std::ifstream ifs(config_file.c_str());
  rapidjson::IStreamWrapper isw(ifs);
  rapidjson::Document document;
  document.ParseStream(isw);
  if (document.HasParseError())
  {
    RCLCPP_ERROR(rclcpp::get_logger("example"), "Parsing config file failed");
    return -1;
  }
  if (!(document.HasMember("input_config")))
  {
    RCLCPP_ERROR(rclcpp::get_logger("example"), "Missing config for input");
    return -2;
  }
// Input plugin
  RCLCPP_DEBUG(rclcpp::get_logger("example"), "this is input_plg init!!!");
  auto input_plg = InputProducerPlugin::GetInstance();
  if (!input_plg)
  {
    RCLCPP_ERROR(rclcpp::get_logger("example"), "get input_plg failed!!!");
  }
  input_plg->set_model_input_width_height(model_input_width_,
                                         model_input_height_);
  int ret_code =
      input_plg->Init(EMPTY, json_to_string(document["input_config"]));
  input_plg->registerWork(this);

  if (0 != ret_code)
  {
    RCLCPP_ERROR(rclcpp::get_logger("example"), "Input plugin init failed");
    return -3;
  }
  return 0;
}

int Workflow::WorkflowStart()
{
  InputProducerPlugin::GetInstance()->Start();
  stop_ = false;
  thread_ = std::make_shared<std::thread>([this] { WorkflowRun(); });
  if (!thread_)
  {
    RCLCPP_ERROR(rclcpp::get_logger("example"), "Start workflow thread failed");
    return -1;
  }
  RCLCPP_INFO(rclcpp::get_logger("example"), "Workflow start");
  return 0;
}

int Workflow::FeedWorkflow(NV12PyramidInputPtr pyramid)
{
  pc_queue_.put(pyramid);
  return 0;
}

void Workflow::WorkflowRun()
{
  while (!stop_)
  {
    NV12PyramidInputPtr pyramid;
    if (!pc_queue_.get(pyramid, 1000))
    {
      if (stop_)
      {
        break;
      }
      continue;
    }

    if (!pyramid)
    {
      RCLCPP_ERROR(rclcpp::get_logger("example"), "pyramid is null !");
      InputProducerPlugin::GetInstance()->Release();
      continue;
    }
    auto inputs = std::vector<std::shared_ptr<DNNInput>>{pyramid};
    auto dnn_output = std::make_shared<FasterRcnnOutput>();
    auto start = currentMicroseconds();
    dnn_output->Predict_Start_time = start;
    uint32_t ret = 0;
    ret = Predict(inputs, nullptr, dnn_output);
    if (ret != 0)
    {
      RCLCPP_ERROR(rclcpp::get_logger("example"), "Run predict failed!");
    }
  }
}

int Workflow::Stop()
{
  stop_ = true;
  thread_->join();
  RCLCPP_INFO(rclcpp::get_logger("example"), "Workflow stop");
  return 0;
}

int Workflow::SetNodePara()
{
  RCLCPP_INFO(rclcpp::get_logger("example"), "Set node para.");
  if (!dnn_node_para_ptr_)
  {
    return -1;
  }
  dnn_node_para_ptr_->model_file = model_file_name_;
  dnn_node_para_ptr_->model_name = model_name_;
  dnn_node_para_ptr_->model_task_type = model_task_type_;
  dnn_node_para_ptr_->task_num = 2;
  return 0;
}

int Workflow::PostProcess(
    const std::shared_ptr<DnnNodeOutput> &node_output)
{
  auto fasterRcnn_output =
      std::dynamic_pointer_cast<FasterRcnnOutput>(node_output);

  static uint64_t max_Predict_duration = 0;
  static uint64_t min_Predict_duration = INT64_MAX;
  static uint64_t total_Predict_duration = 0;

  static uint64_t start_ts = currentMicroseconds();
  static uint64_t end_ts = currentMicroseconds();

  if (fasterRcnn_output)
  {
    fasterRcnn_output->Predict_duration =
      currentMicroseconds() - fasterRcnn_output->Predict_Start_time;
    std::unique_lock<std::mutex> lk(frame_cnt_mtx);
    frame_cnt++;
    if (show_fps_log)
    {
      if (frame_cnt % statistic_cycle == 0)
      {
        if (frame_cnt != (uint64_t)statistic_cycle)
        { // ignore first stat cycle
          end_ts = currentMicroseconds();
          auto fps_data = (1000000.0 * statistic_cycle / (end_ts - start_ts));

          std::cout << YELLOW_COMMENT_START
          << "Throughput: " << fps_data << " fps"
          << YELLOW_COMMENT_END << std::endl;
        }
        start_ts = currentMicroseconds();
      }
    }
    if (show_latency_log)
    {
      max_Predict_duration =
          std::max(max_Predict_duration, fasterRcnn_output->Predict_duration);
      min_Predict_duration =
          std::min(min_Predict_duration, fasterRcnn_output->Predict_duration);
      total_Predict_duration += fasterRcnn_output->Predict_duration;

      if (frame_cnt % statistic_cycle == 0)
      {
        auto avg_data = ((total_Predict_duration - min_Predict_duration -
            max_Predict_duration) / 1000.0) / (statistic_cycle - 2);

        std::cout << RED_COMMENT_START << "Predict latency:  [avg:  "
        << avg_data
        << "ms,  max:  " << max_Predict_duration / 1000.0
        << "ms,  min:  " << min_Predict_duration / 1000.0
        << "ms]" << RED_COMMENT_END << std::endl;

        max_Predict_duration = 0;
        min_Predict_duration = INT64_MAX;
        total_Predict_duration = 0;
      }
    }
  }
  InputProducerPlugin::GetInstance()->Release();
  return 0;
}

int Workflow::Predict(
  std::vector<std::shared_ptr<DNNInput>> &inputs,
  const std::shared_ptr<std::vector<hbDNNRoi>> rois,
  std::shared_ptr<DnnNodeOutput> dnn_output)
{
  RCLCPP_INFO(rclcpp::get_logger("example"), "task_num: %d",
  dnn_node_para_ptr_->task_num);

  return Run(inputs, dnn_output, rois, is_sync_mode_ == 1 ? true : false);
}
