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

#include "dnn_node/dnn_node_impl.h"

#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"

namespace hobot {
namespace dnn_node {

bool DnnNodeRunTimeFpsStat::Update() {
  std::unique_lock<std::mutex> lk(frame_stat_mtx);
  if (!last_frame_tp) {
    last_frame_tp =
        std::make_shared<std::chrono::high_resolution_clock::time_point>();
    *last_frame_tp = std::chrono::system_clock::now();
  }
  auto tp_now = std::chrono::system_clock::now();
  frame_count++;
  auto interval = std::chrono::duration_cast<std::chrono::milliseconds>(
                      tp_now - *last_frame_tp)
                      .count();
  if (interval >= 1000) {
    frame_fps = static_cast<float>(frame_count) /
                (static_cast<float>(interval) / 1000.0);
    frame_count = 0;
    *last_frame_tp = std::chrono::system_clock::now();
    return true;
  }
  return false;
}

float DnnNodeRunTimeFpsStat::Get() {
  std::unique_lock<std::mutex> lk(frame_stat_mtx);
  return frame_fps;
}

DnnNodeImpl::DnnNodeImpl(std::shared_ptr<DnnNodePara> &dnn_node_para_ptr) {
  dnn_node_para_ptr_ = dnn_node_para_ptr;
  dnn_rt_para_ = std::make_shared<DnnNodeRunTimePara>();
  thread_pool_ = std::make_shared<ThreadPool>();
}

DnnNodeImpl::~DnnNodeImpl() {
  if (!dnn_rt_para_) {
    std::unique_lock<std::mutex> lk(load_lock_);
    dnn_rt_para_->models_load.clear();
    hbDNNRelease(packed_dnn_handle_);
  }
}

int DnnNodeImpl::ModelInit() {
  RCLCPP_INFO(rclcpp::get_logger("dnn"), "Model init.");
  if (!dnn_node_para_ptr_ || !dnn_rt_para_) {
    RCLCPP_ERROR(rclcpp::get_logger("dnn"), "Invalid node para!");
    return -1;
  }

  // 1. 加载模型hbm文件，一个hbm中可能包含多个模型
  int ret = 0;
  ret = LoadModels(dnn_rt_para_->models_load,
                   dnn_node_para_ptr_->model_file);
  if (0 != ret) {
    RCLCPP_ERROR(rclcpp::get_logger("dnn"),
                 "Load model: %s fail, ret: %d",
                 dnn_node_para_ptr_->model_file.c_str(),
                 ret);
    if (ret == HB_DNN_CAN_NOT_OPEN_FILE) {
      RCLCPP_ERROR(rclcpp::get_logger("dnn"),
                  "Model file %s is not exist, please install models with apt install!",
                  dnn_node_para_ptr_->model_file.c_str());
    }
    return ret;
  }

  // 2. 根据模型名，加载实际需要管理的模型
  const auto &model_name = dnn_node_para_ptr_->model_name;
  if (model_name.empty()) {
    // 2.1 用户没有指定模型名
    if (static_cast<int>(dnn_rt_para_->models_load.size()) == 1) {
      // 2.1.1 模型文件中只有一个模型，直接使用
      dnn_rt_para_->model_manage = dnn_rt_para_->models_load.at(0);
    } else {
      // 2.1.2 模型文件中有多个模型，用户必须指定需要加载的模型
      RCLCPP_ERROR(rclcpp::get_logger("dnn"),
                   "Model file: %s has %zu models, please set model_name para "
                   "in DnnNodePara with SetNodePara API",
                   dnn_node_para_ptr_->model_file.c_str(),
                   dnn_rt_para_->models_load.size());
      return -1;
    }
  } else {
    // 2.2 用户指定了模型名
    for (auto model : dnn_rt_para_->models_load) {
      if (model->GetName() == model_name) {
        dnn_rt_para_->model_manage = model;
        break;
      }
    }
  }
  if (!dnn_rt_para_->model_manage) {
    RCLCPP_ERROR(rclcpp::get_logger("dnn"),
                 "Find model: %s fail! Check model name on X3PI with cmd: "
                 "hrt_model_exec model_info --model_file %s",
                 model_name.c_str(),
                 dnn_node_para_ptr_->model_file.c_str());
    return -1;
  }

  // 3. 查询模型的输入信息
  for (int idx = 0; idx < dnn_rt_para_->model_manage->GetInputCount(); idx++) {
    hbDNNTensorProperties properties;
    dnn_rt_para_->model_manage->GetInputTensorProperties(properties, idx);
    int in_w = properties.validShape.dimensionSize[3];
    int in_h = properties.validShape.dimensionSize[2];
    RCLCPP_INFO(rclcpp::get_logger("dnn"),
                "The model input %d width is %d and height is %d",
                idx,
                in_w,
                in_h);
  }

  // 4. 打印模型输入输出信息
  std::stringstream ss;
  GetModel()->PrintModelInfo(ss);
  RCLCPP_INFO(
    rclcpp::get_logger("dnn"), "%s", ss.str().c_str());
  return 0;
}

int DnnNodeImpl::LoadModels(std::vector<Model *> &models,
                               const std::string &model_file) {
  const char *file{model_file.c_str()};
  int ret = 0;
  //第一步加载模型
  ret = hbDNNInitializeFromFiles(&packed_dnn_handle_, static_cast<const char **>(&file), 1);
  if (ret != 0) {
    return ret;
  }

  // 第二步获取模型名称
  const char **model_names;
  int32_t model_count = 0;
  ret = hbDNNGetModelNameList(&model_names, &model_count, packed_dnn_handle_);
  if (ret != 0) {
    return ret;
  }

  // 第三步获取dnn_handle
  {
    for (int32_t i{0}; i < model_count; i++) {
      hbDNNHandle_t dnn_handle{nullptr};
      ret = hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle_, model_names[i]);
      if (ret != 0) {
        return ret;
      }
      Model* model = new Model(dnn_handle, model_names[i]);
      models.push_back(model);
    }
  }

  return ret;
}

int DnnNodeImpl::TaskInit() {
  RCLCPP_INFO(rclcpp::get_logger("dnn"), "Task init.");
  if (!dnn_node_para_ptr_ || !dnn_rt_para_ ||
      dnn_node_para_ptr_->task_num < 1) {
    RCLCPP_ERROR(rclcpp::get_logger("dnn"), "Invalid node para!");
    return -1;
  }

  // bpu_core_ids必须为空或者size等于task_num
  if (!dnn_node_para_ptr_->bpu_core_ids.empty() &&
      static_cast<int>(dnn_node_para_ptr_->bpu_core_ids.size()) !=
          dnn_node_para_ptr_->task_num) {
    RCLCPP_ERROR(rclcpp::get_logger("dnn"),
                 "DnnNodePara of bpu_core_ids size %zu should be zero or equal "
                 "with task_num %d",
                 dnn_node_para_ptr_->bpu_core_ids.size(),
                 dnn_node_para_ptr_->task_num);
    return -1;
  }

  // 1. 为每个模型创建task
  // 此处只创建空的task，AllocTask才真正创建task，ReleaseTask时释放task
  // 原因是目前一个task不支持多次预测，即每次预测都要申请一个新的task
  // todo 20220305 需要easy dnn支持task复用
  dnn_rt_para_->tasks.resize(dnn_node_para_ptr_->task_num);

  // 2. 创建idle running task
  {
    int task_num = dnn_rt_para_->tasks.size();
    auto bpu_core_id = HB_BPU_CORE_0;
    std::unique_lock<std::mutex> lg(dnn_rt_para_->task_mtx);
    if (static_cast<int>(dnn_node_para_ptr_->bpu_core_ids.size()) == task_num) {
      // 用户指定了BPU核
      for (int idx = 0; idx < task_num; idx++) {
        auto node_task = std::make_shared<DnnNodeTask>(idx);
        if (dnn_node_para_ptr_->bpu_core_ids.at(idx) !=
            HB_BPU_CORE_ANY) {
          // 指定的是BPU 0或1
          node_task->SetBPUCoreID(dnn_node_para_ptr_->bpu_core_ids.at(idx));
          dnn_rt_para_->idle_tasks[node_task->task_id] = node_task;
        } else {
          // 指定的是BPU_CORE_ANY，选择BPU 0或1
          node_task->SetBPUCoreID(bpu_core_id);
          dnn_rt_para_->idle_tasks[node_task->task_id] = node_task;
          if (idx == task_num - 1) {
            break;
          }
          // 更新bpu_core_id，保证每个task交替使用不同的BPU核
          if (HB_BPU_CORE_0 == bpu_core_id) {
            bpu_core_id = HB_BPU_CORE_1;
          } else if (HB_BPU_CORE_1 == bpu_core_id) {
            bpu_core_id = HB_BPU_CORE_0;
          } else {
            bpu_core_id = HB_BPU_CORE_0;
          }
        }
      }
    } else {
      // 没有指定BPU核，为每个task指定初始BPU核
      for (int idx = 0; idx < task_num; idx++) {
        auto node_task = std::make_shared<DnnNodeTask>(idx);
        node_task->SetBPUCoreID(bpu_core_id);
        dnn_rt_para_->idle_tasks[node_task->task_id] = node_task;
        if (idx == task_num - 1) {
          break;
        }
        // 更新bpu_core_id，保证每个task交替使用不同的BPU核
        if (HB_BPU_CORE_0 == bpu_core_id) {
          bpu_core_id = HB_BPU_CORE_1;
        } else if (HB_BPU_CORE_1 == bpu_core_id) {
          bpu_core_id = HB_BPU_CORE_0;
        } else {
          bpu_core_id = HB_BPU_CORE_0;
        }
      }
    }
  }

  thread_pool_->msg_handle_.CreatThread(dnn_node_para_ptr_->task_num);
  RCLCPP_INFO(rclcpp::get_logger("dnn"),
              "Set task_num [%d]",
              dnn_node_para_ptr_->task_num);

  return 0;
}

int DnnNodeImpl::PreProcess(
    std::vector<std::shared_ptr<DNNInput>> &inputs,
    std::vector<std::shared_ptr<DNNTensor>> &tensor_inputs,
    InputType input_type,
    const TaskId &task_id,
    const std::shared_ptr<std::vector<hbDNNRoi>> rois) {
  if (task_id < 0 || !dnn_node_para_ptr_) {
    return -1;
  }
  uint32_t ret = 0;
  if (ModelTaskType::ModelRoiInferType == dnn_node_para_ptr_->model_task_type) {
    if (!rois) {
      RCLCPP_ERROR(rclcpp::get_logger("dnn"),
                   "Invalid input rois for roi infer task");
      return -1;
    }

    std::shared_ptr<ModelRoiInferTask> infer_task =
        std::dynamic_pointer_cast<ModelRoiInferTask>(GetTask(task_id));
    if (!infer_task) {
      RCLCPP_ERROR(rclcpp::get_logger("dnn"), "Invalid infer task");
      return -1;
    }

    // set roi
    ret = infer_task->SetInputRois(*rois);
    if (ret != 0) {
      RCLCPP_ERROR(rclcpp::get_logger("dnn"), "Failed to set roi inputs");
      return ret;
    }
    if (input_type == InputType::DNN_INPUT) {
      ret = infer_task->SetInputs(inputs);
    } else if (input_type == InputType::DNN_TENSOR) {
      ret = infer_task->SetInputTensors(tensor_inputs);
    } else {
      RCLCPP_ERROR(rclcpp::get_logger("dnn"),
                   "Unsupport input_type: %d",
                   static_cast<int>(input_type));
      return -1;
    }
    if (ret != 0) {
      RCLCPP_ERROR(rclcpp::get_logger("dnn"), "Failed to set inputs");
      return ret;
    }
  } else if (ModelTaskType::ModelInferType ==
             dnn_node_para_ptr_->model_task_type) {

    std::shared_ptr<ModelInferTask> infer_task =
        std::dynamic_pointer_cast<ModelInferTask>(GetTask(task_id));
        
    if (!infer_task) {
      RCLCPP_ERROR(rclcpp::get_logger("dnn"), "Invalid infer task");
      return -1;
    }
    if (input_type == InputType::DNN_INPUT) {
      ret = infer_task->SetInputs(inputs);
    } else if (input_type == InputType::DNN_TENSOR) {
      ret = infer_task->SetInputTensors(tensor_inputs);
    } else {
      RCLCPP_ERROR(rclcpp::get_logger("dnn"),
                   "Unsupport input_type: %d",
                   static_cast<int>(input_type));
      return -1;
    }
    if (ret != 0) {
      RCLCPP_ERROR(rclcpp::get_logger("dnn"), "Failed to set inputs");
      return ret;
    }
  }
  return ret;
}

int DnnNodeImpl::RunProcessInput(const TaskId &task_id,
                   InputType input_type) {
  auto task = std::dynamic_pointer_cast<Task>(GetTask(task_id));

  int ret = 0;
  if (input_type == InputType::DNN_INPUT) {
    ret = task->ProcessInput();
    if (ret != 0) {
      RCLCPP_ERROR(
          rclcpp::get_logger("dnn"), "Failed to process input, ret[%d]", ret);
      return ret;
    }
  }
  return 0;
}

int DnnNodeImpl::RunInferTask(std::shared_ptr<DnnNodeOutput> node_output,
                          const std::shared_ptr<Task> &node_task,
                          const int timeout_ms) {
  if (!dnn_node_para_ptr_ || !node_output || !node_task) {
    RCLCPP_ERROR(rclcpp::get_logger("dnn"), "Invalid node task\n");
    return -1;
  }

  auto tp_now = std::chrono::system_clock::now();
  struct timespec timespec_now = {0, 0};
  clock_gettime(CLOCK_REALTIME, &timespec_now);

  auto &task = node_task;
  int ret = 0;

  ret = task->RunInfer();
  if (ret != 0) {
    RCLCPP_ERROR(
        rclcpp::get_logger("dnn"), "Failed to run infer task, ret[%d]", ret);

    if (HB_DNN_INVALID_ARGUMENT == ret) {
      // task参数错误导致推理失败，尝试使用默认参数
      RCLCPP_WARN(rclcpp::get_logger("dnn"),
                  "Try to reset dnn infer ctrl param");
      hbDNNInferCtrlParam ctrl_param;
      HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&ctrl_param);
      task->SetCtrlParam(ctrl_param);
      ret = task->RunInfer();
      if (ret == 0) {
        // 使用默认参数推理成功，设置禁止配置task参数
        RCLCPP_WARN(rclcpp::get_logger("dnn"),
                    "Run infer success after reset dnn infer ctrl param. Task "
                    "para set will be enable!");
        en_set_task_para_ = false;
      }
    }
    if (ret != 0) {
      return ret;
    }
  }

  ret = task->WaitInferDone(timeout_ms);
  if (ret != 0) {
    RCLCPP_ERROR(
        rclcpp::get_logger("dnn"), "Failed to wait infer done, ret[%d]", ret);
    return ret;
  }

  if (node_output->rt_stat) {
    node_output->rt_stat->infer_time_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now() - tp_now)
            .count();
    node_output->rt_stat->infer_timespec_start = timespec_now;
    clock_gettime(CLOCK_REALTIME, &timespec_now);
    node_output->rt_stat->infer_timespec_end = timespec_now;

    tp_now = std::chrono::system_clock::now();
    clock_gettime(CLOCK_REALTIME, &timespec_now);
    node_output->rt_stat->parse_timespec_start = timespec_now;
  }

  if (ModelTaskType::ModelInferType == dnn_node_para_ptr_->model_task_type) {
    auto model_task = std::dynamic_pointer_cast<ModelInferTask>(task);

    // 获取解析前的DNNTensor
    model_task->GetOutputTensors(node_output->output_tensors);
  
  } 
  else if (ModelTaskType::ModelRoiInferType ==
             dnn_node_para_ptr_->model_task_type) {
    auto model_task = std::dynamic_pointer_cast<ModelRoiInferTask>(task);
    
    // 解析DNNTensor，内部会为算法的每个branch输出调用自定义的Parse接口进行解析
    model_task->GetOutputTensors(node_output->output_tensors);
  }

  if (node_output->rt_stat) {
    node_output->rt_stat->parse_time_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now() - tp_now)
            .count();
    clock_gettime(CLOCK_REALTIME, &timespec_now);
    node_output->rt_stat->parse_timespec_end = timespec_now;
  }

  if (ret != 0) {
    RCLCPP_ERROR(
        rclcpp::get_logger("dnn"), "Failed to get outputs, ret[%d]", ret);
  }

  return ret;
}

TaskId DnnNodeImpl::AllocTask(int timeout_ms) {
  RCLCPP_DEBUG(rclcpp::get_logger("dnn"), "Alloc task");
  TaskId task_id = -1;
  int32_t bpu_core_id = HB_BPU_CORE_0;
  if (!dnn_rt_para_) {
    return task_id;
  }

  // 真正创建task
  std::shared_ptr<Task> task = nullptr;
  // 根据模型类型选择接口创建task,为task添加model
  if (ModelTaskType::ModelInferType == dnn_node_para_ptr_->model_task_type) {

    task = std::make_shared<hobot::easy_dnn::ModelInferTask>();    

    if (!task) {
      RCLCPP_ERROR(rclcpp::get_logger("dnn"), "GetModelInferTask fail");
      return task_id;
    }
    std::dynamic_pointer_cast<ModelInferTask>(task)->SetModel(
        dnn_rt_para_->model_manage);

  }
  else if (ModelTaskType::ModelRoiInferType ==
             dnn_node_para_ptr_->model_task_type) {

    task = std::make_shared<hobot::easy_dnn::ModelRoiInferTask>();
    if (!task) {
      RCLCPP_ERROR(rclcpp::get_logger("dnn"), "GetModelRoiInferTask fail");
      return task_id;
    }
    std::dynamic_pointer_cast<ModelRoiInferTask>(task)->SetModel(
        dnn_rt_para_->model_manage);
  }
  else {
    RCLCPP_ERROR(rclcpp::get_logger("dnn"),
                 "Invalid model task type [%d]",
                 static_cast<int>(dnn_node_para_ptr_->model_task_type));
    return task_id;
  }

  auto alloc_task = [this, &task_id, &bpu_core_id]() {
    auto idle_task = dnn_rt_para_->idle_tasks.begin();
    idle_task->second->alloc_tp = std::chrono::system_clock::now();
    task_id = idle_task->first;
    bpu_core_id = idle_task->second->bpuCoreId;
    dnn_rt_para_->running_tasks[task_id] = idle_task->second;
    dnn_rt_para_->idle_tasks.erase(task_id);
  };

  std::unique_lock<std::mutex> lg(dnn_rt_para_->task_mtx);
  if (!dnn_rt_para_->idle_tasks.empty()) {
    alloc_task();
  } else {
    // wait for idle task
    if (timeout_ms > 0) {
      dnn_rt_para_->task_cv.wait_for(
          lg, std::chrono::milliseconds(timeout_ms), [&]() {
            return !dnn_rt_para_->idle_tasks.empty() || !rclcpp::ok();
          });
    } else {
      dnn_rt_para_->task_cv.wait(lg, [&]() {
        return !dnn_rt_para_->idle_tasks.empty() || !rclcpp::ok();
      });
    }
    if (!rclcpp::ok()) {
      return task_id;
    }
    if (!dnn_rt_para_->idle_tasks.empty()) {
      alloc_task();
    }
  }

  RCLCPP_DEBUG(rclcpp::get_logger("dnn"), "Alloc task id: %d", task_id);
  if (task_id < 0 || task_id >= static_cast<int>(dnn_rt_para_->tasks.size())) {
    RCLCPP_ERROR(rclcpp::get_logger("dnn"), "Invalid task id: %d", task_id);
    return -1;
  }

#ifdef PLATFORM_X5
  en_set_task_para_ = false;
#endif

  if (en_set_task_para_) {
    // 允许配置task参数
    hbDNNInferCtrlParam ctrl_param;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&ctrl_param);
    ctrl_param.bpuCoreId = bpu_core_id;
    RCLCPP_INFO(rclcpp::get_logger("dnn"),
                "task id: %d set bpu core: %d",
                task_id,
                ctrl_param.bpuCoreId);
    task->SetCtrlParam(ctrl_param);
  }

  dnn_rt_para_->tasks[task_id] = std::move(task);

  return task_id;
}

int DnnNodeImpl::ReleaseTask(const TaskId &task_id) {
  RCLCPP_DEBUG(rclcpp::get_logger("dnn"), "Release task id: %d", task_id);

  if (!dnn_rt_para_ || task_id < 0) {
    RCLCPP_ERROR(rclcpp::get_logger("dnn"), "Invalid task_id: %d", task_id);
    return -1;
  }

  auto node_task = std::make_shared<DnnNodeTask>(task_id);
  std::unique_lock<std::mutex> lg(dnn_rt_para_->task_mtx);
  if (dnn_rt_para_->running_tasks.find(task_id) ==
      dnn_rt_para_->running_tasks.end()) {
    RCLCPP_ERROR(
        rclcpp::get_logger("dnn"), "Task id: %d is not running", task_id);
    return -1;
  }

  // 上一次推理任务使用的BPU核
  auto last_bpu_core_id = dnn_rt_para_->running_tasks[task_id]->bpuCoreId;
  // 本次推理任务使用的BPU核
  auto present_bpu_core_id = last_bpu_core_id;
  if (static_cast<int>(dnn_node_para_ptr_->bpu_core_ids.size()) ==
          dnn_node_para_ptr_->task_num &&
      task_id < dnn_node_para_ptr_->task_num &&
      dnn_node_para_ptr_->bpu_core_ids.at(task_id) !=
          HB_BPU_CORE_ANY) {
    // 用户指定的是BPU 0或1，本次推理直接使用上一次推理任务使用的BPU核
    present_bpu_core_id = last_bpu_core_id;
  } else {
    // 交替使用BPU核
    if (HB_BPU_CORE_0 == last_bpu_core_id) {
      present_bpu_core_id = HB_BPU_CORE_1;
    } else if (HB_BPU_CORE_1 == last_bpu_core_id) {
      present_bpu_core_id = HB_BPU_CORE_0;
    } else {
      present_bpu_core_id = HB_BPU_CORE_0;
    }
  }

  // 设置本次推理任务使用的BPU核
  node_task->SetBPUCoreID(present_bpu_core_id);

  dnn_rt_para_->idle_tasks[node_task->task_id] = node_task;
  dnn_rt_para_->running_tasks.erase(task_id);
  dnn_rt_para_->tasks[task_id] = nullptr;
  dnn_rt_para_->task_cv.notify_one();
  lg.unlock();
  RCLCPP_DEBUG(rclcpp::get_logger("dnn"),
               "idle_tasks size: %zu, running_tasks size: %zu",
               dnn_rt_para_->idle_tasks.size(),
               dnn_rt_para_->running_tasks.size());
  return 0;
}

std::shared_ptr<Task> DnnNodeImpl::GetTask(const TaskId &task_id) {
  std::shared_ptr<Task> task = nullptr;
  if (!dnn_rt_para_ || task_id < 0 ||
      task_id >= static_cast<int>(dnn_rt_para_->tasks.size())) {
    RCLCPP_ERROR(rclcpp::get_logger("dnn"), "Invalid task_id: %d", task_id);
    return task;
  }
  task = dnn_rt_para_->tasks.at(task_id);

  return task;
}

Model *DnnNodeImpl::GetModel() {
  if (dnn_rt_para_) {
    return dnn_rt_para_->model_manage;
  }
  return nullptr;
}

int DnnNodeImpl::GetModelInputSize(int32_t input_index, int &w, int &h) {
  if (!dnn_rt_para_ || !dnn_rt_para_->model_manage) {
    RCLCPP_ERROR(rclcpp::get_logger("dnn"), "Invalid input model");
    return -1;
  }
  if (input_index >= dnn_rt_para_->model_manage->GetInputCount()) {
    RCLCPP_ERROR(
        rclcpp::get_logger("dnn"), "Invalid input index: %d", input_index);
    return -1;
  }

  hbDNNTensorProperties properties;
  dnn_rt_para_->model_manage->GetInputTensorProperties(properties, input_index);
  if (properties.tensorLayout == HB_DNN_LAYOUT_NHWC) {
    w = properties.validShape.dimensionSize[2];
    h = properties.validShape.dimensionSize[1];
  } else if (properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
    w = properties.validShape.dimensionSize[3];
    h = properties.validShape.dimensionSize[2];
  }
  return 0;
}

int DnnNodeImpl::Run(
    std::vector<std::shared_ptr<DNNInput>> &inputs,
    std::vector<std::shared_ptr<DNNTensor>> &tensor_inputs,
    InputType input_type,
    const std::shared_ptr<DnnNodeOutput> &output,
    PostProcessCbType post_process,
    const std::shared_ptr<std::vector<hbDNNRoi>> rois,
    const bool is_sync_mode,
    const int alloctask_timeout_ms,
    const int infer_timeout_ms) {
  // 统计输入fps
  input_stat_.Update();
  if (is_sync_mode) {
    return RunImpl(inputs,
                   tensor_inputs,
                   input_type,
                   output,
                   post_process,
                   rois,
                   alloctask_timeout_ms,
                   infer_timeout_ms);
  } else {
    std::lock_guard<std::mutex> lock(thread_pool_->msg_mutex_);
    if (thread_pool_->msg_handle_.GetTaskNum() >=
        thread_pool_->msg_limit_count_) {
      RCLCPP_INFO(rclcpp::get_logger("dnn"),
                  "Task Size: %d exceeds limit: %d. Prediction "
                  "time(rt_stat.infer_time_ms in DnnNodeOutput) is too long "
                  "for this model!",
                  thread_pool_->msg_handle_.GetTaskNum(),
                  thread_pool_->msg_limit_count_);
      // todo [20220622] 返回错误码告知用户推理失败原因
      return -1;
    }

    auto infer_task = [this,
                       inputs,
                       tensor_inputs,
                       input_type,
                       output,
                       post_process,
                       rois,
                       alloctask_timeout_ms,
                       infer_timeout_ms]() {
      RunImpl(inputs,
              tensor_inputs,
              input_type,
              output,
              post_process,
              rois,
              alloctask_timeout_ms,
              infer_timeout_ms);
    };

    thread_pool_->msg_handle_.PostTask(infer_task);
  }

  return 0;
}

int DnnNodeImpl::RunImpl(
    std::vector<std::shared_ptr<DNNInput>> inputs,
    std::vector<std::shared_ptr<DNNTensor>> tensor_inputs,
    InputType input_type,
    const std::shared_ptr<DnnNodeOutput> output,
    PostProcessCbType post_process,
    const std::shared_ptr<std::vector<hbDNNRoi>> rois,
    const int alloctask_timeout_ms,
    const int infer_timeout_ms) {
  
  // 检查参数是否正确
  if (!dnn_rt_para_) {
    RCLCPP_ERROR(rclcpp::get_logger("dnn"), "Invalid Para In Run, ret[%d]", HB_DNN_INVALID_ARGUMENT);
    return HB_DNN_INVALID_ARGUMENT;
  }

  // 1. dnn_output用于存储模型推理输出
  std::shared_ptr<DnnNodeOutput> dnn_output = nullptr;
  if (output) {
    // 使用传入的DnnNodeOutput
    dnn_output = output;
  }
  if (!dnn_output) {
    // 没有传入，创建DnnNodeOutput
    dnn_output = std::make_shared<DnnNodeOutput>();
  }
  if (!dnn_output->rt_stat) {
    dnn_output->rt_stat = std::make_shared<DnnNodeRunTimeStat>();
  }
  // 统计输入fps
  dnn_output->rt_stat->input_fps = input_stat_.Get();
  dnn_output->rois = rois;

  // 对于roi
  // infer，如果当前帧中无roi，不需要推理，更新统计信息后直接执行用户定义的后处理
  if (dnn_node_para_ptr_ &&
      ModelTaskType::ModelRoiInferType == dnn_node_para_ptr_->model_task_type &&
      (!rois || rois->empty())) {
    // 统计输出fps
    dnn_output->rt_stat->fps_updated = output_stat_.Update();
    dnn_output->rt_stat->output_fps = output_stat_.Get();
    if (post_process) {
      post_process(dnn_output);
    }
    return 0;
  }

  // 需要推理

  // 1 申请推理task
  auto task_id = AllocTask(alloctask_timeout_ms);
  if (task_id < 0) {
    return -1;
  }

  // 检查任务是否正常
  auto infer_task = std::dynamic_pointer_cast<Task>(GetTask(task_id));
  if (!infer_task) {
    RCLCPP_ERROR(rclcpp::get_logger("dnn"), "Invalid infer task");
    return -1;
  }

  // 2 将准备好的模型输入数据inputs通过前处理接口输入给模型
  // 并通过推理任务的task_id指定推理任务
  if (PreProcess(inputs, tensor_inputs, input_type, task_id, rois) != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("dnn"), "Run PreProcess failed!");
    return -1;
  }

  // 3 进行预处理
  int ret = 0;
  ret = RunProcessInput(task_id, input_type);
  if (ret != 0) {
    return ret;
  }

  // 4 执行模型推理
  ret = RunInferTask(dnn_output, GetTask(task_id), infer_timeout_ms);
  if (ret != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("dnn"), "Run infer fail\n");
  } else {
    // 统计输出fps
    dnn_output->rt_stat->fps_updated = output_stat_.Update();
    dnn_output->rt_stat->output_fps = output_stat_.Get();
  }

  // 5 推理任务资源释放
  ReleaseTask(task_id);
  
  // 6 执行模型后处理
  // 即使推理失败，也要将对应的（空）结果输出，保证每个推理输入都有输出。
  if (post_process) {
    post_process(dnn_output);
  }

  return 0;
}

}  // namespace dnn_node
}  // namespace hobot
