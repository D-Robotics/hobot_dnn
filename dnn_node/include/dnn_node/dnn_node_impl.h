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

#ifndef DNN_NODE_IMPL_H_
#define DNN_NODE_IMPL_H_

#include <memory>
#include <unordered_map>
#include <vector>

#include "dnn_node/dnn_node_data.h"
#include "easy_dnn/model.h"
#include "util/threads/threadpool.h"

using hobot::easy_dnn::Model;

namespace hobot {
namespace dnn_node {

// 推理完成后的结果回调类型
using PostProcessCbType = std::function<int(std::shared_ptr<DnnNodeOutput> &)>;

struct DnnNodeRunTimePara;
struct ThreadPool;

enum class InputType { DNN_INPUT = 0, DNN_TENSOR };

struct DnnNodeTask {
  explicit DnnNodeTask(TaskId id) {
    task_id = id;
    alloc_tp = std::chrono::system_clock::now();
  }
  void SetBPUCoreID(int32_t bpu_core_id) { bpuCoreId = bpu_core_id; }
  TaskId task_id = -1;
  // 实际运行时使用的BPU核，必须为BPU_CORE_0或BPU_CORE_1
  // 如果用户指定的是BPU_CORE_ANY，将会转成BPU_CORE_0或BPU_CORE_1
  int32_t bpuCoreId = HB_BPU_CORE_0;

  std::chrono::high_resolution_clock::time_point alloc_tp;
};

struct DnnNodeRunTimePara {
  // 使用模型文件加载后的模型列表
  std::vector<Model *> models_load;

  // 根据model_name解析出的需要管理和推理使用的模型
  Model *model_manage = nullptr;

  // 一个DNNNode实例只支持一种ModelTask类型
  std::vector<std::shared_ptr<Task>> tasks{};

  // todo 20220228
  // Add task release strategy according to task alloc_tp to
  // avoid task abnormal leakage
  std::unordered_map<TaskId, std::shared_ptr<DnnNodeTask>> idle_tasks{};
  std::unordered_map<TaskId, std::shared_ptr<DnnNodeTask>> running_tasks{};
  std::mutex task_mtx;
  std::condition_variable task_cv;
};

// 运行时fps统计
struct DnnNodeRunTimeFpsStat {
  // 第一帧的时间
  std::shared_ptr<std::chrono::high_resolution_clock::time_point>
      last_frame_tp = nullptr;
  int frame_count = 0;
  float frame_fps = -1;
  std::mutex frame_stat_mtx;

  bool Update();
  float Get();
};

struct ThreadPool {
  hobot::CThreadPool msg_handle_;
  std::mutex msg_mutex_;
  int msg_limit_count_ = 10;
};

class DnnNodeImpl {
 public:
  explicit DnnNodeImpl(std::shared_ptr<DnnNodePara> &dnn_node_para_ptr);

  ~DnnNodeImpl();

  int ModelInit();

 public:
  // 申请模型预测任务。
  // - 参数
  //   - [in] timeout_ms 申请超时时间。
  // - 返回值
  //   - 返回申请到的task id，小于0为无效id。
  TaskId AllocTask(int timeout_ms = -1);

  // 释放模型预测任务。
  // - 参数
  //   - [in] task_id 需要释放的task id。
  int ReleaseTask(const TaskId &);

  // 根据预测任务ID获取任务task。
  // - 参数
  //   - [in] task_id 预测任务ID。
  std::shared_ptr<Task> GetTask(const TaskId &);

  int TaskInit();

  // 启动推理
  // is_sync_mode 预测模式，true为同步模式，false为异步模式。
  int Run(std::vector<std::shared_ptr<DNNInput>> &dnn_inputs,
          std::vector<std::shared_ptr<DNNTensor>> &tensor_inputs,
          InputType input_type,
          const std::shared_ptr<DnnNodeOutput> &output,
          PostProcessCbType post_process,
          const std::shared_ptr<std::vector<hbDNNRoi>> rois,
          const bool is_sync_mode,
          const int alloctask_timeout_ms,
          const int infer_timeout_ms);

  // 推理实现
  int RunImpl(std::vector<std::shared_ptr<DNNInput>> dnn_inputs,
              std::vector<std::shared_ptr<DNNTensor>> tensor_inputs,
              InputType input_type,
              std::shared_ptr<DnnNodeOutput> output,
              PostProcessCbType post_process,
              const std::shared_ptr<std::vector<hbDNNRoi>> rois,
              const int alloctask_timeout_ms,
              const int infer_timeout_ms);

  // 配置预测任务的输入数据
  // - 参数
  //   - [in] inputs 输入数据智能指针列表。
  //   - [in] task_id 预测任务ID。
  //   - [in] rois 抠图roi数据，只对抠图检测模型有效。
  int PreProcess(std::vector<std::shared_ptr<DNNInput>> &dnn_inputs,
                 std::vector<std::shared_ptr<DNNTensor>> &tensor_inputs,
                 InputType input_type,
                 const TaskId &task_id,
                 const std::shared_ptr<std::vector<hbDNNRoi>> rois = nullptr);

  // 输入预处理
  int RunProcessInput( const TaskId &task_id,
                   InputType input_type);

  // 使用通过SetInputs输入给模型的数据进行推理
  // - 参数
  //   - [in/out] node_output 推理任务输出智能。
  //   - [in] task 推理任务。
  //   - [in] timeout_ms 推理推理超时时间。
  // outputs为模型输出，timeout_ms为推理超时时间
  int RunInferTask(std::shared_ptr<DnnNodeOutput> node_output,
               const std::shared_ptr<Task> &task,
               const int timeout_ms = 1000);

  // 获取dnn node管理和推理使用的模型。
  Model *GetModel();

  // 获取模型的输入size
  // - 参数
  //   - [in] input_index 获取的输入索引，一个模型可能有多个输入。
  //   - [out] w 模型输入的宽度。
  //   - [out] h 模型输入的高度。
  int GetModelInputSize(int32_t input_index, int &w, int &h);

  int LoadModels(std::vector<Model *> &models,
              const std::string &model_file);

 private:
  std::shared_ptr<DnnNodePara> dnn_node_para_ptr_ = nullptr;
  std::shared_ptr<DnnNodeRunTimePara> dnn_rt_para_ = nullptr;
  std::shared_ptr<ThreadPool> thread_pool_ = nullptr;

  hbPackedDNNHandle_t packed_dnn_handle_;

  // 输入的统计
  // 例如对于订阅图片进行推理的场景，此处统计的输入帧率等于订阅到图片的帧率
  DnnNodeRunTimeFpsStat input_stat_;
  // 输出的统计，只统计推理成功（推理+解析模型输出）的帧率
  DnnNodeRunTimeFpsStat output_stat_;

  // 配置task参数使能开关（使用SetCtrlParam接口配置，如推理使用的BPU核）
  // true: 允许用户配置，false: 禁止用户配置
  // 如果用户配置task参数后导致推理失败，尝试使用默认参数推理。
  // 如果尝试成功，将自动设置en_set_task_para_为true，之后的推理任务不再配置task参数。
  // 例如dnn node推理默认使用负载均衡模式（交替指定两个BPU核），
  // 而对于多核模型，推理任务会同时使用两个BPU核，因此如果指定了BPU核将会推理失败。
  bool en_set_task_para_ = true;

  std::mutex load_lock_;
  int32_t core_id_ = HB_BPU_CORE_0;
};

}  // namespace dnn_node
}  // namespace hobot
#endif  // DNN_NODE_IMPL_H_
