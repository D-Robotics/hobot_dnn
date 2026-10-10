[English](./API-Manual_EN.md) | 简体中文

# 1.概述
本文档主要介绍了TROS中DNN Node的API、数据类型、结构体等。

# 2.数据相关类及结构
## 2.1 模型任务实体类型
```cpp
// 任务实体类型，包括以下几种类型
// - ModelInferTask: Pyramid或DDR模型任务
// - ModelRoiInferTask: Resizer模型任务，
//   有且只有一个输入源为resizer (pyramid和roi)，剩余的为pyramid或DDR
enum class ModelTaskType {
  InvalidType = 0,
  ModelInferType = 1,
  ModelRoiInferType = 2
};
```
## 2.2 Dnn Node配置参数
```cpp
struct DnnNodePara {
  // 模型文件名称
  std::string model_file;

  // 模型名，dnn node根据model_name解析出需要管理和推理使用的模型
  // 根据模型文件名称，在X3派上查询模型名的方法为：
  // hrt_model_exec model_info --model_file [model_file（模型文件名称）]
  // 查询输出包含模型名、模型输入和输出的信息，其中[model name]为模型名
  // 如果模型文件中只包含一个模型时，可以不指定模型名
  std::string model_name{""};

  // 模型的task类型，dnn node根据模型类型创建task
  // 当算法的输入包含roi（Region of Interest，例如目标的检测框）时，
  // 选择ModelRoiInferType类型，其他情况下选择ModelInferType类型
  ModelTaskType model_task_type = ModelTaskType::ModelInferType;

  // 创建task超时时间，单位ms
  int timeout_ms = 100;

  // 创建的task数量，一个model支持由多个task执行
  // 如果算法推理耗时较长（推理输出帧率低于输入帧率），需要使用更多的task进行推理
  int task_num = 2;

  // 推理任务使用的BPU核，宏名随平台不同：
  // libdnn平台（X3/Ultra/X5）为HB_BPU_CORE_*，ucp平台（S100）为HB_UCP_BPU_CORE_*
  // - *_CORE_ANY: 不指定BPU核，推理任务交替使用两个BPU核，实现负载均衡
  // - *_CORE_0: 推理任务使用BPU 0
  // - *_CORE_1: 推理任务使用BPU 1
  // bpu_core_ids为空时不指定BPU核（即CORE_ANY模式）；
  // 不为空时size必须等于task_num，分别为每个task指定对应的BPU核
  // 例如task_num为2，bpu_core_ids为{HB_BPU_CORE_0, HB_BPU_CORE_1}时，
  // 两个task分别运行在BPU 0和BPU 1上
  std::vector<int32_t> bpu_core_ids{};
};
```

## 2.3 Dnn Node输出数据

```cpp
struct DnnNodeOutput {
  DnnNodeOutput() {}
  virtual ~DnnNodeOutput() {}

  // 输出结果的消息头，用于匹配对应的输入
  // 在前处理中使用订阅到的图片消息的header填充
  std::shared_ptr<std_msgs::msg::Header> msg_header = nullptr;

  // 模型输出的tensor数据指针列表，维度等于模型输出branch数
  // 推理完成后，用户可以直接解析output_tensors并使用解析后的结构化数据
  std::vector<std::shared_ptr<DNNTensor>> output_tensors;

  // 运行时时间统计指针，用于管理模型推理时间
  std::shared_ptr<DnnNodeRunTimeStat> rt_stat = nullptr;

  // 模型输入的Roi数据指针列表，仅在模型任务为ModelRoiInferTask时有效
  // 仅传递Roi数据用于模型后处理解析，不用于实际推理
  std::shared_ptr<std::vector<hbDNNRoi>> rois;
};
```

用户可以继承DnnNodeOutput来扩展输出内容，增加预测输出对应的输入数据信息。例如对于以图片作为输入的检测任务，可以增加图片名（输入消息的header信息基类已通过`msg_header`提供，无需重复扩展）：

```
struct DetectionOutput : public DnnNodeOutput {
  std::string image_name = "";
};
```

# 3.DnnNode类接口

## 3.1 Init()
```cpp
int Init();
```
执行初始化流程：调用用户实现的`SetNodePara()`配置模型参数，校验`model_task_type`和`bpu_core_ids`合法性，然后加载模型并创建task。

- 返回值
    - 0成功，非0失败。

## 3.2 Run()
```cpp
int Run(std::vector<std::shared_ptr<DNNInput>> &inputs,
        const std::shared_ptr<DnnNodeOutput> &output = nullptr,
        const std::shared_ptr<std::vector<hbDNNRoi>> rois = nullptr,
        const bool is_sync_mode = false,
        const int alloctask_timeout_ms = -1,
        const int infer_timeout_ms = 20000);
```
1. 使用DNNInput类型数据进行推理，一般除了DDR模型之外，都使用此方式推理。
2. 执行推理流程，只做pipeline的串联，具体的每个推理步骤由用户（子类中）实现。
3. 用户可以继承DnnNodeOutput来扩展输出数据智能指针output，例如增加推理结果对应的图片数据、图片名、时间戳、ID等。
4. 如果不需要扩展输出内容，可以不传入output。

- 参数
    - [in] inputs 输入数据智能指针列表
    - [in] output 输出数据智能指针
    - [in] rois 抠图roi数据，只对ModelRoiInferType模型有效
    - [in] is_sync_mode 预测模式，true为同步模式，false为异步模式
    - [in] alloctask_timeout_ms 申请推理任务超时时间，单位毫秒
                            默认一直等待直到申请成功
    - [in] infer_timeout_ms 推理超时时间，单位毫秒，默认20000毫秒推理超时

- 返回值
    - 0成功，非0失败。

## 3.3 SetNodePara()
```cpp
virtual int SetNodePara() = 0;
```
设置DnnNodePara类型的node para，配置模型文件名、模型名、task类型等，供Init()使用。

- 返回值
    - 0成功，非0失败。

## 3.4 PostProcess()
```cpp
virtual int PostProcess(const std::shared_ptr<DnnNodeOutput> &output);
```
处理模型输出数据。虚函数（非纯虚），基类默认实现为空操作，子类一般需要override此接口：解析`output->output_tensors`中的模型输出tensor（可直接使用dnn_node内置的output_parser解析算法，例如detection的yolo/fasterrcnn/fcos/ssd系列、classification、segmentation），将解析结果封装成msg发布到总线。

DnnNodeOutput中包含模型输出tensor数据（output_tensors）和用户自定义的扩展输出内容（如预测输出对应的输入数据信息）。

- 参数
    - [in] output 输出数据智能指针。
- 返回值
    - 0成功，非0失败。

## 3.5 GetModel()
```cpp
Model* GetModel();
```
获取dnn node管理和推理使用的模型。

- 返回值
    - 返回已加载的模型指针。

## 3.6 GetModelInputSize()
```cpp
int GetModelInputSize(int32_t input_index, int& w, int& h);
```
获取模型的输入size。

- 参数
    - [in] input_index 获取的输入索引，一个模型可能有多个输入。
    - [out] w 模型输入的宽度。
    - [out] h 模型输入的高度。
- 返回值
    - 0成功，非0失败。

## 3.7 Run()
```cpp
int Run(std::vector<std::shared_ptr<DNNTensor>> &inputs,
        const std::shared_ptr<DnnNodeOutput> &output = nullptr,
        const bool is_sync_mode = false,
        const int alloctask_timeout_ms = -1,
        const int infer_timeout_ms = 20000);
```
1. 使用DNNTensor类型数据进行推理，一般DDR模型使用此方式推理。
2. 执行推理流程，只做pipeline的串联，具体的每个推理步骤由用户（子类中）实现。
3. 用户可以继承DnnNodeOutput来扩展输出数据智能指针output，例如增加推理结果对应的图片数据、图片名、时间戳、ID等。
4. 如果不需要扩展输出内容，可以不传入output。

- 参数
    - [in] inputs 输入数据智能指针列表
    - [in] output 输出数据智能指针
    - [in] is_sync_mode 预测模式，true为同步模式，false为异步模式
    - [in] alloctask_timeout_ms 申请推理任务超时时间，单位毫秒
                            默认一直等待直到申请成功
    - [in] infer_timeout_ms 推理超时时间，单位毫秒，默认20000毫秒推理超时

- 返回值
    - 0成功，非0失败。
