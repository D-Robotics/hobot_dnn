# 1.概述
本文档主要介绍了地平线机器人开发平台项目中DNN Node的API、数据类型、结构体等。

# 2.数据相关类及结构
## 2.1 模型任务实体类型
```cpp
// 任务实体类型，包括以下几种类型
// - ModelInferTask: Pyramid或DDR模型任务
// - ModelRoiInferTask: Resizer模型任务，有且只有一个输入源为resizer (pyramid和roi)，剩余的为pyramid或DDR
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
  std::string model_name;

  // 模型的task类型，dnn node根据模类型创建task
  // 例如多任务检测或者扣图检测，对应的ModelTask分别是ModelInferTask和ModelRoiInferTask
  ModelTaskType model_task_type;

  // 创建task超时时间，单位ms
  int timeout_ms = 100;
  
  // 创建的task数量，一个model支持由多个task执行
  int task_num = 1;
  
  // 模型输出索引和对应的解析方式
  // 如果用户子类中没有override SetOutputParser接口，
  // dnn_node基类的SetOutputParser将使用output_parsers_设置模型解析方式
  std::vector<std::pair<int32_t, std::shared_ptr<OutputParser>>>
  output_parsers_;
};
```

## 2.3 Dnn Node输出数据

```cpp
struct DnnNodeOutput {
  DnnNodeOutput() {
    outputs.clear();
  }
  virtual ~DnnNodeOutput() {}
  // 输出数据智能指针列表
  std::vector<std::shared_ptr<DNNResult>> outputs;
};
```

用户可以继承DnnNodeOutput来扩展输出内容，增加预测输出对应的输入数据信息。例如对于以图片作为输入的检测任务，可以增加图片名、header信息：

```
struct FasterRcnnOutput : public DnnNodeOutput {
  std::string image_name = "";
  std::shared_ptr<std_msgs::msg::Header> image_msg_header = nullptr;
};
```

# 3.DnnNode类接口

## 3.1 Init()

```cpp
int Init();
```
执行初始化流程，只做pipeline的串联，具体的每个初始化步骤由用户（子类中）实现。

- 返回值
    - 0成功，非0失败。

## 3.2 Run()
```cpp
int Run(std::vector<std::shared_ptr<DNNInput>> &inputs,
            const std::shared_ptr<DnnNodeOutput>& output = nullptr,
            const std::shared_ptr<std::vector<hbDNNRoi>> rois = nullptr,
            const bool is_sync_mode = true,
            const int alloctask_timeout_ms = -1,
            const int infer_timeout_ms = 20000);
```
1. 使用DNNInput类型数据进行推理，一般除了DDR模型之外，都使用此方式推理。
1. 执行推理流程，只做pipeline的串联，具体的每个推理步骤由用户（子类中）实现。
2. 用户可以继承DnnNodeOutput来扩展输出数据智能指针output，例如增加推理结果对应的图片数据、图片名、时间戳、ID等。
3. 如果不需要扩展输出内容，可以不传入output。

- 参数
    - [in] inputs 输入数据智能指针列表
    - [in] outputs 输出数据智能指针
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
设置DnnNodePara类型的node para。

- 返回值
    - 0成功，非0失败。

## 3.4 PostProcess()
```cpp
virtual int PostProcess(const std::shared_ptr<DnnNodeOutput> &output) = 0;
```
处理解析后的模型输出数据，例如将输出封装成msg发布到总线。

DnnNodeOutput中包含用户自定义的扩展输出内容（如预测输出对应的输入数据信息）和DNNResult。

DNNResult是easy dnn中定义的模型输出数据类型，用户必须根据实际使用的模型输出数据类型，继承DNNResult并定义模型输出数据类型。

例如对于检测模型，继承DNNResult的子类中需要定义检测框、置信度等输出数据类型。

- 参数
    - [out] output 输出数据智能指针。
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
    - 返回已加载的模型指针。

## 3.7 Run()
```cpp
int Run(std::vector<std::shared_ptr<DNNInput>> &inputs,
        const std::shared_ptr<DnnNodeOutput> &output = nullptr,
        const std::shared_ptr<std::vector<hbDNNRoi>> rois = nullptr,
        const bool is_sync_mode = true,
        const int alloctask_timeout_ms = -1,
        const int infer_timeout_ms = 20000);
```
1. 使用DNNInput类型数据并指定输出描述进行推理。
2. 执行推理流程，只做pipeline的串联，具体的每个推理步骤由用户（子类中）实现。
3. 用户可以继承DnnNodeOutput来扩展输出数据智能指针output，例如增加推理结果对应的图片数据、图片名、时间戳、ID等。
4. 如果不需要扩展输出内容，可以不传入output。

- 参数
    - [in] inputs 输入数据智能指针列表
    - [in] outputs 输出数据智能指针
    - [in] rois 抠图roi数据，只对ModelRoiInferType模型有效
    - [in] is_sync_mode 预测模式，true为同步模式，false为异步模式
    - [in] alloctask_timeout_ms 申请推理任务超时时间，单位毫秒
                                默认一直等待直到申请成功
    - [in] infer_timeout_ms 推理超时时间，单位毫秒，默认20000毫秒推理超时

- 返回值
    - 0成功，非0失败。


## 3.8 Run()
```cpp
int Run(std::vector<std::shared_ptr<DNNTensor>> &inputs,
        const std::shared_ptr<DnnNodeOutput> &output = nullptr,
        const bool is_sync_mode = true,
        const int alloctask_timeout_ms = -1,
        const int infer_timeout_ms = 20000);
```
1. 使用DNNTensor类型数据并指定输出描述进行推理，一般DDR模型使用此方式推理
2. 执行推理流程，只做pipeline的串联，具体的每个推理步骤由用户（子类中）实现。
3. 用户可以继承DnnNodeOutput来扩展输出数据智能指针output，例如增加推理结果对应的图片数据、图片名、时间戳、ID等。
4. 如果不需要扩展输出内容，可以不传入output。

- 参数
    - [in] inputs 输入数据智能指针列表
    - [in] outputs 输出数据智能指针
    - [in] is_sync_mode 预测模式，true为同步模式，false为异步模式
    - [in] alloctask_timeout_ms 申请推理任务超时时间，单位毫秒
                                默认一直等待直到申请成功
    - [in] infer_timeout_ms 推理超时时间，单位毫秒，默认20000毫秒推理超时

- 返回值
    - 0成功，非0失败。
