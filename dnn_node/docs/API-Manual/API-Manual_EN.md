English | [简体中文](./API-Manual.md)

# 1. Overview
This document describes the APIs, data types, and structures of DNN Node in TROS.

# 2. Data Classes and Structures
## 2.1 Model Task Entity Types
```cpp
// Task entity types, including the following:
// - ModelInferTask: Pyramid or DDR model task
// - ModelRoiInferTask: Resizer model task, with one and only one input
//   source being resizer (pyramid and roi), and the remaining ones
//   being pyramid or DDR
enum class ModelTaskType {
  InvalidType = 0,
  ModelInferType = 1,
  ModelRoiInferType = 2
};
```
## 2.2 Dnn Node Configuration Parameters
```cpp
struct DnnNodePara {
  // Model file name
  std::string model_file;

  // Model name. The dnn node resolves, based on model_name, the model
  // to be managed and used for inference.
  // On the X3 Pi, the way to query the model name from the model file is:
  // hrt_model_exec model_info --model_file [model_file (model file name)]
  // The output contains the model name, model inputs and outputs,
  // where [model name] is the model name.
  // If the model file contains only one model, the model name can be omitted.
  std::string model_name{""};

  // Task type of the model. The dnn node creates tasks according to
  // the model type.
  // Select ModelRoiInferType when the algorithm input contains rois
  // (Regions of Interest, e.g. detection boxes of targets);
  // otherwise select ModelInferType.
  ModelTaskType model_task_type = ModelTaskType::ModelInferType;

  // Timeout for creating a task, in ms
  int timeout_ms = 100;

  // Number of tasks to create. One model can be executed by multiple tasks.
  // If model inference is time-consuming (the inference output frame rate
  // is lower than the input frame rate), more tasks are needed for inference.
  int task_num = 2;

  // BPU cores used by inference tasks. Macro names differ by platform:
  // HB_BPU_CORE_* on libdnn platforms (X3/Ultra/X5),
  // HB_UCP_BPU_CORE_* on ucp platforms (S100).
  // - *_CORE_ANY: No BPU core is specified; inference tasks alternately
  //   use the two BPU cores for load balancing.
  // - *_CORE_0: Inference tasks use BPU 0.
  // - *_CORE_1: Inference tasks use BPU 1.
  // When bpu_core_ids is empty, no BPU core is specified (CORE_ANY mode).
  // When it is not empty, its size must equal task_num, specifying
  // the corresponding BPU core for each task.
  // For example, with task_num of 2 and bpu_core_ids of
  // {HB_BPU_CORE_0, HB_BPU_CORE_1}, the two tasks run on BPU 0 and BPU 1
  // respectively.
  std::vector<int32_t> bpu_core_ids{};
};
```

## 2.3 Dnn Node Output Data

```cpp
struct DnnNodeOutput {
  DnnNodeOutput() {}
  virtual ~DnnNodeOutput() {}

  // Message header of the output result, used to match the corresponding
  // input. Filled in preprocessing with the header of the subscribed
  // image message.
  std::shared_ptr<std_msgs::msg::Header> msg_header = nullptr;

  // List of model output tensor pointers, with the dimension equal to the
  // number of model output branches.
  // After inference completes, users can directly parse output_tensors
  // and use the parsed structured data.
  std::vector<std::shared_ptr<DNNTensor>> output_tensors;

  // Runtime statistics pointer, used to manage model inference time
  std::shared_ptr<DnnNodeRunTimeStat> rt_stat = nullptr;

  // List of model input roi pointers, valid only when the model task is
  // ModelRoiInferTask.
  // The roi data is passed only for model post-process parsing and is not
  // used for actual inference.
  std::shared_ptr<std::vector<hbDNNRoi>> rois;
};
```

Users can inherit from DnnNodeOutput to extend the output content, adding input data information associated with the inference output. For example, for a detection task that takes images as input, an image name can be added (the header information of the input message is already provided by the base class via `msg_header`, so it does not need to be extended again):

```
struct DetectionOutput : public DnnNodeOutput {
  std::string image_name = "";
};
```

# 3. DnnNode Class Interfaces

## 3.1 Init()
```cpp
int Init();
```
Runs the initialization flow: calls the user-implemented `SetNodePara()` to configure model parameters, validates `model_task_type` and `bpu_core_ids`, then loads the model and creates tasks.

- Return value
    - 0 for success, non-zero for failure.

## 3.2 Run()
```cpp
int Run(std::vector<std::shared_ptr<DNNInput>> &inputs,
        const std::shared_ptr<DnnNodeOutput> &output = nullptr,
        const std::shared_ptr<std::vector<hbDNNRoi>> rois = nullptr,
        const bool is_sync_mode = false,
        const int alloctask_timeout_ms = -1,
        const int infer_timeout_ms = 20000);
```
1. Runs inference with DNNInput data. This is the usual inference path for all models except DDR models.
2. Executes the inference flow, which only chains the pipeline; each concrete inference step is implemented by the user (in the subclass).
3. Users can inherit from DnnNodeOutput to extend the output pointer `output`, for example adding the image data, image name, timestamp, ID, etc. associated with the inference result.
4. If the output content does not need to be extended, `output` can be omitted.

- Parameters
    - [in] inputs List of input data pointers
    - [in] output Output data pointer
    - [in] rois ROI data for cropping, only valid for ModelRoiInferType models
    - [in] is_sync_mode Inference mode: true for synchronous mode, false for asynchronous mode
    - [in] alloctask_timeout_ms Timeout for requesting an inference task, in milliseconds.
                               By default it waits indefinitely until the request succeeds.
    - [in] infer_timeout_ms Inference timeout in milliseconds; defaults to a 20000 ms inference timeout.

- Return value
    - 0 for success, non-zero for failure.

## 3.3 SetNodePara()
```cpp
virtual int SetNodePara() = 0;
```
Sets the DnnNodePara-typed node para, configuring the model file name, model name, task type, etc., for use by Init().

- Return value
    - 0 for success, non-zero for failure.

## 3.4 PostProcess()
```cpp
virtual int PostProcess(const std::shared_ptr<DnnNodeOutput> &output);
```
Processes the model output data. This is a virtual (non-pure) function whose base-class default implementation is a no-op; subclasses generally need to override it: parse the model output tensors in `output->output_tensors` (the output_parser algorithms built into dnn_node can be used directly, e.g. the yolo/fasterrcnn/fcos/ssd detection series, classification, segmentation), and encapsulate the parsed results into messages published to the bus.

DnnNodeOutput contains the model output tensor data (output_tensors) and the user-defined extended output content (e.g. input data information associated with the inference output).

- Parameters
    - [in] output Output data pointer.
- Return value
    - 0 for success, non-zero for failure.

## 3.5 GetModel()
```cpp
Model* GetModel();
```
Gets the model managed and used for inference by the dnn node.

- Return value
    - Returns the loaded model pointer.

## 3.6 GetModelInputSize()
```cpp
int GetModelInputSize(int32_t input_index, int& w, int& h);
```
Gets the input size of the model.

- Parameters
    - [in] input_index Index of the input to query; a model may have multiple inputs.
    - [out] w Width of the model input.
    - [out] h Height of the model input.
- Return value
    - 0 for success, non-zero for failure.

## 3.7 Run()
```cpp
int Run(std::vector<std::shared_ptr<DNNTensor>> &inputs,
        const std::shared_ptr<DnnNodeOutput> &output = nullptr,
        const bool is_sync_mode = false,
        const int alloctask_timeout_ms = -1,
        const int infer_timeout_ms = 20000);
```
1. Runs inference with DNNTensor data. DDR models generally use this inference path.
2. Executes the inference flow, which only chains the pipeline; each concrete inference step is implemented by the user (in the subclass).
3. Users can inherit from DnnNodeOutput to extend the output pointer `output`, for example adding the image data, image name, timestamp, ID, etc. associated with the inference result.
4. If the output content does not need to be extended, `output` can be omitted.

- Parameters
    - [in] inputs List of input data pointers
    - [in] output Output data pointer
    - [in] is_sync_mode Inference mode: true for synchronous mode, false for asynchronous mode
    - [in] alloctask_timeout_ms Timeout for requesting an inference task, in milliseconds.
                               By default it waits indefinitely until the request succeeds.
    - [in] infer_timeout_ms Inference timeout in milliseconds; defaults to a 20000 ms inference timeout.

- Return value
    - 0 for success, non-zero for failure.
