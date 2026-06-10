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

#include "dnn_node/dnn_node_data.h"

namespace hobot {
namespace dnn_node {
namespace dnn_node_sample {
// 定义算法输出数据类型
struct YoloV5Result {
  // 目标类别ID
  int id;
  // 目标检测框
  float xmin;
  float ymin;
  float xmax;
  float ymax;
  // 检测结果的置信度
  float score;
  // 目标类别
  std::string class_name;

  YoloV5Result(int id_,
               float xmin_,
               float ymin_,
               float xmax_,
               float ymax_,
               float score_,
               std::string class_name_)
      : id(id_),
        xmin(xmin_),
        ymin(ymin_),
        xmax(xmax_),
        ymax(ymax_),
        score(score_),
        class_name(class_name_) {}

  friend bool operator>(const YoloV5Result &lhs, const YoloV5Result &rhs) {
    return (lhs.score > rhs.score);
  }
};

// 自定义的算法输出解析方法
// - 参数
//   - [in] node_output dnn node输出，包含算法推理输出
//          解析时，如果不需要使用前处理参数，可以直接使用DnnNodeOutput中的
//          std::vector<std::shared_ptr<DNNTensor>>
//          output_tensors成员作为Parse的入口参数
//   - [in/out] results 解析后的结构化数据，YoloV5Result为自定义的算法输出数据类型
// - 返回值
//   - 0 成功
//   - -1 失败
int32_t Parse(
    const std::shared_ptr<hobot::dnn_node::DnnNodeOutput> &node_output,
    std::vector<std::shared_ptr<YoloV5Result>> &results);




}  // namespace dnn_node_sample
}  // namespace dnn_node
}  // namespace hobot