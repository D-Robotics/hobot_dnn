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

#ifndef _OUTPUT_PARSER_PERCEPTION_COMMON_H_
#define _OUTPUT_PARSER_PERCEPTION_COMMON_H_

#include <algorithm>
#include <iomanip>
#include <iterator>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace hobot {
namespace dnn_node {
namespace output_parser {

typedef struct Anchor {
  float cx;
  float cy;
  float w;
  float h;
  Anchor(float cx, float cy, float w, float h) : cx(cx), cy(cy), w(w), h(h) {}

  friend std::ostream &operator<<(std::ostream &os, const Anchor &anchor) {
    os << "[" << anchor.cx << "," << anchor.cy << "," << anchor.w << ","
       << anchor.h << "]";
    return os;
  }
} Anchor;

/**
 * Bounding box definition
 */
typedef struct Bbox {
  float xmin;
  float ymin;
  float xmax;
  float ymax;

  Bbox() {}

  Bbox(float xmin, float ymin, float xmax, float ymax)
      : xmin(xmin), ymin(ymin), xmax(xmax), ymax(ymax) {}

  friend std::ostream &operator<<(std::ostream &os, const Bbox &bbox) {
    os << "[" << std::fixed << std::setprecision(6) << bbox.xmin << ","
       << bbox.ymin << "," << bbox.xmax << "," << bbox.ymax << "]";
    return os;
  }

  ~Bbox() {}
} Bbox;

typedef struct Detection {
  int id;
  float score;
  Bbox bbox;
  const char *class_name = nullptr;
  Detection() {}

  Detection(int id, float score, Bbox bbox)
      : id(id), score(score), bbox(bbox) {}

  Detection(int id, float score, Bbox bbox, const char *class_name)
      : id(id), score(score), bbox(bbox), class_name(class_name) {}

  friend bool operator>(const Detection &lhs, const Detection &rhs) {
    return (lhs.score > rhs.score);
  }

  friend std::ostream &operator<<(std::ostream &os, const Detection &det) {
    os << "{"
       << R"("bbox")"
       << ":" << det.bbox << ","
       << R"("prob")"
       << ":" << std::fixed << std::setprecision(6) << det.score << ","
       << R"("label")"
       << ":" << det.id << ","
       << R"("class_name")"
       << ":\"" << det.class_name << "\"}";
    return os;
  }

  ~Detection() {}
} Detection;

typedef struct Classification {
  int id;
  float score;
  const char *class_name;

  Classification() : class_name(0) {}

  Classification(int id, float score, const char *class_name)
      : id(id), score(score), class_name(class_name) {}

  friend bool operator>(const Classification &lhs, const Classification &rhs) {
    return (lhs.score > rhs.score);
  }

  friend std::ostream &operator<<(std::ostream &os, const Classification &cls) {
    os << "{"
       << R"("prob")"
       << ":" << std::fixed << std::setprecision(5) << cls.score << ","
       << R"("label")"
       << ":" << cls.id << ","
       << R"("class_name")"
       << ":"
       << "\"" << cls.class_name << "\""
       << "}";
    return os;
  }

  ~Classification() {}
} Classification;

struct Parsing {
  std::vector<int8_t> seg; // 生成本地渲染图片用到
  std::vector<float> data; // 生成web展示时用到
  int32_t num_classes = 0;
  int32_t width = 0; // 图像送入模型前经过resize的w
  int32_t valid_h = 0; // seg的h
  int32_t valid_w = 0; // seg的w
  int32_t height = 0; // 图像送入模型前经过resize的h
  int32_t channel = 0;
};

typedef struct YOLOSegmentation : public Detection {
  std::vector<float> mask;
  YOLOSegmentation() {}

  YOLOSegmentation(int id, float score, Bbox bbox, std::vector<float> &&mask)
      : Detection(id, score, bbox), mask(std::move(mask)) {}
  
  YOLOSegmentation(int id, float score, Bbox bbox, const char *class_name, std::vector<float> &&mask)
      : Detection(id, score, bbox, class_name), mask(std::move(mask)) {}
  
  friend bool operator>(const YOLOSegmentation &lhs, const YOLOSegmentation &rhs) {
    return (lhs.score > rhs.score);
  }

  friend std::ostream &operator<<(std::ostream &os, const YOLOSegmentation &seg) {
    os << "{"
       << R"("prob")"
       << ":" << std::fixed << std::setprecision(5) << seg.score << ","
       << R"("label")"
       << ":" << seg.id << ","
       << R"("class_name")"
       << ":"
       << "\"" << seg.class_name << "\""
       << "}";
    return os;
  }

  ~YOLOSegmentation() {}
} YOLOSeg;

struct MaskResultInfo {
  int32_t width = 0;
  int32_t height = 0;
  float h_base = 0;
  float w_base = 0;
  std::vector<Detection> det_info;
  std::vector<float> mask_info;
};

struct Perception {
  // Perception data
  std::vector<Detection> det;
  std::vector<Classification> cls;
  Parsing seg;
  MaskResultInfo mask;
  float h_base = 1;
  float w_base = 1;

  // Perception type
  enum {
    DET = (1 << 0),
    CLS = (1 << 1),
    SEG = (1 << 2),
    MASK = (1 << 3),
  } type;

  friend std::ostream &operator<<(std::ostream &os, Perception &perception) {
    os << "[";
    if (perception.type == Perception::DET) {
      auto &detection = perception.det;
      for (size_t i = 0; i < detection.size(); i++) {
        if (i != 0) {
          os << ",";
        }
        os << detection[i];
      }

    } else if (perception.type == Perception::CLS) {
      auto &cls = perception.cls;
      for (size_t i = 0; i < cls.size(); i++) {
        if (i != 0) {
          os << ",";
        }
        os << cls[i];
      }
    } else if (perception.type == Perception::SEG) {
      auto &seg = perception.seg;
      for (size_t i = 0; i < seg.seg.size(); i++) {
        if (i != 0) {
          os << ",";
        }
        os << static_cast<int>(seg.seg[i]);
      }
    } else if (perception.type == Perception::MASK) {
      auto &detection = perception.mask.det_info;
      for (size_t i = 0; i < detection.size(); i++) {
        if (i != 0) {
          os << ",";
        }
        os << detection[i];
      }
    }
    os << "]";
    return os;
  }
};

// 算法输出数据类型
struct DnnParserResult {
  Perception perception;
  void Reset() {
    perception.seg.seg.clear();
    perception.seg.data.clear();
    perception.det.clear();
    perception.cls.clear();
  }
};

}  // namespace output_parser
}  // namespace dnn_node
}  // namespace hobot
#endif  // _OUTPUT_PARSER_PERCEPTION_COMMON_H_
