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

#include "dnn_node/util/output_parser/segmentation/ptq_yolo26e_seg_output_parser.h"

#include <fstream>
#include <future>
#include <iostream>
#include <queue>

#include "rapidjson/document.h"
#include "rclcpp/rclcpp.hpp"

#include "dnn_node/util/output_parser/detection/nms.h"
#include "dnn_node/util/output_parser/utils.h"

namespace hobot {
namespace dnn_node {
namespace parser_yolo26e_seg {

/**
 * Finds the greatest element in the range [first, last)
 */
template <class ForwardIterator>
inline size_t argmax(ForwardIterator first, ForwardIterator last) {
  return std::distance(first, std::max_element(first, last));
}

/**
 * Config definition for Yolo26eSeg
 */
struct PTQYolo26eSegConfig {
  std::vector<int> strides;
  int class_num;
  int num_mask;
  std::vector<std::string> class_names;
  bool combined_output;

  std::string Str() {
    std::stringstream ss;
    ss << "strides: ";
    for (const auto &stride : strides) {
      ss << stride << " ";
    }
    ss << "; class_num: " << class_num;
    ss << "; num_mask: " << num_mask;
    ss << "; combined_output: " << combined_output;
    return ss.str();
  }
};

PTQYolo26eSegConfig default_yolo26e_seg_config = {
    {8, 16, 32},
    80,
    32,
    {"person",        "bicycle",      "car",
     "motorcycle",    "airplane",     "bus",
     "train",         "truck",        "boat",
     "traffic light", "fire hydrant", "stop sign",
     "parking meter", "bench",        "bird",
     "cat",           "dog",          "horse",
     "sheep",         "cow",          "elephant",
     "bear",          "zebra",        "giraffe",
     "backpack",      "umbrella",     "handbag",
     "tie",           "suitcase",     "frisbee",
     "skis",          "snowboard",    "sports ball",
     "kite",          "baseball bat", "baseball glove",
     "skateboard",    "surfboard",    "tennis racket",
     "bottle",        "wine glass",   "cup",
     "fork",          "knife",        "spoon",
     "bowl",          "banana",       "apple",
     "sandwich",      "orange",       "broccoli",
     "carrot",        "hot dog",      "pizza",
     "donut",         "cake",         "chair",
     "couch",         "potted plant", "bed",
     "dining table",  "toilet",       "tv",
     "laptop",        "mouse",        "remote",
     "keyboard",      "cell phone",   "microwave",
     "oven",          "toaster",      "sink",
     "refrigerator",  "book",         "clock",
     "vase",          "scissors",     "teddy bear",
     "hair drier",    "toothbrush"},
    true};

PTQYolo26eSegConfig yolo26e_seg_config_ = default_yolo26e_seg_config;
float score_threshold_ = 0.25;
float nms_threshold_ = 0.65;
int nms_top_k_ = 5000;
bool output_roi_ = true;
bool have_background = false;
int background_id = 0;
bool combined_output_ = true;

int InitClassNum(const int &class_num) {
  if (class_num > 0) {
    yolo26e_seg_config_.class_num = class_num;
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo26e_seg_parser"),
                 "class_num = %d is not allowed, only support class_num > 0",
                 class_num);
    return -1;
  }
  return 0;
}

int InitClassNames(const std::string &cls_name_file) {
  std::ifstream fi(cls_name_file);
  if (fi) {
    yolo26e_seg_config_.class_names.clear();
    std::string line;
    while (std::getline(fi, line)) {
      yolo26e_seg_config_.class_names.push_back(line);
      std::transform(line.begin(), line.end(), line.begin(), [](unsigned char c) {
        return std::tolower(c);
      });
      if (line.compare("background") == 0 || line.compare("bg") == 0) {
        have_background = true;
        background_id =
            static_cast<int>(yolo26e_seg_config_.class_names.size()) - 1;
      }
    }
    int size = yolo26e_seg_config_.class_names.size();
    if (size != yolo26e_seg_config_.class_num) {
      RCLCPP_ERROR(rclcpp::get_logger("Yolo26e_seg_parser"),
                   "class_names length %d is not equal to class_num %d", size,
                   yolo26e_seg_config_.class_num);
      return -1;
    }
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo26e_seg_parser"),
                 "can not open cls name file: %s", cls_name_file.c_str());
    return -1;
  }
  return 0;
}

int InitNumMask(const int &num_mask) {
  if (num_mask > 0) {
    yolo26e_seg_config_.num_mask = num_mask;
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo26e_seg_parser"),
                 "num_mask = %d is not allowed, only support num_mask > 0",
                 num_mask);
    return -1;
  }
  return 0;
}

int InitStrides(const std::vector<int> &strides,
                const int &model_output_count) {
  int size = strides.size();
  if (yolo26e_seg_config_.combined_output) {
    // yolo26e combined format: 1 detection tensor + 1 proto tensor = 2 outputs
    if (model_output_count != 2) {
      RCLCPP_WARN(rclcpp::get_logger("Yolo26e_seg_parser"),
                  "combined_output mode expects model_output_count=2, got %d",
                  model_output_count);
    }
  } else {
    // Per-stride format: cls + box + mask per stride + proto
    if (size * 3 + 1 != model_output_count) {
      RCLCPP_ERROR(
          rclcpp::get_logger("Yolo26e_seg_parser"),
          "strides size %d is not equal to model_output_count %d, expected %d",
          size, model_output_count, size * 3 + 1);
      return -1;
    }
  }
  yolo26e_seg_config_.strides.clear();
  for (size_t i = 0; i < strides.size(); i++) {
    yolo26e_seg_config_.strides.push_back(strides[i]);
  }
  return 0;
}

int LoadConfig(const rapidjson::Document &document) {
  int model_output_count = 0;
  if (document.HasMember("model_output_count")) {
    model_output_count = document["model_output_count"].GetInt();
    if (model_output_count <= 0) {
      RCLCPP_ERROR(rclcpp::get_logger("Yolo26e_seg_parser"),
                   "model_output_count = %d <= 0 is not allowed",
                   model_output_count);
      return -1;
    }
  }
  if (document.HasMember("combined_output")) {
    yolo26e_seg_config_.combined_output = document["combined_output"].GetBool();
  }
  combined_output_ = yolo26e_seg_config_.combined_output;
  if (document.HasMember("class_num")) {
    int class_num = document["class_num"].GetInt();
    if (InitClassNum(class_num) < 0) {
      return -1;
    }
  }
  if (document.HasMember("cls_names_list")) {
    std::string cls_name_file = document["cls_names_list"].GetString();
    if (InitClassNames(cls_name_file) < 0) {
      return -1;
    }
  }
  if (document.HasMember("num_mask")) {
    int num_mask = document["num_mask"].GetInt();
    if (InitNumMask(num_mask) < 0) {
      return -1;
    }
  }
  if (document.HasMember("strides")) {
    std::vector<int> strides;
    for (size_t i = 0; i < document["strides"].Size(); i++) {
      strides.push_back(document["strides"][i].GetInt());
    }
    if (InitStrides(strides, model_output_count) < 0) {
      return -1;
    }
  }
  if (document.HasMember("score_threshold")) {
    score_threshold_ = document["score_threshold"].GetFloat();
  }

  if (!combined_output_) {
    // Per-stride format: convert score threshold to logit space
    float safe_thres =
        std::max(1e-6f, std::min(score_threshold_, 1.0f - 1e-6f));
    score_threshold_ = -std::log(1.0f / safe_thres - 1.0f);
  }
  // Combined format: scores are already sigmoided, use raw threshold directly

  if (document.HasMember("nms_threshold")) {
    nms_threshold_ = document["nms_threshold"].GetFloat();
  }
  if (document.HasMember("nms_top_k")) {
    nms_top_k_ = document["nms_top_k"].GetInt();
  }
  if (document.HasMember("output_roi")) {
    output_roi_ = document["output_roi"].GetBool();
  }
  return 0;
}

int PostProcess(std::vector<std::shared_ptr<DNNTensor>> &output_tensors,
                int resized_img_h,
                int resized_img_w,
                int model_h,
                int model_w,
                Perception &perception);

// Per-stride tensor parsing (same as yolo26_seg)
void ParseTensorPerStride(std::shared_ptr<DNNTensor> clses,
                          std::shared_ptr<DNNTensor> boxes,
                          std::shared_ptr<DNNTensor> masks,
                          int layer,
                          std::vector<YOLOSeg> &dets) {
  clses->CACHE_INVALIDATE();
  boxes->CACHE_INVALIDATE();
  masks->CACHE_INVALIDATE();
  int num_classes = yolo26e_seg_config_.class_num;
  int num_mask = yolo26e_seg_config_.num_mask;
  int stride = yolo26e_seg_config_.strides[layer];

  int height, width;
  auto ret =
      hobot::dnn_node::output_parser::get_tensor_hw(boxes, &height, &width);
  if (ret != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo26e_seg_parser"),
                 "get_tensor_hw failed");
    return;
  }

  float *cls_data = clses->GetTensorData<float>();
  float *box_data = boxes->GetTensorData<float>();

  int32_t *mask_raw = nullptr;
  float *mask_float = nullptr;
  float *mask_scale = nullptr;
  bool mask_quantized = (masks->properties.quantiType != 0);

  if (mask_quantized) {
    mask_raw = masks->GetTensorData<int32_t>();
    mask_scale =
        reinterpret_cast<float *>(masks->properties.scale.scaleData);
  } else {
    mask_float = masks->GetTensorData<float>();
  }

  std::vector<float> col_center(width);
  std::vector<float> row_center(height);
  for (int w = 0; w < width; ++w) {
    col_center[w] = (static_cast<float>(w) + 0.5f) * stride;
  }
  for (int h = 0; h < height; ++h) {
    row_center[h] = (static_cast<float>(h) + 0.5f) * stride;
  }

  int det_count = 0;
  dets.reserve(dets.size() + height * width / 20);

  for (int h = 0; h < height; ++h) {
    float gc_y = row_center[h];
    for (int w = 0; w < width; ++w) {
      float *cur_cls_data = cls_data;
      float *cur_box_data = box_data;

      cls_data += num_classes;
      box_data += 4;

      float max_logit = cur_cls_data[0];
      int id = 0;
      for (int c = 1; c < num_classes; ++c) {
        if (cur_cls_data[c] > max_logit) {
          max_logit = cur_cls_data[c];
          id = c;
        }
      }

      int32_t *cur_mask_raw = mask_raw;
      float *cur_mask_float = mask_float;
      if (mask_raw) {
        mask_raw += num_mask;
      } else if (mask_float) {
        mask_float += num_mask;
      }

      if (max_logit < score_threshold_) {
        continue;
      }

      float confidence = 1.0f / (1.0f + std::exp(-max_logit));

      float gc_x = col_center[w];
      float xmin = gc_x - cur_box_data[0] * stride;
      float ymin = gc_y - cur_box_data[1] * stride;
      float xmax = gc_x + cur_box_data[2] * stride;
      float ymax = gc_y + cur_box_data[3] * stride;

      if (xmax <= 0 || ymax <= 0) {
        continue;
      }
      if (xmin > xmax || ymin > ymax) {
        continue;
      }

      Bbox bbox(xmin, ymin, xmax, ymax);

      std::vector<float> mask(num_mask, 0);
      for (int i = 0; i < num_mask; ++i) {
        if (mask_quantized) {
          mask[i] = static_cast<float>(cur_mask_raw[i]) * mask_scale[i];
        } else {
          mask[i] = cur_mask_float[i];
        }
      }
      dets.emplace_back(
          static_cast<int>(id),
          confidence,
          bbox,
          yolo26e_seg_config_.class_names[static_cast<int>(id)].c_str(),
          std::move(mask));
      det_count++;
    }
  }
  RCLCPP_DEBUG(rclcpp::get_logger("Yolo26e_seg_parser"),
              "ParseTensorPerStride layer=%d detections=%d", layer, det_count);
}

// Combined tensor parsing for yolo26e format
// Detection tensor: NCHW layout [1, C, N, 1] where C = 4(box)+num_classes+num_mask
// Boxes are in decoded xywh absolute coordinates, scores are already sigmoided
void ParseTensorCombined(std::shared_ptr<DNNTensor> det_tensor,
                         std::vector<YOLOSeg> &dets) {
  det_tensor->CACHE_INVALIDATE();
  int num_classes = yolo26e_seg_config_.class_num;
  int num_mask = yolo26e_seg_config_.num_mask;
  int channels = 4 + num_classes + num_mask;  // 4 box + 80 cls + 32 mask = 116

  auto &validShape = det_tensor->properties.validShape;

  // NCHW: dim[0]=N, dim[1]=C, dim[2]=H, dim[3]=W
  int num_anchors = validShape.dimensionSize[2] * validShape.dimensionSize[3];
  
  int aligned_hw = det_tensor->properties.stride[1] / det_tensor->properties.stride[3];
  

  RCLCPP_DEBUG(rclcpp::get_logger("Yolo26e_seg_parser"),
              "Combined tensor: validShape=[%d,%d,%d,%d] "
              "num_anchors=%d channels=%d aligned_hw=%d",
              validShape.dimensionSize[0], validShape.dimensionSize[1], validShape.dimensionSize[2], validShape.dimensionSize[3],
              num_anchors, channels, aligned_hw);

  float *data = det_tensor->GetTensorData<float>();

  int det_count = 0;
  dets.reserve(dets.size() + num_anchors / 20);

  // NCHW access: data[channel * aligned_hw + anchor_idx]
  for (int i = 0; i < num_anchors; ++i) {
    // Box: channels 0..3, xywh absolute coordinates → xyxy
    float cx = data[0 * aligned_hw + i];
    float cy = data[1 * aligned_hw + i];
    float bw = data[2 * aligned_hw + i];
    float bh = data[3 * aligned_hw + i];
    float xmin = cx - bw * 0.5f;
    float ymin = cy - bh * 0.5f;
    float xmax = cx + bw * 0.5f;
    float ymax = cy + bh * 0.5f;

    if (xmax <= 0 || ymax <= 0) continue;
    if (xmin > xmax || ymin > ymax) continue;

    // Class scores: channels 4..(4+num_classes-1), already sigmoided
    float max_score = data[4 * aligned_hw + i];
    int id = 0;
    for (int c = 1; c < num_classes; ++c) {
      float s = data[(4 + c) * aligned_hw + i];
      if (s > max_score) {
        max_score = s;
        id = c;
      }
    }

    if (max_score < score_threshold_) continue;

    Bbox bbox(xmin, ymin, xmax, ymax);

    // Mask coefficients: channels (4+num_classes)..(channels-1)
    std::vector<float> mask(num_mask, 0);
    for (int j = 0; j < num_mask; ++j) {
      mask[j] = data[(4 + num_classes + j) * aligned_hw + i];
    }

    dets.emplace_back(
        static_cast<int>(id),
        max_score,
        bbox,
        yolo26e_seg_config_.class_names[static_cast<int>(id)].c_str(),
        std::move(mask));
    det_count++;
  }
  RCLCPP_DEBUG(rclcpp::get_logger("Yolo26e_seg_parser"),
              "ParseTensorCombined detections=%d", det_count);
}

int32_t Parse(
    const std::shared_ptr<hobot::dnn_node::DnnNodeOutput> &node_output,
    const int resized_img_h,
    const int resized_img_w,
    const int model_h,
    const int model_w,
    std::shared_ptr<DnnParserResult> &result) {
  if (!result) {
    result = std::make_shared<DnnParserResult>();
  }

  int ret = PostProcess(node_output->output_tensors,
                        resized_img_h,
                        resized_img_w,
                        model_h,
                        model_w,
                        result->perception);

  if (ret != 0) {
    RCLCPP_INFO(rclcpp::get_logger("Yolo26e_seg_parser"),
                "postprocess return error, code = %d",
                ret);
  }

  std::stringstream ss;
  ss << "Yolo26e_seg_parser parse finished, predict result: "
     << result->perception;
  RCLCPP_DEBUG(
      rclcpp::get_logger("Yolo26e_seg_parser"), "%s", ss.str().c_str());
  return ret;
}

int PostProcess(std::vector<std::shared_ptr<DNNTensor>> &output_tensors,
                int resized_img_h,
                int resized_img_w,
                int model_h,
                int model_w,
                Perception &perception) {
  perception.type = Perception::SEG;
  std::vector<YOLOSeg> dets;

  auto ts_start = std::chrono::steady_clock::now();

  if (combined_output_) {
    // Combined format: output_tensors[0] = detection, output_tensors[1] = proto
    if (output_tensors.size() < 2) {
      RCLCPP_ERROR(rclcpp::get_logger("Yolo26e_seg_parser"),
                   "Combined output expects at least 2 tensors, got %zu",
                   output_tensors.size());
      return -1;
    }
    ParseTensorCombined(output_tensors[0], dets);
  } else {
    // Per-stride format: parallel parsing per stride (same as yolo26_seg)
    std::vector<std::future<std::shared_ptr<std::vector<YOLOSeg>>>> futs;
    auto output_size = output_tensors.size() / 3;
    for (size_t i = 0; i < output_size; ++i) {
      auto fut = std::async(std::launch::async, [&output_tensors, i]() {
        std::shared_ptr<std::vector<YOLOSeg>> sp_det = nullptr;
        std::vector<YOLOSeg> _dets;
        auto start = std::chrono::steady_clock::now();
        ParseTensorPerStride(output_tensors[i * 3],
                    output_tensors[i * 3 + 1],
                    output_tensors[i * 3 + 2],
                    static_cast<int>(i),
                    _dets);
        int time_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start)
                .count();
        RCLCPP_DEBUG_STREAM(rclcpp::get_logger("Yolo26e_seg_parser"),
                            "parse tensor "
                                << i << " cost [" << time_ms << "]");
        if (!_dets.empty()) {
          sp_det = std::make_shared<std::vector<YOLOSeg>>(_dets);
        }
        return sp_det;
      });
      futs.push_back(std::move(fut));
    }
    for (size_t i = 0; i < futs.size(); i++) {
      if (!futs[i].valid()) {
        RCLCPP_ERROR(rclcpp::get_logger("Yolo26e_seg_parser"),
                     "fut is not valid");
        return -1;
      }
      futs[i].wait();
      auto det = futs[i].get();
      if (det) {
        dets.insert(dets.end(), std::make_move_iterator(det->begin()),
                    std::make_move_iterator(det->end()));
      }
    }
  }

  int parse_tensor_time_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - ts_start)
          .count();
  ts_start = std::chrono::steady_clock::now();

  std::vector<YOLOSeg> results;
  yolo_seg_nms(dets, nms_threshold_, nms_top_k_, results, false);

  int nms_time_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - ts_start)
          .count();

  RCLCPP_DEBUG(rclcpp::get_logger("Yolo26e_seg_parser"),
              "Parse done: dets=%zu nms_results=%zu time_parse=%dms time_nms=%dms",
              dets.size(), results.size(), parse_tensor_time_ms, nms_time_ms);

  // proto is the last output tensor
  std::shared_ptr<DNNTensor> proto =
      output_tensors[output_tensors.size() - 1];
  proto->CACHE_INVALIDATE();

  int num_mask = yolo26e_seg_config_.num_mask;
  int proto_h = model_h / yolo26e_seg_config_.strides[0] * 2;
  int proto_w = model_w / yolo26e_seg_config_.strides[0] * 2;

  int proto_quanti = proto->properties.quantiType;
  auto &pvs = proto->properties.validShape;
  RCLCPP_DEBUG(rclcpp::get_logger("Yolo26e_seg_parser"),
              "Proto: quantiType=%d validShape=[%d,%d,%d,%d] "
              "computed_h=%d computed_w=%d num_mask=%d",
              proto_quanti,
              pvs.dimensionSize[0], pvs.dimensionSize[1],
              pvs.dimensionSize[2], pvs.dimensionSize[3],
              proto_h, proto_w, num_mask);

  float valid_h_ratio =
      static_cast<float>(resized_img_h) / static_cast<float>(model_h);
  float valid_w_ratio =
      static_cast<float>(resized_img_w) / static_cast<float>(model_w);
  float proto_h_ratio = static_cast<float>(proto_h) / static_cast<float>(model_h);
  float proto_w_ratio = static_cast<float>(proto_w) / static_cast<float>(model_w);

  int valid_h = static_cast<int>(valid_h_ratio * proto_h);
  int valid_w = static_cast<int>(valid_w_ratio * proto_w);

  RCLCPP_DEBUG(rclcpp::get_logger("Yolo26e_seg_parser"),
              "Seg dims: valid_h=%d valid_w=%d proto_h_ratio=%.4f proto_w_ratio=%.4f "
              "resized_img=%dx%d model=%dx%d",
              valid_h, valid_w, proto_h_ratio, proto_w_ratio,
              resized_img_h, resized_img_w, model_h, model_w);

  perception.seg.valid_h = valid_h;
  perception.seg.valid_w = valid_w;
  perception.seg.height = static_cast<int>(model_h * valid_h_ratio);
  perception.seg.width = static_cast<int>(model_w * valid_w_ratio);

  // Proto data: may be quantized (S16 + per-tensor SCALE) or float (NONE)
  float *proto_data = nullptr;
  std::vector<float> proto_dequant_buf;
  if (proto_quanti != 0) {
    int16_t *proto_raw = proto->GetTensorData<int16_t>();
    float proto_scale = proto->properties.scale.scaleData[0];
    int total_elem = proto_h * proto_w * num_mask;
    proto_dequant_buf.resize(total_elem);
    for (int i = 0; i < total_elem; ++i) {
      proto_dequant_buf[i] = static_cast<float>(proto_raw[i]) * proto_scale;
    }
    proto_data = proto_dequant_buf.data();
  } else {
    proto_data = proto->GetTensorData<float>();
  }

  // The model declares NCHW layout with stride confirming it:
  // stride[1]/sizeof(float) = 102400/4 = 25600 = 160*160 = H*W
  // This is definitively NCHW: proto_data[c * H * W + h * W + w]
  bool proto_is_nchw = true;
  RCLCPP_DEBUG(rclcpp::get_logger("Yolo26e_seg_parser"),
              "Proto layout: NCHW (dims=[%d,%d,%d,%d])",
              pvs.dimensionSize[0], pvs.dimensionSize[1],
              pvs.dimensionSize[2], pvs.dimensionSize[3]);

  perception.seg.data.resize(valid_h * valid_w);
  perception.seg.seg.resize(valid_h * valid_w);

  if (results.empty()) {
    RCLCPP_WARN(rclcpp::get_logger("Yolo26e_seg_parser"),
                "No detections after NMS — masks will be empty");
  }

  int mask_pixel_count = 0;
  for (const auto &result : results) {
    const auto &mask = result.mask;
    const auto &box = result.bbox;

    if (output_roi_) {
      perception.det.emplace_back(
          result.id, result.score, result.bbox, result.class_name);
    } else {
      RCLCPP_WARN_ONCE(rclcpp::get_logger("Yolo26e_seg_parser"),
                       "Roi output is not enabled");
    }

    int x1_crop = static_cast<int>(box.xmin * proto_w_ratio + 1.0f);
    int y1_crop = static_cast<int>(box.ymin * proto_h_ratio + 1.0f);
    int x2_crop = static_cast<int>(box.xmax * proto_w_ratio);
    int y2_crop = static_cast<int>(box.ymax * proto_h_ratio);
    if (x1_crop < 0) x1_crop = 0;
    if (y1_crop < 0) y1_crop = 0;
    if (x2_crop < 0) x2_crop = 0;
    if (y2_crop < 0) y2_crop = 0;
    if (x2_crop >= perception.seg.valid_w) x2_crop = perception.seg.valid_w - 1;
    if (y2_crop >= perception.seg.valid_h) y2_crop = perception.seg.valid_h - 1;
    if (x1_crop >= 0 && x1_crop < perception.seg.valid_w &&
        x2_crop >= x1_crop && x2_crop < perception.seg.valid_w &&
        y1_crop >= 0 && y1_crop < perception.seg.valid_h &&
        y2_crop >= y1_crop && y2_crop < perception.seg.valid_h) {
    } else {
      RCLCPP_ERROR(rclcpp::get_logger("Yolo26e_seg_parser"),
                   "invalid box: [%d, %d, %d, %d], valid w: %d, h: %d",
                   x1_crop, y1_crop, x2_crop, y2_crop,
                   perception.seg.valid_w, perception.seg.valid_h);
      continue;
    }

    float sum;
    int proto_spatial = proto_h * proto_w;
    for (int h = y1_crop; h < y2_crop && h < valid_h; ++h) {
      for (int w = x1_crop; w < x2_crop && w < valid_w; ++w) {
        sum = 0.0f;
        if (proto_is_nchw) {
          // NCHW: proto_data[c * H * W + h * W + w]
          for (int i = 0; i < num_mask; ++i) {
            sum += mask[i] * proto_data[i * proto_spatial + h * proto_w + w];
          }
        } else {
          // NHWC: proto_data[(h * W + w) * C + i]
          int base = (h * proto_w + w) * num_mask;
          for (int i = 0; i < num_mask; ++i) {
            sum += mask[i] * proto_data[base + i];
          }
        }
        if (sum > 0.0f) {
          hobot::dnn_node::output_parser::seg_background_adjust(
              &perception.seg.seg[h * valid_w + w],
              &perception.seg.data[h * valid_w + w],
              result.id,
              background_id,
              have_background);
          mask_pixel_count++;
        }
      }
    }
  }
  RCLCPP_DEBUG(rclcpp::get_logger("Yolo26e_seg_parser"),
              "Mask generation done: total_pixels=%d", mask_pixel_count);
  perception.seg.channel = 1;
  perception.seg.num_classes = yolo26e_seg_config_.class_num;
  return 0;
}

}  // namespace parser_yolo26e_seg
}  // namespace dnn_node
}  // namespace hobot
