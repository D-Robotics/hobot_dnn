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
#include "dnn_node/util/output_parser/segmentation/ptq_yolo26_seg_output_parser.h"

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
namespace parser_yolo26_seg {

/**
 * Finds the greatest element in the range [first, last)
 * @tparam[in] ForwardIterator: iterator type
 * @param[in] first: fist iterator
 * @param[in] last: last iterator
 * @return Iterator to the greatest element in the range [first, last)
 */
template <class ForwardIterator>
inline size_t argmax(ForwardIterator first, ForwardIterator last) {
  return std::distance(first, std::max_element(first, last));
}

/**
 * Config definition for Yolo26Seg
 */
struct PTQYolo26SegConfig {
  std::vector<int> strides;
  int class_num;
  int num_mask;
  std::vector<std::string> class_names;

  std::string Str() {
    std::stringstream ss;
    ss << "strides: ";
    for (const auto &stride : strides) {
      ss << stride << " ";
    }

    ss << "; class_num: " << class_num;
    ss << "; num_mask: " << num_mask;
    return ss.str();
  }
};

PTQYolo26SegConfig default_yolo26_seg_config = {
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
     "hair drier",    "toothbrush"}};

PTQYolo26SegConfig yolo26_seg_config_ = default_yolo26_seg_config;
float score_threshold_ = 0.25;
float nms_threshold_ = 0.65;
int nms_top_k_ = 5000;
bool output_roi_ = true;
bool have_background = false;
int background_id = 0;

int InitClassNum(const int &class_num) {
  if (class_num > 0) {
    yolo26_seg_config_.class_num = class_num;
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo26_seg_parser"),
                 "class_num = %d is not allowed, only support class_num > 0",
                 class_num);
    return -1;
  }
  return 0;
}

int InitClassNames(const std::string &cls_name_file) {
  std::ifstream fi(cls_name_file);
  if (fi) {
    yolo26_seg_config_.class_names.clear();
    std::string line;
    while (std::getline(fi, line)) {
      yolo26_seg_config_.class_names.push_back(line);
      std::transform(line.begin(), line.end(), line.begin(), [](unsigned char c) {
        return std::tolower(c);
      });
      if (line.compare("background") == 0 || line.compare("bg") == 0) {
        have_background = true;
        background_id =
            static_cast<int>(yolo26_seg_config_.class_names.size()) - 1;
      }
    }
    int size = yolo26_seg_config_.class_names.size();
    if (size != yolo26_seg_config_.class_num) {
      RCLCPP_ERROR(rclcpp::get_logger("Yolo26_seg_parser"),
                   "class_names length %d is not equal to class_num %d", size,
                   yolo26_seg_config_.class_num);
      return -1;
    }
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo26_seg_parser"),
                 "can not open cls name file: %s", cls_name_file.c_str());
    return -1;
  }
  return 0;
}

int InitNumMask(const int &num_mask) {
  if (num_mask > 0) {
    yolo26_seg_config_.num_mask = num_mask;
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo26_seg_parser"),
                 "num_mask = %d is not allowed, only support num_mask > 0",
                 num_mask);
    return -1;
  }
  return 0;
}

int InitStrides(const std::vector<int> &strides,
                const int &model_output_count) {
  int size = strides.size();
  if (size * 3 + 1 != model_output_count) {
    RCLCPP_ERROR(
        rclcpp::get_logger("Yolo26_seg_parser"),
        "strides size %d is not equal to model_output_count %d, expected %d",
        size, model_output_count, size * 3 + 1);
    return -1;
  }
  yolo26_seg_config_.strides.clear();
  for (size_t i = 0; i < strides.size(); i++) {
    yolo26_seg_config_.strides.push_back(strides[i]);
  }
  return 0;
}

int LoadConfig(const rapidjson::Document &document) {
  int model_output_count = 0;
  if (document.HasMember("model_output_count")) {
    model_output_count = document["model_output_count"].GetInt();
    if (model_output_count <= 0) {
      RCLCPP_ERROR(rclcpp::get_logger("Yolo26_seg_parser"),
                   "model_output_count = %d <= 0 is not allowed",
                   model_output_count);
      return -1;
    }
  }
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

  // Convert score threshold to logit space: sigmoid(x) >= score  ⇔  x >= logit_thres.
  // This lets us filter on raw logits without computing sigmoid for every cell.
  float safe_thres =
      std::max(1e-6f, std::min(score_threshold_, 1.0f - 1e-6f));
  score_threshold_ = -std::log(1.0f / safe_thres - 1.0f);

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

void ParseTensor(std::shared_ptr<DNNTensor> clses,
                 std::shared_ptr<DNNTensor> boxes,
                 std::shared_ptr<DNNTensor> masks,
                 int layer,
                 std::vector<YOLOSeg> &dets) {
  clses->CACHE_INVALIDATE();
  boxes->CACHE_INVALIDATE();
  masks->CACHE_INVALIDATE();
  int num_classes = yolo26_seg_config_.class_num;
  int num_mask = yolo26_seg_config_.num_mask;
  int stride = yolo26_seg_config_.strides[layer];

  int height, width;
  auto ret =
      hobot::dnn_node::output_parser::get_tensor_hw(boxes, &height, &width);
  if (ret != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo26_seg_parser"),
                 "get_tensor_hw failed");
    return;
  }

  // YOLO26 uses direct LTRB offsets (4 values), no distribution-based decoding
  float *cls_data = clses->GetTensorData<float>();
  float *box_data = boxes->GetTensorData<float>();

  // Mask data: may be quantized (S32 + per-channel SCALE) or float (NONE)
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

  RCLCPP_DEBUG(rclcpp::get_logger("Yolo26_seg_parser"),
              "ParseTensor layer=%d masks quantized=%d h=%d w=%d",
              layer, mask_quantized, height, width);

  // Pre-compute grid center coordinates to avoid redundant (w+0.5)*stride per cell
  std::vector<float> col_center(width);
  std::vector<float> row_center(height);
  for (int w = 0; w < width; ++w) {
    col_center[w] = (static_cast<float>(w) + 0.5f) * stride;
  }
  for (int h = 0; h < height; ++h) {
    row_center[h] = (static_cast<float>(h) + 0.5f) * stride;
  }

  int det_count = 0;
  // Pre-allocate to avoid repeated realloc (typical detection density < 5%)
  dets.reserve(dets.size() + height * width / 20);

  for (int h = 0; h < height; ++h) {
    float gc_y = row_center[h];
    for (int w = 0; w < width; ++w) {
      float *cur_cls_data = cls_data;
      float *cur_box_data = box_data;

      cls_data += num_classes;
      box_data += 4;  // ltrb

      // Inline argmax: faster than std::max_element + std::distance on ARM
      float max_logit = cur_cls_data[0];
      int id = 0;
      for (int c = 1; c < num_classes; ++c) {
        if (cur_cls_data[c] > max_logit) {
          max_logit = cur_cls_data[c];
          id = c;
        }
      }

      // Advance mask pointer for every cell (before any continue)
      int32_t *cur_mask_raw = mask_raw;
      float *cur_mask_float = mask_float;
      if (mask_raw) {
        mask_raw += num_mask;
      } else {
        mask_float += num_mask;
      }

      // threshold in logit space
      if (max_logit < score_threshold_) {
        continue;
      }

      float confidence = 1.0f / (1.0f + std::exp(-max_logit));

      // YOLO26 box encoding: (left, top, right, bottom) distances from anchor center.
      // Decoded as:  x1 = cx - l*stride,  y1 = cy - t*stride,
      //              x2 = cx + r*stride,  y2 = cy + b*stride
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
          yolo26_seg_config_.class_names[static_cast<int>(id)].c_str(),
          std::move(mask));
      det_count++;
    }
  }
  RCLCPP_DEBUG(rclcpp::get_logger("Yolo26_seg_parser"),
              "ParseTensor layer=%d detections=%d", layer, det_count);
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

  auto ts_start = std::chrono::steady_clock::now();
  int ret = PostProcess(node_output->output_tensors,
                        resized_img_h,
                        resized_img_w,
                        model_h,
                        model_w,
                        result->perception);

  int process_time_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - ts_start)
          .count();

  if (ret != 0) {
    RCLCPP_INFO(rclcpp::get_logger("Yolo26_seg_parser"),
                "postprocess return error, code = %d",
                ret);
  }

  std::stringstream ss;
  ss << "Yolo26_seg_parser parse finished, predict result: "
     << result->perception;
  RCLCPP_DEBUG(
      rclcpp::get_logger("Yolo26_seg_parser"), "%s", ss.str().c_str());
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
  std::vector<std::future<std::shared_ptr<std::vector<YOLOSeg>>>> futs;
  // output_tensors layout: [cls_s8, box_s8, mc_s8, cls_s16, box_s16, mc_s16,
  //                         cls_s32, box_s32, mc_s32, proto]
  // Each stride consumes 3 consecutive tensors; proto is the last one.
  auto output_size = output_tensors.size() / 3;
  for (size_t i = 0; i < output_size; ++i) {
    auto fut = std::async(std::launch::async, [&output_tensors, i]() {
      std::shared_ptr<std::vector<YOLOSeg>> sp_det = nullptr;
      std::vector<YOLOSeg> _dets;
      auto start = std::chrono::steady_clock::now();
      ParseTensor(output_tensors[i * 3],
                  output_tensors[i * 3 + 1],
                  output_tensors[i * 3 + 2],
                  static_cast<int>(i),
                  _dets);
      int time_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - start)
              .count();
      RCLCPP_DEBUG_STREAM(rclcpp::get_logger("Yolo26_seg_parser"),
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
      RCLCPP_ERROR(rclcpp::get_logger("Yolo26_seg_parser"),
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

  RCLCPP_DEBUG(rclcpp::get_logger("Yolo26_seg_parser"),
              "ParseTensor done: dets=%zu nms_results=%zu time_parse=%dms time_nms=%dms",
              dets.size(), results.size(), parse_tensor_time_ms, nms_time_ms);

  // the last output tensor is proto
  std::shared_ptr<DNNTensor> proto =
      output_tensors[output_tensors.size() - 1];
  proto->CACHE_INVALIDATE();

  int num_mask = yolo26_seg_config_.num_mask;
  // Compute proto spatial size from model dimensions (same as yolo8_seg).
  // Do NOT use get_tensor_hw() — it may misread dimensions due to NHWC/NCHW
  // layout confusion (interpreting C dim as W when quantizeAxis is mismatched).
  int proto_h = model_h / yolo26_seg_config_.strides[0] * 2;
  int proto_w = model_w / yolo26_seg_config_.strides[0] * 2;

  int proto_quanti = proto->properties.quantiType;
  auto &vs = proto->properties.validShape;
  RCLCPP_DEBUG(rclcpp::get_logger("Yolo26_seg_parser"),
              "Proto: quantiType=%d validShape=[%d,%d,%d,%d] "
              "computed_h=%d computed_w=%d num_mask=%d",
              proto_quanti,
              vs.dimensionSize[0], vs.dimensionSize[1],
              vs.dimensionSize[2], vs.dimensionSize[3],
              proto_h, proto_w, num_mask);

  // valid_h_ratio accounts for letterbox padding: the resized image may not fill
  // the entire model input area (e.g. 640×480 in a 640×640 model → ratio=0.75).
  float valid_h_ratio =
      static_cast<float>(resized_img_h) / static_cast<float>(model_h);
  float valid_w_ratio =
      static_cast<float>(resized_img_w) / static_cast<float>(model_w);

  // Ratio to map box coordinates from model space (e.g. 0–640) to proto space (0–160).
  float proto_h_ratio = static_cast<float>(proto_h) / static_cast<float>(model_h);
  float proto_w_ratio = static_cast<float>(proto_w) / static_cast<float>(model_w);

  int valid_h = static_cast<int>(valid_h_ratio * proto_h);
  int valid_w = static_cast<int>(valid_w_ratio * proto_w);

  RCLCPP_DEBUG(rclcpp::get_logger("Yolo26_seg_parser"),
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

  perception.seg.data.resize(valid_h * valid_w);
  perception.seg.seg.resize(valid_h * valid_w);

  if (results.empty()) {
    RCLCPP_WARN(rclcpp::get_logger("Yolo26_seg_parser"),
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
      RCLCPP_WARN_ONCE(rclcpp::get_logger("Yolo26_seg_parser"),
                       "Roi output is not enabled");
    }

    int x1_crop = static_cast<int>(box.xmin * proto_w_ratio + 1.0f);
    int y1_crop = static_cast<int>(box.ymin * proto_h_ratio + 1.0f);
    int x2_crop = static_cast<int>(box.xmax * proto_w_ratio);
    int y2_crop = static_cast<int>(box.ymax * proto_h_ratio);
    if (x1_crop < 0) {
      x1_crop = 0;
    }
    if (y1_crop < 0) {
      y1_crop = 0;
    }
    if (x2_crop < 0) {
      x2_crop = 0;
    }
    if (y2_crop < 0) {
      y2_crop = 0;
    }
    if (x2_crop >= perception.seg.valid_w) {
      x2_crop = perception.seg.valid_w - 1;
    }
    if (y2_crop >= perception.seg.valid_h) {
      y2_crop = perception.seg.valid_h - 1;
    }
    if (x1_crop >= 0 && x1_crop < perception.seg.valid_w &&
        x2_crop >= x1_crop && x2_crop < perception.seg.valid_w &&
        y1_crop >= 0 && y1_crop < perception.seg.valid_h &&
        y2_crop >= y1_crop && y2_crop < perception.seg.valid_h) {
      // check success
    } else {
      RCLCPP_ERROR(rclcpp::get_logger("Yolo26_seg_parser"),
                   "invalid box: [%d, %d, %d, %d], valid w: %d, h: %d",
                   x1_crop, y1_crop, x2_crop, y2_crop,
                   perception.seg.valid_w, perception.seg.valid_h);
      continue;
    }

    float sum;
    for (int h = y1_crop; h < y2_crop && h < valid_h; ++h) {
      float *cur_proto_data =
          proto_data + (h * proto_w + x1_crop) * num_mask;
      for (int w = x1_crop; w < x2_crop && w < valid_w; ++w) {
        // For each pixel (h,w): mask = sigmoid( Σ_i mc[i] · proto[h,w,i] )
        // sigmoid(x) > 0.5  ⇔  x > 0, so we threshold on the raw dot product.
        sum = 0.0f;
        for (int i = 0; i < num_mask; ++i) {
          sum += mask[i] * cur_proto_data[i];
        }
        // sigmoid(sum) > 0.5 is equivalent to sum > 0
        if (sum > 0.0f) {
          hobot::dnn_node::output_parser::seg_background_adjust(
              &perception.seg.seg[h * valid_w + w],
              &perception.seg.data[h * valid_w + w],
              result.id,
              background_id,
              have_background);
          mask_pixel_count++;
        }
        cur_proto_data += num_mask;
      }
    }
  }
  RCLCPP_DEBUG(rclcpp::get_logger("Yolo26_seg_parser"),
              "Mask generation done: total_pixels=%d", mask_pixel_count);
  perception.seg.channel = 1;
  perception.seg.num_classes = yolo26_seg_config_.class_num;
  return 0;
}

}  // namespace parser_yolo26_seg
}  // namespace dnn_node
}  // namespace hobot
