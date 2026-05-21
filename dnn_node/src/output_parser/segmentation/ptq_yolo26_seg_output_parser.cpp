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

#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <queue>
#include <thread>

#include "rapidjson/document.h"
#include "rclcpp/rclcpp.hpp"

#include "dnn_node/util/output_parser/detection/nms.h"
#include "dnn_node/util/output_parser/neon_utils.h"
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

  // Pre-compute grid center coordinates
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

      // Inline argmax
      float max_logit = cur_cls_data[0];
      int id = 0;
      for (int c = 1; c < num_classes; ++c) {
        if (cur_cls_data[c] > max_logit) {
          max_logit = cur_cls_data[c];
          id = c;
        }
      }

      // Advance mask pointer (before any continue)
      int32_t *cur_mask_raw = mask_raw;
      float *cur_mask_float = mask_float;
      if (mask_raw) {
        mask_raw += num_mask;
      } else {
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

      if (xmax <= 0 || ymax <= 0) continue;
      if (xmin > xmax || ymin > ymax) continue;

      Bbox bbox(xmin, ymin, xmax, ymax);

      // NEON-accelerated mask coefficient dequantization
      std::vector<float> mask(num_mask);
      if (mask_quantized) {
        hobot::dnn_node::output_parser::neon_dequant_s32c_to_f32(
            mask.data(), cur_mask_raw, mask_scale, num_mask);
      } else {
        std::memcpy(mask.data(), cur_mask_float, num_mask * sizeof(float));
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

  RCLCPP_DEBUG(rclcpp::get_logger("Yolo26_seg_parser"),
               "PostProcess took %d ms",
               static_cast<int>(
                   std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - ts_start)
                       .count()));

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
  int num_mask = yolo26_seg_config_.num_mask;

  // Pre-compute proto parameters (needed for async dequantization)
  int proto_h = model_h / yolo26_seg_config_.strides[0] * 2;
  int proto_w = model_w / yolo26_seg_config_.strides[0] * 2;

  float valid_h_ratio =
      static_cast<float>(resized_img_h) / static_cast<float>(model_h);
  float valid_w_ratio =
      static_cast<float>(resized_img_w) / static_cast<float>(model_w);

  int valid_h = static_cast<int>(valid_h_ratio * proto_h);
  int valid_w = static_cast<int>(valid_w_ratio * proto_w);

  float proto_h_ratio = static_cast<float>(proto_h) / static_cast<float>(model_h);
  float proto_w_ratio = static_cast<float>(proto_w) / static_cast<float>(model_w);

  // Phase 1: Launch all parallel tasks
  //   - 3 ParseTensor tasks (one per stride layer)
  //   - 1 Proto dequantization task (runs in parallel with ParseTensor)

  auto ts_start = std::chrono::steady_clock::now();

  // --- ParseTensor async tasks ---
  std::vector<std::future<std::shared_ptr<std::vector<YOLOSeg>>>> futs;
  auto output_size = output_tensors.size() / 3;
  for (size_t i = 0; i < output_size; ++i) {
    auto fut = std::async(std::launch::async, [&output_tensors, i]() {
      std::shared_ptr<std::vector<YOLOSeg>> sp_det = nullptr;
      std::vector<YOLOSeg> _dets;
      ParseTensor(output_tensors[i * 3],
                  output_tensors[i * 3 + 1],
                  output_tensors[i * 3 + 2],
                  static_cast<int>(i),
                  _dets);
      if (!_dets.empty()) {
        sp_det = std::make_shared<std::vector<YOLOSeg>>(_dets);
      }
      return sp_det;
    });
    futs.push_back(std::move(fut));
  }

  // --- Proto dequantization async task (runs in parallel with ParseTensor) ---
  std::shared_ptr<DNNTensor> proto =
      output_tensors[output_tensors.size() - 1];
  proto->CACHE_INVALIDATE();

  int proto_quanti = proto->properties.quantiType;
  float *proto_data = nullptr;
  std::vector<float> proto_dequant_buf;
  std::future<void> proto_fut;

  if (proto_quanti != 0) {
    int16_t *proto_raw = proto->GetTensorData<int16_t>();
    float proto_scale = proto->properties.scale.scaleData[0];
    int total_elem = proto_h * proto_w * num_mask;
    proto_dequant_buf.resize(total_elem);

    // Launch dequantization as async task
    proto_fut = std::async(std::launch::async, [&]() {
      hobot::dnn_node::output_parser::neon_dequant_i16_to_f32(
          proto_dequant_buf.data(), proto_raw, proto_scale, total_elem);
    });
  } else {
    proto_data = proto->GetTensorData<float>();
  }

  // Phase 2: Wait for ParseTensor, merge detections
  std::vector<YOLOSeg> dets;
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

  int parse_tensor_time_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - ts_start)
          .count();
  ts_start = std::chrono::steady_clock::now();

  // Phase 3: NMS
  std::vector<YOLOSeg> results;
  yolo_seg_nms(dets, nms_threshold_, nms_top_k_, results, false);

  int nms_time_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - ts_start)
          .count();

  RCLCPP_DEBUG(rclcpp::get_logger("Yolo26_seg_parser"),
              "ParseTensor done: dets=%zu nms_results=%zu "
              "time_parse=%dus time_nms=%dus",
              dets.size(), results.size(),
              parse_tensor_time_us, nms_time_us);

  // Phase 4: Wait for proto dequantization (if not already done)
  if (proto_fut.valid()) {
    proto_fut.wait();
    proto_data = proto_dequant_buf.data();
  }

  ts_start = std::chrono::steady_clock::now();

  // Phase 5: Setup perception output
  perception.seg.valid_h = valid_h;
  perception.seg.valid_w = valid_w;
  perception.seg.height = static_cast<int>(model_h * valid_h_ratio);
  perception.seg.width = static_cast<int>(model_w * valid_w_ratio);
  perception.seg.data.resize(valid_h * valid_w);
  perception.seg.seg.resize(valid_h * valid_w);

  // Pre-populate detection results (before parallel mask generation)
  if (output_roi_) {
    for (const auto &result : results) {
      perception.det.emplace_back(
          result.id, result.score, result.bbox, result.class_name);
    }
  } else if (results.empty()) {
    RCLCPP_WARN_ONCE(rclcpp::get_logger("Yolo26_seg_parser"),
                     "Roi output is not enabled");
  }

  if (results.empty()) {
    RCLCPP_WARN(rclcpp::get_logger("Yolo26_seg_parser"),
                "No detections after NMS — masks will be empty");
    perception.seg.channel = 1;
    perception.seg.num_classes = yolo26_seg_config_.class_num;
    return 0;
  }

  // Phase 6: Multi-threaded mask generation with NEON dot product
  // Use tile-based parallelism: split rows across threads.
  // Each thread processes its row range for all detections.
  // Pre-clip boxes to proto coordinates for use in the strip workers.

  struct ClippedBox {
    int x1, y1, x2, y2;
    int id;
    float score;
    Bbox bbox;
    const char *class_name;
    std::vector<float> mask;
  };

  std::vector<ClippedBox> clipped;
  clipped.reserve(results.size());
  for (const auto &r : results) {
    int x1 = static_cast<int>(r.bbox.xmin * proto_w_ratio + 1.0f);
    int y1 = static_cast<int>(r.bbox.ymin * proto_h_ratio + 1.0f);
    int x2 = static_cast<int>(r.bbox.xmax * proto_w_ratio);
    int y2 = static_cast<int>(r.bbox.ymax * proto_h_ratio);

    if (x1 < 0) x1 = 0;
    if (y1 < 0) y1 = 0;
    if (x2 < 0) x2 = 0;
    if (y2 < 0) y2 = 0;
    if (x2 >= valid_w) x2 = valid_w - 1;
    if (y2 >= valid_h) y2 = valid_h - 1;

    if (x2 < x1 || y2 < y1) continue;

    clipped.push_back({x1, y1, x2, y2, r.id, r.score,
                       r.bbox, r.class_name, r.mask});
  }

  int num_threads = std::min(
      static_cast<int>(std::thread::hardware_concurrency()), 4);
  num_threads = std::min(num_threads, valid_h);  // at most 1 row per thread
  if (num_threads < 1) num_threads = 1;

  int rows_per_thread = (valid_h + num_threads - 1) / num_threads;

  std::vector<std::future<void>> mask_futs;
  for (int t = 0; t < num_threads; ++t) {
    int row_start = t * rows_per_thread;
    int row_end = std::min(row_start + rows_per_thread, valid_h);
    if (row_start >= row_end) break;

    mask_futs.push_back(std::async(std::launch::async, [&, row_start, row_end]() {
      for (const auto &cb : clipped) {
        // Only process rows that overlap with this detection
        int h_start = std::max(cb.y1, row_start);
        int h_end = std::min(cb.y2 + 1, row_end);
        if (h_start >= h_end) continue;

        int x1 = cb.x1;
        int x2 = cb.x2;
        const float *mask_data = cb.mask.data();

        for (int h = h_start; h < h_end; ++h) {
          const float *cur_proto =
              proto_data + (static_cast<int64_t>(h) * proto_w + x1) * num_mask;
          for (int w = x1; w <= x2; ++w) {
            float sum = hobot::dnn_node::output_parser::neon_dot_product_f32(
                mask_data, cur_proto, num_mask);
            if (sum > 0.0f) {
              hobot::dnn_node::output_parser::seg_background_adjust(
                  &perception.seg.seg[h * valid_w + w],
                  &perception.seg.data[h * valid_w + w],
                  cb.id,
                  background_id,
                  have_background);
            }
            cur_proto += num_mask;
          }
        }
      }
    }));
  }

  for (auto &f : mask_futs) {
    f.wait();
  }

  int mask_time_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - ts_start)
          .count();

  RCLCPP_INFO(rclcpp::get_logger("Yolo26_seg_parser"),
              "PostProcess timing: parse=%dus nms=%dus mask=%dus "
              "dets=%zu results=%zu threads=%d",
              parse_tensor_time_us, nms_time_us, mask_time_us,
              dets.size(), results.size(), num_threads);

  perception.seg.channel = 1;
  perception.seg.num_classes = yolo26_seg_config_.class_num;
  return 0;
}

}  // namespace parser_yolo26_seg
}  // namespace dnn_node
}  // namespace hobot
