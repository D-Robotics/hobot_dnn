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

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "dnn/hb_sys.h"
#include "rclcpp/rclcpp.hpp"

#include "include/util/image_proc.h"

namespace hobot {
namespace dnn_node {

std::shared_ptr<NV12PyramidInput> ImageProc::GetNV12PyramidFromNV12Img(
    const char *in_img_data,
    const int &in_img_height,
    const int &in_img_width,
    const int &scaled_img_height,
    const int &scaled_img_width) {
  auto *y = new hbSysMem;
  auto *uv = new hbSysMem;
  auto w_stride = ALIGN_16(scaled_img_width);
  hbSysAllocCachedMem(y, scaled_img_height * w_stride);
  hbSysAllocCachedMem(uv, scaled_img_height / 2 * w_stride);
  //内存初始化
  memset(y->virAddr, 0, scaled_img_height * w_stride);
  memset(uv->virAddr, 0, scaled_img_height / 2 * w_stride);

  const uint8_t *data = reinterpret_cast<const uint8_t *>(in_img_data);
  auto *hb_y_addr = reinterpret_cast<uint8_t *>(y->virAddr);
  auto *hb_uv_addr = reinterpret_cast<uint8_t *>(uv->virAddr);
  int copy_w = std::min(in_img_width, scaled_img_width);
  int copy_h = std::min(in_img_height, scaled_img_height);

  // padding y
  for (int h = 0; h < copy_h; ++h) {
    auto *raw = hb_y_addr + h * w_stride;
    auto *src = data + h * in_img_width;
    memcpy(raw, src, copy_w);
  }

  // padding uv
  auto uv_data = in_img_data + in_img_height * in_img_width;
  for (int32_t h = 0; h < copy_h / 2; ++h) {
    auto *raw = hb_uv_addr + h * w_stride;
    auto *src = uv_data + h * in_img_width;
    memcpy(raw, src, copy_w);
  }

  hbSysFlushMem(y, HB_SYS_MEM_CACHE_CLEAN);
  hbSysFlushMem(uv, HB_SYS_MEM_CACHE_CLEAN);
  auto pyramid = new NV12PyramidInput;
  pyramid->width = scaled_img_width;
  pyramid->height = scaled_img_height;
  pyramid->y_vir_addr = y->virAddr;
  pyramid->y_phy_addr = y->phyAddr;
  pyramid->y_stride = w_stride;
  pyramid->uv_vir_addr = uv->virAddr;
  pyramid->uv_phy_addr = uv->phyAddr;
  pyramid->uv_stride = w_stride;
  return std::shared_ptr<NV12PyramidInput>(pyramid,
                                           [y, uv](NV12PyramidInput *pyramid) {
                                             // Release memory after deletion
                                             hbSysFreeMem(y);
                                             hbSysFreeMem(uv);
                                             delete y;
                                             delete uv;
                                             delete pyramid;
                                           });
}

std::shared_ptr<NV12PyramidInput> ImageProc::GetNV12PyramidFromNV12Img(
    const char *in_img_data,
    const int &in_img_height,
    const int &in_img_width,
    const int &scaled_img_height,
    const int &scaled_img_width,
    int &padding_l,
    int &padding_t,
    int &padding_r,
    int &padding_b) {
  // 1 要求输入图片分辨率小于模型输入分辨率
  if (in_img_width > scaled_img_width && in_img_height > scaled_img_height) {
    return nullptr;
  }

  // 2 计算padding参数
  auto w_stride = ALIGN_16(scaled_img_width);
  if (w_stride > in_img_width) {
    // 需要在左边padding空相素
    padding_l = (w_stride - in_img_width) / 2;
    // 取偶数
    padding_l = padding_l % 2 == 0 ? padding_l : padding_l + 1;
  } else {
    padding_l = 0;
  }
  if (scaled_img_height > in_img_height) {
    // 需要在上方padding空相素
    padding_t = (scaled_img_height - in_img_height) / 2;
    padding_t = padding_t % 2 == 0 ? padding_t : padding_t + 1;
  } else {
    padding_t = 0;
  }
  padding_r = scaled_img_width - in_img_width - padding_l;
  padding_b = scaled_img_height - in_img_height - padding_t;

  // 3 申请内存并初始化
  auto *y = new hbSysMem;
  auto *uv = new hbSysMem;
  hbSysAllocCachedMem(y, scaled_img_height * w_stride);
  hbSysAllocCachedMem(uv, scaled_img_height / 2 * w_stride);
  memset(y->virAddr, 0, scaled_img_height * w_stride);
  memset(uv->virAddr, 0, scaled_img_height / 2 * w_stride);

  // 4 拷贝数据并padding
  const uint8_t *data = reinterpret_cast<const uint8_t *>(in_img_data);
  auto *hb_y_addr =
      reinterpret_cast<uint8_t *>(y->virAddr) + padding_t * w_stride;
  auto *hb_uv_addr =
      reinterpret_cast<uint8_t *>(uv->virAddr) + (padding_t / 2) * w_stride;
  // padding y
  for (uint32_t h = 0; h < in_img_height; ++h) {
    auto *raw = hb_y_addr + h * w_stride + padding_l;
    auto *src = data + h * in_img_width;
    memcpy(raw, src, in_img_width);
  }
  // padding uv
  auto uv_data = in_img_data + in_img_height * in_img_width;
  for (uint32_t h = 0; h < in_img_height / 2; ++h) {
    auto *raw = hb_uv_addr + h * w_stride + padding_l;
    auto *src = uv_data + h * in_img_width;
    memcpy(raw, src, in_img_width);
  }

  // 5 生成pym数据
  hbSysFlushMem(y, HB_SYS_MEM_CACHE_CLEAN);
  hbSysFlushMem(uv, HB_SYS_MEM_CACHE_CLEAN);
  auto pyramid = new NV12PyramidInput;
  pyramid->width = scaled_img_width;
  pyramid->height = scaled_img_height;
  pyramid->y_vir_addr = y->virAddr;
  pyramid->y_phy_addr = y->phyAddr;
  pyramid->y_stride = w_stride;
  pyramid->uv_vir_addr = uv->virAddr;
  pyramid->uv_phy_addr = uv->phyAddr;
  pyramid->uv_stride = w_stride;
  return std::shared_ptr<NV12PyramidInput>(pyramid,
                                           [y, uv](NV12PyramidInput *pyramid) {
                                             // Release memory after deletion
                                             hbSysFreeMem(y);
                                             hbSysFreeMem(uv);
                                             delete y;
                                             delete uv;
                                             delete pyramid;
                                           });
}

std::shared_ptr<NV12PyramidInput> ImageProc::GetNV12PyramidFromBGRImg(
    const cv::Mat &bgr_mat, int scaled_img_height, int scaled_img_width) {
  cv::Mat nv12_mat;
  cv::Mat mat_tmp;
  mat_tmp.create(scaled_img_height, scaled_img_width, bgr_mat.type());
  cv::resize(bgr_mat, mat_tmp, mat_tmp.size(), 0, 0);
  // cv::imwrite("resized_img.jpg", mat_tmp);
  auto ret = ImageProc::BGRToNv12(mat_tmp, nv12_mat);
  if (ret) {
    RCLCPP_ERROR(rclcpp::get_logger("image_proc"), "get nv12 image failed ");
    return nullptr;
  }

  auto *y = new hbSysMem;
  auto *uv = new hbSysMem;

  auto w_stride = ALIGN_16(scaled_img_width);
  hbSysAllocCachedMem(y, scaled_img_height * w_stride);
  hbSysAllocCachedMem(uv, scaled_img_height / 2 * w_stride);

  uint8_t *data = nv12_mat.data;
  auto *hb_y_addr = reinterpret_cast<uint8_t *>(y->virAddr);
  auto *hb_uv_addr = reinterpret_cast<uint8_t *>(uv->virAddr);

  // padding y
  for (int h = 0; h < scaled_img_height; ++h) {
    auto *raw = hb_y_addr + h * w_stride;
    for (int w = 0; w < scaled_img_width; ++w) {
      *raw++ = *data++;
    }
  }

  // padding uv
  auto uv_data = nv12_mat.data + scaled_img_height * scaled_img_width;
  for (int32_t h = 0; h < scaled_img_height / 2; ++h) {
    auto *raw = hb_uv_addr + h * w_stride;
    for (int32_t w = 0; w < scaled_img_width; ++w) {
      *raw++ = *uv_data++;
    }
  }

  hbSysFlushMem(y, HB_SYS_MEM_CACHE_CLEAN);
  hbSysFlushMem(uv, HB_SYS_MEM_CACHE_CLEAN);
  auto pyramid = new NV12PyramidInput;
  pyramid->width = scaled_img_width;
  pyramid->height = scaled_img_height;
  pyramid->y_vir_addr = y->virAddr;
  pyramid->y_phy_addr = y->phyAddr;
  pyramid->y_stride = w_stride;
  pyramid->uv_vir_addr = uv->virAddr;
  pyramid->uv_phy_addr = uv->phyAddr;
  pyramid->uv_stride = w_stride;
  return std::shared_ptr<NV12PyramidInput>(pyramid,
                                           [y, uv](NV12PyramidInput *pyramid) {
                                             // Release memory after deletion
                                             hbSysFreeMem(y);
                                             hbSysFreeMem(uv);
                                             delete y;
                                             delete uv;
                                             delete pyramid;
                                           });
}

std::shared_ptr<NV12PyramidInput> ImageProc::GetNV12PyramidFromBGR(
    const std::string &image_file,
    int scaled_img_height,
    int scaled_img_width) {
  cv::Mat nv12_mat;
  cv::Mat bgr_mat = cv::imread(image_file, cv::IMREAD_COLOR);
  int original_img_width = bgr_mat.cols;
  int original_img_height = bgr_mat.rows;

  auto w_stride = ALIGN_16(scaled_img_width);
  cv::Mat pad_frame;
  if (static_cast<uint32_t>(original_img_width) != w_stride ||
      original_img_height != scaled_img_height) {
    pad_frame =
        cv::Mat(scaled_img_height, w_stride, CV_8UC3, cv::Scalar::all(0));
    if (static_cast<uint32_t>(original_img_width) > w_stride ||
        original_img_height > scaled_img_height) {
      float ratio_w =
          static_cast<float>(original_img_width) / static_cast<float>(w_stride);
      float ratio_h = static_cast<float>(original_img_height) /
                      static_cast<float>(scaled_img_height);
      float dst_ratio = std::max(ratio_w, ratio_h);
      uint32_t resized_width =
          static_cast<float>(original_img_width) / dst_ratio;
      uint32_t resized_height =
          static_cast<float>(original_img_height) / dst_ratio;
      cv::resize(bgr_mat, bgr_mat, cv::Size(resized_width, resized_height));      
    }

    // 复制到目标图像中间
    bgr_mat.copyTo(pad_frame(cv::Rect((w_stride - bgr_mat.cols) / 2,
                                      (scaled_img_height - bgr_mat.rows) / 2,
                                      bgr_mat.cols,
                                      bgr_mat.rows)));
  } else {
    pad_frame = bgr_mat;
  }
  // cv::imwrite("resized_img.jpg", pad_frame);
  auto ret = ImageProc::BGRToNv12(pad_frame, nv12_mat);
  if (ret) {
    RCLCPP_ERROR(rclcpp::get_logger("image_proc"), "get nv12 image from bgr failed ");
    return nullptr;
  }
  original_img_height = bgr_mat.rows;
  original_img_width = bgr_mat.cols;

  auto *y = new hbSysMem;
  auto *uv = new hbSysMem;

  hbSysAllocCachedMem(y, scaled_img_height * w_stride);
  hbSysAllocCachedMem(uv, scaled_img_height / 2 * w_stride);

  uint8_t *data = nv12_mat.data;
  auto *hb_y_addr = reinterpret_cast<uint8_t *>(y->virAddr);
  auto *hb_uv_addr = reinterpret_cast<uint8_t *>(uv->virAddr);

  // padding y
  for (int h = 0; h < scaled_img_height; ++h) {
    auto *raw = hb_y_addr + h * w_stride;
    for (uint32_t w = 0; w < w_stride; ++w) {
      *raw++ = *data++;
    }
  }

  // padding uv
  auto uv_data = nv12_mat.data + scaled_img_height * w_stride;
  for (int32_t h = 0; h < scaled_img_height / 2; ++h) {
    auto *raw = hb_uv_addr + h * w_stride;
    for (uint32_t w = 0; w < w_stride; ++w) {
      *raw++ = *uv_data++;
    }
  }

  hbSysFlushMem(y, HB_SYS_MEM_CACHE_CLEAN);
  hbSysFlushMem(uv, HB_SYS_MEM_CACHE_CLEAN);
  auto pyramid = new NV12PyramidInput;
  pyramid->width = w_stride;
  pyramid->height = scaled_img_height;
  pyramid->y_vir_addr = y->virAddr;
  pyramid->y_phy_addr = y->phyAddr;
  pyramid->y_stride = w_stride;
  pyramid->uv_vir_addr = uv->virAddr;
  pyramid->uv_phy_addr = uv->phyAddr;
  pyramid->uv_stride = w_stride;
  return std::shared_ptr<NV12PyramidInput>(pyramid,
                                           [y, uv](NV12PyramidInput *pyramid) {
                                             // Release memory after deletion
                                             hbSysFreeMem(y);
                                             hbSysFreeMem(uv);
                                             delete y;
                                             delete uv;
                                             delete pyramid;
                                           });
}

std::shared_ptr<DNNTensor> ImageProc::GetNV12TensorFromNV12(const std::string &image_file,
                                                      int scaled_img_height,
                                                      int scaled_img_width) {
  cv::Mat nv12_mat;
  cv::Mat bgr_mat = cv::imread(image_file, cv::IMREAD_COLOR);
  cv::Mat mat_tmp;
  mat_tmp.create(scaled_img_height, scaled_img_width, bgr_mat.type());
  cv::resize(bgr_mat, mat_tmp, mat_tmp.size());
  auto ret = ImageProc::BGRToNv12(mat_tmp, nv12_mat);
  if (ret) {
    RCLCPP_ERROR(rclcpp::get_logger("image_proc"), "get nv12 image failed ");
    return nullptr;
  }
  int original_img_height = bgr_mat.rows;
  int original_img_width = bgr_mat.cols;

  auto *y = new hbSysMem;
  auto *uv = new hbSysMem;

  auto w_stride = ALIGN_16(scaled_img_width);
  hbSysAllocCachedMem(y, scaled_img_height * w_stride);
  hbSysAllocCachedMem(uv, scaled_img_height / 2 * w_stride);

  uint8_t *data = nv12_mat.data;
  auto *hb_y_addr = reinterpret_cast<uint8_t *>(y->virAddr);
  auto *hb_uv_addr = reinterpret_cast<uint8_t *>(uv->virAddr);

  // padding y
  for (int h = 0; h < scaled_img_height; ++h) {
    auto *raw = hb_y_addr + h * w_stride;
    for (int w = 0; w < scaled_img_width; ++w) {
      *raw++ = *data++;
    }
  }

  // padding uv
  auto uv_data = nv12_mat.data + scaled_img_height * scaled_img_width;
  for (int32_t h = 0; h < scaled_img_height / 2; ++h) {
    auto *raw = hb_uv_addr + h * w_stride;
    for (int32_t w = 0; w < scaled_img_width; ++w) {
      *raw++ = *uv_data++;
    }
  }

  hbSysFlushMem(y, HB_SYS_MEM_CACHE_CLEAN);
  hbSysFlushMem(uv, HB_SYS_MEM_CACHE_CLEAN);
  auto input_tensor = new DNNTensor;
  input_tensor->sysMem[0].virAddr = reinterpret_cast<void *>(y->virAddr);
  input_tensor->sysMem[0].phyAddr = y->phyAddr;
  input_tensor->sysMem[0].memSize = scaled_img_height * scaled_img_width;
  input_tensor->sysMem[1].virAddr = reinterpret_cast<void *>(uv->virAddr);
  input_tensor->sysMem[1].phyAddr = uv->phyAddr;
  input_tensor->sysMem[1].memSize = scaled_img_height * scaled_img_width / 2;
  return std::shared_ptr<DNNTensor>(
      input_tensor, [y, uv](DNNTensor *input_tensor) {
        // Release memory after deletion
        hbSysFreeMem(y);
        hbSysFreeMem(uv);
        delete y;
        delete uv;
        delete input_tensor;
      });
}

std::shared_ptr<DNNTensor> ImageProc::GetBGRTensorFromBGR(const std::string &image_file,
                                                      int scaled_img_height,
                                                      int scaled_img_width,
                                                      hbDNNTensorProperties &tensor_properties,
                                                      float &ratio,
                                                      ImageType image_type,
                                                      bool is_pad,
                                                      bool is_center_crop,
                                                      bool is_scale) {
  auto w_stride = ALIGN_16(scaled_img_width);
  int channel = 3;

  cv::Mat bgr_mat = cv::imread(image_file, cv::IMREAD_COLOR);
  int original_img_width = bgr_mat.cols;
  int original_img_height = bgr_mat.rows;

  cv::Mat pad_frame;
  if (is_pad) {
    if (static_cast<uint32_t>(original_img_width) != w_stride ||
        original_img_height != scaled_img_height) {
      pad_frame =
          cv::Mat(scaled_img_height, w_stride, CV_8UC3, cv::Scalar::all(0));
      if (static_cast<uint32_t>(original_img_width) > w_stride ||
          original_img_height > scaled_img_height) {
        float ratio_w =
            static_cast<float>(original_img_width) / static_cast<float>(w_stride);
        float ratio_h = static_cast<float>(original_img_height) /
                        static_cast<float>(scaled_img_height);
        float dst_ratio = std::max(ratio_w, ratio_h);
        ratio = dst_ratio;
        uint32_t resized_width =
            static_cast<float>(original_img_width) / dst_ratio;
        uint32_t resized_height =
            static_cast<float>(original_img_height) / dst_ratio;
        cv::resize(bgr_mat, bgr_mat, cv::Size(resized_width, resized_height));
      }

      if (is_center_crop) {
        // 复制到目标图像中间
        bgr_mat.copyTo(pad_frame(
          cv::Rect((w_stride - bgr_mat.cols) / 2,
                    (scaled_img_height - bgr_mat.rows) / 2,
                    bgr_mat.cols,
                    bgr_mat.rows)));
      } else {
        // 复制到目标图像到起点
        bgr_mat.copyTo(pad_frame(
          cv::Rect(0, 0, bgr_mat.cols, bgr_mat.rows)));
      }
    } else {
      pad_frame = bgr_mat;
    }
  } else {
    cv::resize(bgr_mat, pad_frame, cv::Size(w_stride, scaled_img_height));
  }

  cv::Mat mat_tmp;
  int src_elem_size = 1;
  switch (tensor_properties.tensorType)
  {
    case HB_DNN_TENSOR_TYPE_U8: src_elem_size = 1; mat_tmp = pad_frame; break;
    case HB_DNN_TENSOR_TYPE_F32: {
      src_elem_size = 4; 
      pad_frame.convertTo(mat_tmp, CV_32F); 
      if (is_scale) {
        mat_tmp /= 255.0;
      }
    } break;
    default: RCLCPP_ERROR(rclcpp::get_logger("image_proc"), 
          "Tensor Type %d is not support", tensor_properties.tensorType); break;  
  }

  if (image_type == ImageType::RGB) {
    cv::cvtColor(mat_tmp, mat_tmp, cv::COLOR_BGR2RGB);
  }

  auto *mem = new hbSysMem;
  hbSysAllocCachedMem(mem, scaled_img_height * w_stride * channel * src_elem_size);
  uint8_t *data = mat_tmp.data;
  auto *hb_mem_addr = reinterpret_cast<uint8_t *>(mem->virAddr);

  if (tensor_properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
    for (int h = 0; h < scaled_img_height; ++h) {
      for (int w = 0; w < scaled_img_width; ++w) {
        for (int c = 0; c < channel; ++c) {
          auto *raw = hb_mem_addr + c * scaled_img_height * w_stride * src_elem_size + h * w_stride * src_elem_size + w * src_elem_size;
          auto *src = data + h * scaled_img_width * channel * src_elem_size + w * channel * src_elem_size + c * src_elem_size;
          memcpy(raw, src, src_elem_size);
        }
      }
    }
  } else {
    for (int h = 0; h < scaled_img_height; ++h) {
      auto *raw = hb_mem_addr + h * w_stride * channel * src_elem_size;
      auto *src = data + h * scaled_img_width * channel * src_elem_size;
      memcpy(raw, src, scaled_img_width * channel * src_elem_size);
    }
  }

  hbSysFlushMem(mem, HB_SYS_MEM_CACHE_CLEAN);
  auto input_tensor = new DNNTensor;
  input_tensor->properties = tensor_properties;
  input_tensor->sysMem[0].virAddr = reinterpret_cast<void *>(mem->virAddr);
  input_tensor->sysMem[0].phyAddr = mem->phyAddr;
  input_tensor->sysMem[0].memSize = scaled_img_height * scaled_img_width * channel * src_elem_size;

  return std::shared_ptr<DNNTensor>(
      input_tensor, [mem](DNNTensor *input_tensor) {
        // Release memory after deletion
        hbSysFreeMem(mem);
        delete mem;
        delete input_tensor;
      });
}

std::shared_ptr<DNNTensor> ImageProc::GetBGRTensorFromBGRImg(
                                                    const cv::Mat &bgr_mat_tmp, 
                                                    int scaled_img_height, 
                                                    int scaled_img_width,
                                                    hbDNNTensorProperties &tensor_properties,
                                                    float &ratio,
                                                    ImageType image_type) {
  cv::Mat bgr_mat;
  bgr_mat_tmp.copyTo(bgr_mat);
  auto w_stride = ALIGN_16(scaled_img_width);
  int channel = 3;
  int original_img_width = bgr_mat.cols;
  int original_img_height = bgr_mat.rows;

  cv::Mat pad_frame;
  if (static_cast<uint32_t>(original_img_width) != w_stride ||
      original_img_height != scaled_img_height) {
    pad_frame =
        cv::Mat(scaled_img_height, w_stride, CV_8UC3, cv::Scalar::all(0));
    if (static_cast<uint32_t>(original_img_width) > w_stride ||
        original_img_height > scaled_img_height) {
      float ratio_w =
          static_cast<float>(original_img_width) / static_cast<float>(w_stride);
      float ratio_h = static_cast<float>(original_img_height) /
                      static_cast<float>(scaled_img_height);
      float dst_ratio = std::max(ratio_w, ratio_h);
      ratio = dst_ratio;
      uint32_t resized_width =
          static_cast<float>(original_img_width) / dst_ratio;
      uint32_t resized_height =
          static_cast<float>(original_img_height) / dst_ratio;
      cv::resize(bgr_mat, bgr_mat, cv::Size(resized_width, resized_height));
    }
    // 复制到目标图像到起点
    bgr_mat.copyTo(pad_frame(
      cv::Rect(0, 0, bgr_mat.cols, bgr_mat.rows)));
  } else {
    pad_frame = bgr_mat;
  }

  cv::Mat mat_tmp;
  int src_elem_size = 1;
  switch (tensor_properties.tensorType)
  {
    case HB_DNN_TENSOR_TYPE_U8: src_elem_size = 1; mat_tmp = pad_frame; break;
    case HB_DNN_TENSOR_TYPE_F32: {
      src_elem_size = 4; 
      pad_frame.convertTo(mat_tmp, CV_32F); 
    } break;
    default: RCLCPP_ERROR(rclcpp::get_logger("image_proc"), 
          "Tensor Type %d is not support", tensor_properties.tensorType); break;  
  }

  if (image_type == ImageType::RGB) {
    cv::cvtColor(mat_tmp, mat_tmp, cv::COLOR_BGR2RGB);
  }

  auto *mem = new hbSysMem;
  hbSysAllocCachedMem(mem, scaled_img_height * w_stride * channel * src_elem_size);
  uint8_t *data = mat_tmp.data;
  auto *hb_mem_addr = reinterpret_cast<uint8_t *>(mem->virAddr);

  if (tensor_properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
    for (int h = 0; h < scaled_img_height; ++h) {
      for (int w = 0; w < scaled_img_width; ++w) {
        for (int c = 0; c < channel; ++c) {
          auto *raw = hb_mem_addr + c * scaled_img_height * w_stride * src_elem_size + h * w_stride * src_elem_size + w * src_elem_size;
          auto *src = data + h * scaled_img_width * channel * src_elem_size + w * channel * src_elem_size + c * src_elem_size;
          memcpy(raw, src, src_elem_size);
        }
      }
    }
  } else {
    for (int h = 0; h < scaled_img_height; ++h) {
      auto *raw = hb_mem_addr + h * w_stride * channel * src_elem_size;
      auto *src = data + h * scaled_img_width * channel * src_elem_size;
      memcpy(raw, src, w_stride * channel * src_elem_size);
    }
  }

  hbSysFlushMem(mem, HB_SYS_MEM_CACHE_CLEAN);
  auto input_tensor = new DNNTensor;
  input_tensor->properties = tensor_properties;
  input_tensor->sysMem[0].virAddr = reinterpret_cast<void *>(mem->virAddr);
  input_tensor->sysMem[0].phyAddr = mem->phyAddr;
  input_tensor->sysMem[0].memSize = scaled_img_height * scaled_img_width * channel * src_elem_size;

  return std::shared_ptr<DNNTensor>(
      input_tensor, [mem](DNNTensor *input_tensor) {
        // Release memory after deletion
        hbSysFreeMem(mem);
        delete mem;
        delete input_tensor;
      });
}


std::shared_ptr<DNNTensor> ImageProc::GetBGRTensorFromBGRImg(
    const char *in_img_data,
    const int &in_img_height,
    const int &in_img_width,
    const int &scaled_img_height,
    const int &scaled_img_width,
    hbDNNTensorProperties &tensor_properties) {
  auto w_stride = ALIGN_16(scaled_img_width);

  int src_elem_size = 1;
  switch (tensor_properties.tensorType)
  {
    case HB_DNN_TENSOR_TYPE_S8:
    case HB_DNN_TENSOR_TYPE_U8: src_elem_size = 1; break;
    case HB_DNN_TENSOR_TYPE_F16:
    case HB_DNN_TENSOR_TYPE_S16: 
    case HB_DNN_TENSOR_TYPE_U16: src_elem_size = 2; break;
    case HB_DNN_TENSOR_TYPE_F32:
    case HB_DNN_TENSOR_TYPE_S32:
    case HB_DNN_TENSOR_TYPE_U32: src_elem_size = 4; break;
    case HB_DNN_TENSOR_TYPE_F64:
    case HB_DNN_TENSOR_TYPE_S64: 
    case HB_DNN_TENSOR_TYPE_U64: src_elem_size = 8; break;
    default: RCLCPP_ERROR(rclcpp::get_logger("image_proc"), 
        "Tensor Type %d is not support", tensor_properties.tensorType); break;
  }

  auto *mem = new hbSysMem;
  hbSysAllocCachedMem(mem, scaled_img_height * w_stride * 3 * src_elem_size);
  //内存初始化
  memset(mem->virAddr, 0, scaled_img_height * w_stride * 3 * src_elem_size);

  const uint8_t *data = reinterpret_cast<const uint8_t *>(in_img_data);
  auto *hb_mem_addr = reinterpret_cast<uint8_t *>(mem->virAddr);
  int copy_w = std::min(in_img_width, scaled_img_width);
  int copy_h = std::min(in_img_height, scaled_img_height);

  // padding mem
  for (int h = 0; h < copy_h; ++h) {
    auto *raw = hb_mem_addr + h * w_stride * 3 * src_elem_size;
    auto *src = data + h * in_img_width * 3 * src_elem_size;
    memcpy(raw, src, copy_w * 3 * src_elem_size);
  }

  hbSysFlushMem(mem, HB_SYS_MEM_CACHE_CLEAN);
  auto input_tensor = new DNNTensor;

  input_tensor->properties = tensor_properties;
  input_tensor->sysMem[0].virAddr = reinterpret_cast<void *>(mem->virAddr);
  input_tensor->sysMem[0].phyAddr = mem->phyAddr;
  input_tensor->sysMem[0].memSize = scaled_img_height * scaled_img_width * 3 * src_elem_size;
  return std::shared_ptr<DNNTensor>(
      input_tensor, [mem](DNNTensor *input_tensor) {
        // Release memory after deletion
        hbSysFreeMem(mem);
        delete mem;
        delete input_tensor;
      });
}

int32_t ImageProc::BGRToNv12(cv::Mat &bgr_mat, cv::Mat &img_nv12) {
  auto height = bgr_mat.rows;
  auto width = bgr_mat.cols;

  if (height % 2 || width % 2) {
    std::cerr << "input img height and width must aligned by 2!";
    return -1;
  }
  cv::Mat yuv_mat;
  cv::cvtColor(bgr_mat, yuv_mat, cv::COLOR_BGR2YUV_I420);
  if (yuv_mat.data == nullptr) {
    std::cerr << "yuv_mat.data is null pointer" << std::endl;
    return -1;
  }

  auto *yuv = yuv_mat.ptr<uint8_t>();
  if (yuv == nullptr) {
    std::cerr << "yuv is null pointer" << std::endl;
    return -1;
  }
  img_nv12 = cv::Mat(height * 3 / 2, width, CV_8UC1);
  auto *ynv12 = img_nv12.ptr<uint8_t>();

  int32_t uv_height = height / 2;
  int32_t uv_width = width / 2;

  // copy y data
  int32_t y_size = height * width;
  memcpy(ynv12, yuv, y_size);

  // copy uv data
  uint8_t *nv12 = ynv12 + y_size;
  uint8_t *u_data = yuv + y_size;
  uint8_t *v_data = u_data + uv_height * uv_width;

  for (int32_t i = 0; i < uv_width * uv_height; i++) {
    *nv12++ = *u_data++;
    *nv12++ = *v_data++;
  }
  return 0;
}

int32_t ImageProc::Nv12ToBGR(const char *in_img_data, const int &in_img_height, const int &in_img_width, cv::Mat &bgr_mat) {
  cv::Mat Nv12_image(in_img_height + in_img_height / 2, in_img_width, CV_8UC1, (void*)in_img_data);

  cv::cvtColor(Nv12_image, bgr_mat, cv::COLOR_YUV2BGR_NV12);
  return 0;
}

}  // namespace dnn_node
}  // namespace hobot
