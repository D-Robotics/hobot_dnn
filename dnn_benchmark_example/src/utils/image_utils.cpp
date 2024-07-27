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

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>

#include "include/utils/image_utils.h"
#include "dnn/hb_sys.h"

std::shared_ptr<NV12PyramidInput> ImageUtils::GetNV12Pyramid(
    const std::string &image_file, ImageType image_type,
    int scaled_img_height, int scaled_img_width) {
  if (ImageType::BGR == image_type) {
    int original_img_height = 0, original_img_width = 0;
    return GetNV12Pyramid(image_file,
                          scaled_img_height,
                          scaled_img_width,
                          original_img_height,
                          original_img_width);
  } else if (ImageType::NV12 == image_type) {
    std::shared_ptr<NV12PyramidInput> pyramid = nullptr;
    std::ifstream ifs(image_file, std::ios::in | std::ios::binary);
    if (!ifs)
    {
      std::cout << "open image_file failed " << std::endl;
      return pyramid;
    }
    ifs.seekg(0, std::ios::end);
    int len = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    char *data = new char[len];
    ifs.read(data, len);

    int y_img_len = len / 3 * 2;
    int uv_img_len = len / 3;

    auto *y = new hbSysMem;
    auto *uv = new hbSysMem;

    auto w_stride = ALIGN_16(scaled_img_width);
    hbSysAllocCachedMem(y, scaled_img_height * w_stride);
    hbSysAllocCachedMem(uv, scaled_img_height / 2 * w_stride);

    memcpy(reinterpret_cast<uint8_t *>(y->virAddr), data,
          y_img_len);
    memcpy(reinterpret_cast<uint8_t *>(uv->virAddr),
          data + y_img_len, uv_img_len);

    hbSysFlushMem(y, HB_SYS_MEM_CACHE_CLEAN);
    hbSysFlushMem(uv, HB_SYS_MEM_CACHE_CLEAN);
    auto pym_in = new NV12PyramidInput;
    pym_in->width = scaled_img_width;
    pym_in->height = scaled_img_height;
    pym_in->y_vir_addr = y->virAddr;
    pym_in->y_phy_addr = y->phyAddr;
    pym_in->y_stride = w_stride;
    pym_in->uv_vir_addr = uv->virAddr;
    pym_in->uv_phy_addr = uv->phyAddr;
    pym_in->uv_stride = w_stride;
    pyramid = std::shared_ptr<NV12PyramidInput>(
        pym_in, [y, uv](NV12PyramidInput *pym_in) {
          // Release memory after deletion
          hbSysFreeMem(y);
          hbSysFreeMem(uv);
          delete y;
          delete uv;
          delete pym_in;
        });
    return pyramid;
  } else {
    return nullptr;
  }
}

std::shared_ptr<NV12PyramidInput> ImageUtils::GetNV12Pyramid(
    const std::string &image_file,
    int scaled_img_height,
    int scaled_img_width,
    int &original_img_height,
    int &original_img_width) {
  cv::Mat nv12_mat;
  cv::Mat bgr_mat = cv::imread(image_file, cv::IMREAD_COLOR);
  cv::Mat mat_tmp;
  mat_tmp.create(scaled_img_height, scaled_img_width, bgr_mat.type());
  cv::resize(bgr_mat, mat_tmp, mat_tmp.size(), 0, 0);
  // cv::imwrite("resized_img.jpg", mat_tmp);
  auto ret = ImageUtils::BGRToNv12(mat_tmp, nv12_mat);
  if (ret) {
    std::cout << "get nv12 image failed " << std::endl;
    return nullptr;
  }
  original_img_height = bgr_mat.rows;
  original_img_width = bgr_mat.cols;

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
  return std::shared_ptr<NV12PyramidInput>(
      pyramid, [y, uv](NV12PyramidInput *pyramid) {
        // Release memory after deletion
        hbSysFreeMem(y);
        hbSysFreeMem(uv);
        delete y;
        delete uv;
        delete pyramid;
      });
}

std::shared_ptr<NV12PyramidInput> ImageUtils::GetNV12Pyramid(
    const cv::Mat &bgr_mat,
    int scaled_img_height,
    int scaled_img_width) {
  cv::Mat nv12_mat;
  cv::Mat mat_tmp;
  mat_tmp.create(scaled_img_height, scaled_img_width, bgr_mat.type());
  cv::resize(bgr_mat, mat_tmp, mat_tmp.size(), 0, 0);
  // cv::imwrite("resized_img.jpg", mat_tmp);
  auto ret = ImageUtils::BGRToNv12(mat_tmp, nv12_mat);
  if (ret) {
    std::cout << "get nv12 image failed " << std::endl;
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
  return std::shared_ptr<NV12PyramidInput>(
      pyramid, [y, uv](NV12PyramidInput *pyramid) {
        // Release memory after deletion
        hbSysFreeMem(y);
        hbSysFreeMem(uv);
        delete y;
        delete uv;
        delete pyramid;
      });
}

std::shared_ptr<NV12PyramidInput> ImageUtils::GetNV12PyramidFromNV12Img(
    const char* in_img_data,
    int in_img_height,
    int in_img_width,
    int scaled_img_height,
    int scaled_img_width) {
  auto *y = new hbSysMem;
  auto *uv = new hbSysMem;
  auto w_stride = ALIGN_16(scaled_img_width);
  hbSysAllocCachedMem(y, scaled_img_height * w_stride);
  hbSysAllocCachedMem(uv, scaled_img_height / 2 * w_stride);

  const uint8_t *data = reinterpret_cast<const uint8_t*>(in_img_data);
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
  return std::shared_ptr<NV12PyramidInput>(
      pyramid, [y, uv](NV12PyramidInput *pyramid) {
        // Release memory after deletion
        hbSysFreeMem(y);
        hbSysFreeMem(uv);
        delete y;
        delete uv;
        delete pyramid;
      });
}

int32_t ImageUtils::BGRToNv12(cv::Mat &bgr_mat, cv::Mat &img_nv12) {
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

void ImageUtils::GetNV12Tensor(std::string &image_file,
                               std::shared_ptr<DNNTensor> &dnn_tensor) {
  hbDNNTensorProperties properties = dnn_tensor->properties;
  cv::Mat bgr_mat = cv::imread(image_file, cv::IMREAD_COLOR);
  if (bgr_mat.empty()) {
    std::cout << "image file not exist!" << std::endl;
    return;
  }

  auto height = bgr_mat.rows;
  auto width = bgr_mat.cols;

  cv::Mat yuv_mat;
  cv::cvtColor(bgr_mat, yuv_mat, cv::COLOR_BGR2YUV_I420);

  auto *yuv = yuv_mat.ptr<uint8_t>();
  cv::Mat img_nv12 = cv::Mat(height * 3 / 2, width, CV_8UC1);
  auto *ynv12 = img_nv12.ptr<uint8_t>();

  int uv_height = height / 2;
  int uv_width = width / 2;

  // copy y data
  int y_size = height * width;
  memcpy(ynv12, yuv, y_size);

  // copy uv data
  uint8_t *nv12 = ynv12 + y_size;
  uint8_t *u_data = yuv + y_size;
  uint8_t *v_data = u_data + uv_height * uv_width;

  for (int i = 0; i < uv_width * uv_height; i++) {
    *nv12++ = *u_data++;
    *nv12++ = *v_data++;
  }

  uint8_t *data = img_nv12.data;
  auto stride = properties.alignedShape.dimensionSize[3];
  auto *y = reinterpret_cast<uint8_t *>(dnn_tensor->sysMem[0].virAddr);
  for (int h = 0; h < height; ++h) {
    auto *raw = y + h * stride;
    for (int w = 0; w < width; ++w) {
      *raw++ = *data++;
    }
  }
  // Copy uv data to data1
  auto *uv = reinterpret_cast<uint8_t *>(dnn_tensor->sysMem[1].virAddr);
  memcpy(uv, img_nv12.data + height * width, height * width / 2);
  hbSysFlushMem(&dnn_tensor->sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
  hbSysFlushMem(&dnn_tensor->sysMem[1], HB_SYS_MEM_CACHE_CLEAN);
}
