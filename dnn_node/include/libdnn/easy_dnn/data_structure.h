// Copyright (c) [2024] [D-Robotics].
//
// You can use this software according to the terms and conditions of
// the Apache v2.0.
// You may obtain a copy of Apache v2.0. at:
//
//     http: //www.apache.org/licenses/LICENSE-2.0
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See Apache v2.0 for more details.

#ifndef _EASY_DNN_DATA_STRUCTURE_H_
#define _EASY_DNN_DATA_STRUCTURE_H_

#include <climits>
#include <memory>
#include <ostream>
#include <string>

#include "dnn/hb_dnn.h"

namespace hobot {
namespace easy_dnn {

class DNNInput {
 public:
  virtual void Reset() {}
  virtual ~DNNInput() = default;
};

class NV12PyramidInput : public DNNInput {
 public:
  uint64_t y_phy_addr;
  void *y_vir_addr;
  uint64_t uv_phy_addr;
  void *uv_vir_addr;
  int32_t height;
  int32_t width;
  int32_t y_stride;
  int32_t uv_stride;
  void Reset() override {}
};

class DNNTensor : public hbDNNTensor {
 public:
  DNNTensor() : hbDNNTensor() {}
  DNNTensor(const hbDNNTensor& other) : hbDNNTensor(other) {}

  virtual void Reset() {}
  virtual ~DNNTensor() = default;

  int CACHE_INVALIDATE() {return hbSysFlushMem(&(sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);}

  int CACHE_CLEAN() {return hbSysFlushMem(&(sysMem[0]), HB_SYS_MEM_CACHE_CLEAN);}

  template <typename T>
  T* GetTensorData() {
    return reinterpret_cast<T*>(sysMem[0].virAddr);
  }
};

}  // namespace easy_dnn
}  // namespace hobot

#endif  // _EASY_DNN_DATA_STRUCTURE_H_
