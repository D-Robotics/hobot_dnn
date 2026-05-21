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

#ifndef _OUTPUT_PARSER_NEON_UTILS_H_
#define _OUTPUT_PARSER_NEON_UTILS_H_

#include <cstdint>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace hobot {
namespace dnn_node {
namespace output_parser {

// NEON-accelerated dot product: sum(a[i] * b[i]) for i in [0, n)
// Typically n = 32 (num_mask), called per pixel in mask generation.
inline float neon_dot_product_f32(const float* a, const float* b, int n) {
#ifdef __ARM_NEON
  float32x4_t sum0 = vdupq_n_f32(0.0f);
  float32x4_t sum1 = vdupq_n_f32(0.0f);
  int i = 0;
  // Unroll by 8: process two 4-float chunks per iteration
  for (; i + 7 < n; i += 8) {
    float32x4_t va0 = vld1q_f32(a + i);
    float32x4_t vb0 = vld1q_f32(b + i);
    sum0 = vmlaq_f32(sum0, va0, vb0);
    float32x4_t va1 = vld1q_f32(a + i + 4);
    float32x4_t vb1 = vld1q_f32(b + i + 4);
    sum1 = vmlaq_f32(sum1, va1, vb1);
  }
  sum0 = vaddq_f32(sum0, sum1);
  float32x2_t sum2 = vadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
  float result = vget_lane_f32(vpadd_f32(sum2, sum2), 0);
  for (; i < n; ++i) {
    result += a[i] * b[i];
  }
  return result;
#else
  float sum = 0.0f;
  for (int i = 0; i < n; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
#endif
}

// NEON-accelerated dequantization: dst[i] = src[i] * scale for i in [0, n)
inline void neon_dequant_i16_to_f32(float* dst, const int16_t* src,
                                     float scale, int n) {
#ifdef __ARM_NEON
  int i = 0;
  float32x4_t vscale = vdupq_n_f32(scale);
  for (; i + 7 < n; i += 8) {
    int16x8_t vi = vld1q_s16(src + i);
    int32x4_t vi32_low = vmovl_s16(vget_low_s16(vi));
    int32x4_t vi32_high = vmovl_s16(vget_high_s16(vi));
    float32x4_t vf_low = vcvtq_f32_s32(vi32_low);
    float32x4_t vf_high = vcvtq_f32_s32(vi32_high);
    vst1q_f32(dst + i, vmulq_f32(vf_low, vscale));
    vst1q_f32(dst + i + 4, vmulq_f32(vf_high, vscale));
  }
  for (; i < n; ++i) {
    dst[i] = static_cast<float>(src[i]) * scale;
  }
#else
  for (int i = 0; i < n; ++i) {
    dst[i] = static_cast<float>(src[i]) * scale;
  }
#endif
}

// NEON-accelerated int32 to float with per-channel scale
inline void neon_dequant_s32c_to_f32(float* dst, const int32_t* src,
                                      const float* scales, int n) {
#ifdef __ARM_NEON
  int i = 0;
  for (; i + 3 < n; i += 4) {
    int32x4_t vi = vld1q_s32(src + i);
    float32x4_t vscale = vld1q_f32(scales + i);
    float32x4_t vf = vcvtq_f32_s32(vi);
    vst1q_f32(dst + i, vmulq_f32(vf, vscale));
  }
  for (; i < n; ++i) {
    dst[i] = static_cast<float>(src[i]) * scales[i];
  }
#else
  for (int i = 0; i < n; ++i) {
    dst[i] = static_cast<float>(src[i]) * scales[i];
  }
#endif
}

}  // namespace output_parser
}  // namespace dnn_node
}  // namespace hobot

#endif  // _OUTPUT_PARSER_NEON_UTILS_H_
