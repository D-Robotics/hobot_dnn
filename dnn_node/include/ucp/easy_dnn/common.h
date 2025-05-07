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

#ifndef _COMMON_H_
#define _COMMON_H_

#include <sstream>
#include <utility>

// statement may involves function calls
#define RETURN_IF_FAILED(statement)   \
  {                                   \
    int32_t const code_{(statement)}; \
    if ((code_) != HB_DNN_SUCCESS) {     \
      return code_;                   \
    }                                 \
  }

#endif  // _COMMON_H_
