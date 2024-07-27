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

#ifndef _UTILS_UTILS_H_
#define _UTILS_UTILS_H_

#include <string>
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "rapidjson/writer.h"

#define RED_COMMENT_START "\033[31m "
#define RED_COMMENT_END " \033[0m"
#define YELLOW_COMMENT_START "\033[33m "
#define YELLOW_COMMENT_END "\033[0m"

/**
 * Dump rapidjson to string
 * @param[in] rapidjson::Value val
 * @return json string
 */
std::string json_to_string(rapidjson::Value &val);

uint64_t currentMicroseconds();

#endif  // _UTILS_UTILS_H_
