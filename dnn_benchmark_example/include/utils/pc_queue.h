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

#ifndef _UTILS_PC_QUEUE_H_
#define _UTILS_PC_QUEUE_H_

#include <condition_variable>
#include <mutex>
#include <queue>

template <typename T>
class PCQueue {
 public:
  explicit PCQueue(int capacity = INT32_MAX) : capacity_(capacity) {}

  void put(T t) {
    {
      std::unique_lock<std::mutex> lck(mutex_);
      cv_.wait(lck, [this] {
          return (static_cast<int>(q_.size())) <= capacity_;
        });
      q_.push(t);
    }
    cv_.notify_all();
  }

  bool get(T& t, int timeout = 0) {
    {
      std::unique_lock<std::mutex> lck(mutex_);
      if (timeout > 0) {
        cv_.wait_for(lck, std::chrono::milliseconds(timeout), [this] {
          return !this->q_.empty();
        });
      } else {
        cv_.wait(lck, [this] { return !q_.empty(); });
      }
      if (q_.empty()) {
        return false;
      } else {
        t = q_.front();
        q_.pop();
      }
    }
    cv_.notify_all();
    return true;
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  std::queue<T> q_;
  int capacity_;
};

#endif  // _UTILS_PC_QUEUE_H_
