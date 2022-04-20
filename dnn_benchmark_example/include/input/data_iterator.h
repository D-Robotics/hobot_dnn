// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _INPUT_DATA_ITERATOR_H_
#define _INPUT_DATA_ITERATOR_H_

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "rclcpp/rclcpp.hpp"
#include "input/input_data.h"

class DataIterator {
 public:
  explicit DataIterator(std::string module_name)
      : module_name_(std::move(module_name)) {}

  /**
   * Init Data iterator from file
   * @param[in] config_file: config file path
   *            the config file should be in json format
   * @param[in] config_string: config string
   *            same as config_file
   * @return 0 if success
   */
  virtual int Init(std::string config_file,
                   std::string config_string);

  /**
   * Next Data
   * @param[out] NV12PyramidInputPtr: NV12PyramidInput
   * @return TRUE if success
   */
  virtual bool Next(NV12PyramidInputPtr &pyramid) = 0;

  /**
   * Check if has next input data
   * @return 0 if has next input data
   */
  virtual bool HasNext() = 0;

  /**
   * Get next frame id
   * @return next frame id
   */
  virtual int NextFrameId() { return ++last_frame_id; }

  /**
   * Get DataIterator Implementation instance
   * @param[in]: module_name
   * @return DataIterator implementation instance
   */
  static DataIterator *GetImpl(const std::string &module_name);

  virtual ~DataIterator() = default;

  void set_model_input_width_height(int model_input_w,
                                    int model_input_h)
  {
    this->model_input_width_ = model_input_w;
    this->model_input_height_ = model_input_h;
  }

 protected:
  virtual int LoadConfig(std::string &config_string)
  {
    RCLCPP_INFO(rclcpp::get_logger("example"),
        " %s ", config_string.c_str());
    return 0;
  }
  int model_input_width_ = 960;
  int model_input_height_ = 544;

 private:
  int LoadConfigFile(std::string &config_file);

 private:
  std::string module_name_;

 protected:
  bool is_finish_ = false;
  int last_frame_id = -1;
};

#define DEFINE_AND_REGISTER_DATA_ITERATOR(iterator_name, class_name)       \
  DataIterator *iterator_name##_input_creator() { return new class_name; } \
  static DataIteratorRegistry process_registry(#iterator_name,             \
                                               iterator_name##_input_creator);

typedef DataIterator *(*inputCreator)();

class DataIteratorFactory {
 public:
  static DataIteratorFactory *GetInstance() {
    static DataIteratorFactory ins;
    return &ins;
  }

  DataIterator *GetDataIterator(const char *data_iterator_name)
  {
    if (input_process_registry_.find(data_iterator_name) ==
        input_process_registry_.end())
    {
      RCLCPP_ERROR(rclcpp::get_logger("example"),
        "process %s has not been registered.", data_iterator_name);
      return nullptr;
    }
    return input_process_registry_[data_iterator_name]();
  }

  int32_t InputRegister(const char *data_iterator_name, inputCreator func)
  {
    if (input_process_registry_.find(data_iterator_name) !=
        input_process_registry_.end())
    {
      RCLCPP_DEBUG(rclcpp::get_logger("example"),
        "process %s has been registered.", data_iterator_name);
    }
    input_process_registry_[data_iterator_name] = func;
    return 0;
  }
  std::unordered_map<std::string, inputCreator> input_process_registry_;
};

class DataIteratorRegistry {
 public:
  explicit DataIteratorRegistry(const char *data_iterator_name,
                                inputCreator func) noexcept {
    DataIteratorFactory::GetInstance()->InputRegister(data_iterator_name, func);
  }
};

#endif  // _INPUT_DATA_ITERATOR_H_
