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

#ifndef _INPUT_IMAGE_LIST_ITERATOR_H_
#define _INPUT_IMAGE_LIST_ITERATOR_H_

#include <string>
#include <vector>

#include "data_iterator.h"

class ImageListDataIterator : public DataIterator {
 public:
  ImageListDataIterator() : DataIterator("image_list_data_iterator") {}

  /**
   * Init Image Data iterator from file
   * @param[in] config_file: config file
   *        the file content should be in the json format
   *        for example:
   *        {
   *            "image_list_file" : "image.list" #  one image file per line
   *            "width": 960,
   *            "height": 544,
   *            "data_type": 1
   *        }
   *
   * default
   * @param[in] config_string: config string
   *        same as config_file
   * @return 0 if success
   */
  int Init(std::string config_file,
           std::string config_string) override;

  /**
   * Next Image Data read from file system
   * @param[out] NV12PyramidInputPtr: NV12PyramidInput
   * @return TRUE if success
   */
  bool Next(NV12PyramidInputPtr &pyramid) override;

  /**
   * Check if has next image
   * @return 0 if finish
   */
  bool HasNext() override;

  void set_model_input_width_height(int model_input_w,
                                    int model_input_h) override;

  ~ImageListDataIterator() override;

 private:
  int LoadConfig(std::string &config_string) override;
  NV12PyramidInputPtr PraseLocalImageToMemory(int image_type,
      std::string imagefilepath);

 private:
  std::string image_list_file_;
  std::vector<std::string> image_files_;
  bool cache_able_ = false;
  bool loop_able_ = true;
  size_t max_cache_size_ = 10;
  int max_frame_count_ = INT32_MAX;
  size_t send_index_ = 0;
  int data_type_ = 0;
  std::vector<NV12PyramidInputPtr> cache_;
};

#endif  // _INPUT_IMAGE_LIST_ITERATOR_H_
