// Copyright 2019 Bold Hearts
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

#include "v4l2_camera/v4l2_camera.hpp"

#include <sensor_msgs/image_encodings.hpp>

#include <sstream>
#include <stdexcept>
#include <string>
#include <memory>
#include <utility>
#include <vector>
#include <algorithm>

#include "v4l2_camera/fourcc.hpp"

#include "v4l2_camera/v4l2_camera_device.hpp"

#ifdef ENABLE_CUDA
#include <cuda.h>
#include <nppi_color_conversion.h>
#endif

using namespace std::chrono_literals;

namespace v4l2_camera
{
V4L2Camera::V4L2Camera(ros::NodeHandle node, ros::NodeHandle private_nh)
  : image_transport_(private_nh)
{
  private_nh.getParam("publish_rate", publish_rate_);
  private_nh.getParam("video_device", device);
  private_nh.getParam("use_v4l2_buffer_timestamps", use_v4l2_buffer_timestamps);
  private_nh.getParam("timestamp_offset", timestamp_offset);

  if(std::abs(publish_rate_) < std::numeric_limits<double>::epsilon()){
    ROS_WARN("Invalid publish_rate = 0. Use default value -1 instead");
    publish_rate_ = -1.0;
  }
  if(publish_rate_ > 0){
    const auto publish_period = ros::Duration(publish_rate_);
    image_pub_timer_ = nh_.createTimer(publish_period, [this](){this->publish_next_frame_=true;});
    publish_next_frame_ = false;
  }
  else{
    publish_next_frame_ = true;
  }
    camera_transport_pub_ = image_transport_.advertiseCamera("image_raw", 10);

  ros::Duration timestamp_offset_duration = ros::Duration::from_nanoseconds(timestamp_offset);

  camera_ = std::make_shared<V4l2CameraDevice>(device, use_v4l2_buffer_timestamps, timestamp_offset_duration);

  if (!camera_->open()) {
    return;
  }

  cinfo_ = std::make_shared<camera_info_manager::CameraInfoManager>(this, camera_->getCameraName());
#ifdef ENABLE_CUDA
  src_dev_ = std::allocate_shared<GPUMemoryManager>(allocator_);
  dst_dev_ = std::allocate_shared<GPUMemoryManager>(allocator_);
#endif

  // Start the camera
  if (!camera_->start()) {
    return;
  }

  // Start capture thread
  capture_thread_ = std::thread{
    [this]() -> void {
      while (ros::ok() && !canceled_.load()) {
        ROS_DEBUG("Capture...");
        auto img = camera_->capture();
        if (img == nullptr) {
          // Failed capturing image, assume it is temporarily and continue a bit later
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
          continue;
        }
        if(publish_next_frame_ == false){
          continue;
        }

        auto stamp = img->header.stamp;
        if (img->encoding != output_encoding_) {
#ifdef ENABLE_CUDA
          img = convertOnGpu(*img);
#else
          img = convert(*img);
#endif
        }
        img->header.stamp = stamp;
        img->header.frame_id = camera_frame_id_;

        auto ci = std::make_unique<sensor_msgs::CameraInfo>(cinfo_->getCameraInfo());
        if (!checkCameraInfo(*img, *ci)) {
          *ci = sensor_msgs::CameraInfo{};
          ci->height = img->height;
          ci->width = img->width;
        }

        ci->header.stamp = stamp;
        ci->header.frame_id = camera_frame_id_;
        publish_next_frame_ = publish_rate_ < 0;

        camera_transport_pub_.publish(*img, *ci);
      }
    }
  };
}

V4L2Camera::~V4L2Camera()
{
  canceled_.store(true);
  if (capture_thread_.joinable()) {
    capture_thread_.join();
  }
}

#endif
}  // namespace v4l2_camera
