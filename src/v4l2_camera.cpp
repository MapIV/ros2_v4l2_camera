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

#include <sensor_msgs/image_encodings.h>

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
  : image_transport_(private_nh),
    canceled_{false}
{
  private_nh.getParam("publish_rate", publish_rate_);
  private_nh.getParam("video_device", device);
  private_nh.getParam("use_v4l2_buffer_timestamps", use_v4l2_buffer_timestamps);
  private_nh.getParam("timestamp_offset", timestamp_offset);

  if(std::abs(publish_rate_) < std::numeric_limits<double>::epsilon()){
    ROS_WARN("Invalid publish_rate = 0. Use default value -1 instead");
    publish_rate_ = -1.0;
  }
  // if(publish_rate_ > 0){
  //   const auto publish_period = ros::Duration(publish_rate_);
  //   image_pub_timer_ = node.createTimer(publish_period, &V4L2Camera::publishTimer, this);
  //   publish_next_frame_ = false;
  // }
  // else{
  //   publish_next_frame_ = true;
  // }
  camera_transport_pub_ = image_transport_.advertiseCamera("image_raw", 10);

  ros::Duration timestamp_offset_duration(0, timestamp_offset);

  camera_ = std::make_shared<V4l2CameraDevice>(device, use_v4l2_buffer_timestamps, timestamp_offset_duration);
  
  if (!camera_->open()) {
    return;
  }
  auto camera_info_url_ = "";
  cinfo_ = std::make_shared<camera_info_manager::CameraInfoManager>(private_nh, camera_->getCameraName(), camera_info_url_);
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

void V4L2Camera::publishTimer()
{
  this->publish_next_frame_=true;
}

bool V4L2Camera::checkCameraInfo(
  sensor_msgs::Image const & img,
  sensor_msgs::CameraInfo const & ci)
{
  return ci.width == img.width && ci.height == img.height;
}

static unsigned char CLIPVALUE(int val)
{
  // Old method (if)
  val = val < 0 ? 0 : val;
  return val > 255 ? 255 : val;
}

/**
 * Conversion from YUV to RGB.
 * The normal conversion matrix is due to Julien (surname unknown):
 *
 * [ R ]   [  1.0   0.0     1.403 ] [ Y ]
 * [ G ] = [  1.0  -0.344  -0.714 ] [ U ]
 * [ B ]   [  1.0   1.770   0.0   ] [ V ]
 *
 * and the firewire one is similar:
 *
 * [ R ]   [  1.0   0.0     0.700 ] [ Y ]
 * [ G ] = [  1.0  -0.198  -0.291 ] [ U ]
 * [ B ]   [  1.0   1.015   0.0   ] [ V ]
 *
 * Corrected by BJT (coriander's transforms RGB->YUV and YUV->RGB
 *                   do not get you back to the same RGB!)
 * [ R ]   [  1.0   0.0     1.136 ] [ Y ]
 * [ G ] = [  1.0  -0.396  -0.578 ] [ U ]
 * [ B ]   [  1.0   2.041   0.002 ] [ V ]
 *
 */

static void YUV2RGB(
  const unsigned char y, const unsigned char u, const unsigned char v, unsigned char * r,
  unsigned char * g, unsigned char * b)
{
  const int y2 = static_cast<int>(y);
  const int u2 = static_cast<int>(u) - 128;
  const int v2 = static_cast<int>(v) - 128;
  // std::cerr << "YUV=("<<y2<<","<<u2<<","<<v2<<")"<<std::endl;

  // This is the normal YUV conversion, but
  // appears to be incorrect for the firewire cameras
  //   int r2 = y2 + ( (v2*91947) >> 16);
  //   int g2 = y2 - ( ((u2*22544) + (v2*46793)) >> 16 );
  //   int b2 = y2 + ( (u2*115999) >> 16);
  // This is an adjusted version (UV spread out a bit)
  int r2 = y2 + ((v2 * 37221) >> 15);
  int g2 = y2 - (((u2 * 12975) + (v2 * 18949)) >> 15);
  int b2 = y2 + ((u2 * 66883) >> 15);
  // std::cerr << "   RGB=("<<r2<<","<<g2<<","<<b2<<")"<<std::endl;

  // Cap the values.
  *r = CLIPVALUE(r2);
  *g = CLIPVALUE(g2);
  *b = CLIPVALUE(b2);
}

static void yuyv2rgb(unsigned char const * YUV, unsigned char * RGB, int NumPixels)
{
  int i, j;
  unsigned char y0, y1, u, v;
  unsigned char r, g, b;

  for (i = 0, j = 0; i < (NumPixels << 1); i += 4, j += 6) {
    y0 = YUV[i + 0];
    u = YUV[i + 1];
    y1 = YUV[i + 2];
    v = YUV[i + 3];
    YUV2RGB(y0, u, v, &r, &g, &b);
    RGB[j + 0] = r;
    RGB[j + 1] = g;
    RGB[j + 2] = b;
    YUV2RGB(y1, u, v, &r, &g, &b);
    RGB[j + 3] = r;
    RGB[j + 4] = g;
    RGB[j + 5] = b;
  }
}

sensor_msgs::ImagePtr V4L2Camera::convert(sensor_msgs::Image& img)
{
  // TODO(sander): temporary until cv_bridge and image_proc are available in ROS 2
  if (img.encoding == sensor_msgs::image_encodings::YUV422 &&
    output_encoding_ == sensor_msgs::image_encodings::RGB8)
  {
    auto outImg = boost::make_shared<sensor_msgs::Image>();
    outImg->width = img.width;
    outImg->height = img.height;
    outImg->step = img.width * 3;
    outImg->encoding = output_encoding_;
    outImg->data.resize(outImg->height * outImg->step);
    for (auto i = 0u; i < outImg->height; ++i) {
      yuyv2rgb(
        img.data.data() + i * img.step, outImg->data.data() + i * outImg->step,
        outImg->width);
    }
    return outImg;
  } else {
    ROS_WARN_ONCE(
      "Conversion not supported yet: %s -> %s", img.encoding.c_str(), output_encoding_.c_str());
    return nullptr;
  }
}

V4L2Camera::~V4L2Camera()
{
  canceled_.store(true);
  if (capture_thread_.joinable()) {
    capture_thread_.join();
  }
}
}  // namespace v4l2_camera
