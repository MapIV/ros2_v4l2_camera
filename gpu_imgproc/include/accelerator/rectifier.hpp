#pragma once

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <nppdefs.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core.hpp>

using CameraInfo = sensor_msgs::msg::CameraInfo;
using Image = sensor_msgs::msg::Image;

namespace Rectifier {

class NPPRectifier {
public:
    NPPRectifier(int width, int height,
                  const Npp32f *map_x, const Npp32f *map_y,
                  int interpolation = NPPI_INTER_LINEAR);
    NPPRectifier(const CameraInfo &info,
                  int interpolation = NPPI_INTER_LINEAR,
                  bool use_opencv_map = false);
    ~NPPRectifier();

    Image::UniquePtr rectify(const Image &msg);
private:
    Npp32f *pxl_map_x_;
    Npp32f *pxl_map_y_;
    int pxl_map_x_step_;
    int pxl_map_y_step_;
    int interpolation_;
    cudaStream_t stream_;
};

#ifdef ENABLE_OPENCV
class OpenCVRectifierCPU {
public:
    OpenCVRectifierCPU(const CameraInfo &info);
    ~OpenCVRectifierCPU();

    Image::UniquePtr rectify(const Image &msg);
private:
    cv::Mat map_x_;
    cv::Mat map_y_;
};
#endif

#ifdef ENABLE_OPENCV_CUDA
class OpenCVRectifierGPU {
public:
    OpenCVRectifierGPU(const CameraInfo &info);
    ~OpenCVRectifierGPU();

    Image::UniquePtr rectify(const Image &msg);
private:
    cv::cuda::GpuMat map_x_;
    cv::cuda::GpuMat map_y_;
};
#endif

} // namespace Rectifier