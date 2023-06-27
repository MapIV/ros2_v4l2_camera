#pragma once

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <nppdefs.h>

#ifdef ENABLE_OPENCV
#include <opencv2/core.hpp>
#endif
#ifdef ENABLE_OPENCV_CUDA
#include <opencv2/core/cuda.hpp>
#endif

using CameraInfo = sensor_msgs::msg::CameraInfo;
using Image = sensor_msgs::msg::Image;

namespace Rectifier {

enum class Implementation {
    NPP,
    OpenCV_CPU,
    OpenCV_GPU
};

enum class MappingImpl {
    NPP,
    OpenCV
};

class NPPRectifier {
public:
    NPPRectifier(int width, int height,
                 const Npp32f *map_x, const Npp32f *map_y);
    NPPRectifier(const CameraInfo &info,
                 MappingImpl impl = MappingImpl::NPP);
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
    OpenCVRectifierCPU(const CameraInfo &info,
                       MappingImpl impl = MappingImpl::OpenCV);
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
    OpenCVRectifierGPU(const CameraInfo &info,
                       MappingImpl impl = MappingImpl::OpenCV);
    ~OpenCVRectifierGPU();

    Image::UniquePtr rectify(const Image &msg);
private:
    cv::cuda::GpuMat map_x_;
    cv::cuda::GpuMat map_y_;
};
#endif

} // namespace Rectifier