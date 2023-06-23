#include "accelerator/rectifier.hpp"
#include <rclcpp/rclcpp.hpp>

#include <npp.h>
#include <nppi_data_exchange_and_initialization.h>
#include <nppi_geometry_transforms.h>
#include <nppi_support_functions.h>

#ifdef ENABLE_OPENCV
#include <image_geometry/pinhole_camera_model.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs.hpp>

#ifdef ENABLE_OPENCV_CUDA
// #include <opencv2/cudafeatures2d.hpp>
// #include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/core.hpp>
#endif
#endif


#define CHECK_NPP_STATUS(status) \
    if (status != NPP_SUCCESS) { \
        RCLCPP_ERROR(rclcpp::get_logger("v4l2_camera"), "NPP error: %d (%s:%d)", status, __FILE__, __LINE__); \
    }

#define CHECK_CUDA_STATUS(status) \
    if (status != cudaSuccess) { \
        RCLCPP_ERROR(rclcpp::get_logger("v4l2_camera"), "CUDA error: %s (%s:%d)", cudaGetErrorName(status), __FILE__, __LINE__); \
    }

namespace Rectifier {

void compute_maps(int width, int height, const double *D, const double *P,
                  float *map_x, float *map_y) {
    double fx = P[0];
    double fy = P[5];
    double cx = P[2];
    double cy = P[6];

    double k1 = D[0];
    double k2 = D[1];
    double p1 = D[2];
    double p2 = D[3];
    double k3 = D[4];

    for (int v = 0; v < height; v++) {
        for (int u = 0; u < width; u++) {
            double x = (u - cx) / fx;
            double y = (v - cy) / fy;
            double r2 = x * x + y * y;
            double r4 = r2 * r2;
            double r6 = r4 * r2;
            double cdist = 1 + k1 * r2 + k2 * r4 + k3 * r6;
            double xd = x * cdist;
            double yd = y * cdist;
            double x2 = xd * xd;
            double y2 = yd * yd;
            double xy = xd * yd;
            double kr = 1 + p1 * r2 + p2 * r4;
            map_x[v * width + u] = (float)(fx * (xd * kr + 2 * p1 * xy + p2 * (r2 + 2 * x2)) + cx);
            map_y[v * width + u] = (float)(fy * (yd * kr + p1 * (r2 + 2 * y2) + 2 * p2 * xy) + cy);
        }
    }
}

#ifdef ENABLE_OPENCV
void compute_maps_opencv(const CameraInfo &info, float *map_x, float *map_y) {
    cv::Mat camera_intrinsics(3, 3, CV_64F);
    cv::Mat distortion_coefficients(1, 5, CV_64F);

    for (int row=0; row<3; row++) {
        for (int col=0; col<3; col++) {
            camera_intrinsics.at<double>(row, col) = info.k[row * 3 + col];
        }
    }

    for (int col=0; col<5; col++) {
        distortion_coefficients.at<double>(col) = info.d[col];
    }

    cv::Mat m1(info.height, info.width, CV_32FC1, map_x);
    cv::Mat m2(info.height, info.width, CV_32FC1, map_y);
    cv::initUndistortRectifyMap(camera_intrinsics,
        distortion_coefficients,
        cv::Mat(),
        camera_intrinsics,
        cv::Size(info.width, info.height),
        CV_32FC1,
        m1, m2);
}
#endif

NPPRectifier::NPPRectifier(int width, int height,
                             const Npp32f *map_x, const Npp32f *map_y,
                             int interpolation) : pxl_map_x_(nullptr), pxl_map_y_(nullptr) {
    cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);

    pxl_map_x_ = nppiMalloc_32f_C1(width, height, &pxl_map_x_step_);
    if (pxl_map_x_ == nullptr) {
        RCLCPP_ERROR(rclcpp::get_logger("v4l2_camera"), "Failed to allocate GPU memory");
        return;
    }
    pxl_map_y_ = nppiMalloc_32f_C1(width, height, &pxl_map_y_step_);
    if (pxl_map_y_ == nullptr) {
        RCLCPP_ERROR(rclcpp::get_logger("v4l2_camera"), "Failed to allocate GPU memory");
        return;
    }
    CHECK_CUDA_STATUS(cudaMemcpy2D(pxl_map_x_, pxl_map_x_step_, map_x, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice));
    CHECK_CUDA_STATUS(cudaMemcpy2D(pxl_map_y_, pxl_map_y_step_, map_y, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice));
    interpolation_ = interpolation;
}

NPPRectifier::NPPRectifier(const CameraInfo& info, int interpolation, bool use_opencv_map)
    : interpolation_(interpolation) {
    cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);

    nppSetStream(stream_);

    pxl_map_x_ = nppiMalloc_32f_C1(info.width, info.height, &pxl_map_x_step_);
    if (pxl_map_x_ == nullptr) {
        RCLCPP_ERROR(rclcpp::get_logger("v4l2_camera"), "Failed to allocate GPU memory");
        return;
    }
    pxl_map_y_ = nppiMalloc_32f_C1(info.width, info.height, &pxl_map_y_step_);
    if (pxl_map_y_ == nullptr) {
        RCLCPP_ERROR(rclcpp::get_logger("v4l2_camera"), "Failed to allocate GPU memory");
        return;
    }

    // Create rectification map
    // TODO: Verify this works
    float *map_x = new float[info.width * info.height];
    float *map_y = new float[info.width * info.height];

#ifdef ENABLE_OPENCV
    if (use_opencv_map)
        compute_maps_opencv(info, map_x, map_y);
    else
#endif
    compute_maps(info.width, info.height,
                 info.d.data(), info.p.data(),
                 map_x, map_y);

    CHECK_CUDA_STATUS(cudaMemcpy2D(pxl_map_x_, pxl_map_x_step_, map_x, info.width * sizeof(float), info.width * sizeof(float), info.height, cudaMemcpyHostToDevice));
    CHECK_CUDA_STATUS(cudaMemcpy2D(pxl_map_y_, pxl_map_y_step_, map_y, info.width * sizeof(float), info.width * sizeof(float), info.height, cudaMemcpyHostToDevice));

    interpolation_ = interpolation;

    delete[] map_x;
    delete[] map_y;
}

NPPRectifier::~NPPRectifier() {
    if (pxl_map_x_ != nullptr) {
        nppiFree(pxl_map_x_);
    }

    if (pxl_map_y_ != nullptr) {
        nppiFree(pxl_map_y_);
    }

    cudaStreamDestroy(stream_);
}

Image::UniquePtr NPPRectifier::rectify(const Image &msg) {
    Image::UniquePtr result = std::make_unique<Image>();
    result->header = msg.header;
    result->height = msg.height;
    result->width = msg.width;
    result->encoding = msg.encoding;
    result->is_bigendian = msg.is_bigendian;
    result->step = msg.step;

    result->data.resize(msg.data.size());

    NppiRect src_roi = {0, 0, (int)msg.width, (int)msg.height};
    NppiSize src_size = {(int)msg.width, (int)msg.height};
    NppiSize dst_roi_size = {(int)msg.width, (int)msg.height};
    int src_step;
    int dst_step;

    Npp8u *src = nppiMalloc_8u_C3(msg.width, msg.height, &src_step);
    Npp8u *dst = nppiMalloc_8u_C3(msg.width, msg.height, &dst_step);

    CHECK_CUDA_STATUS(cudaMemcpy2D(src, src_step, msg.data.data(), msg.step, msg.width * 3, msg.height, cudaMemcpyHostToDevice));

    NppiInterpolationMode interpolation = NPPI_INTER_LINEAR;

    CHECK_NPP_STATUS(nppiRemap_8u_C3R(
        src, src_size, src_step, src_roi,
        pxl_map_x_, pxl_map_x_step_, pxl_map_y_, pxl_map_y_step_,
        dst, dst_step, dst_roi_size, interpolation));

    CHECK_CUDA_STATUS(cudaMemcpy2D(result->data.data(), result->step, dst, dst_step, msg.width * 3, msg.height, cudaMemcpyDeviceToHost));

    // cv::Mat image(msg.height, msg.width, CV_8UC3, result->data.data(), result->step);
    // cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    // // save with timestamp
    // std::string filename = "rectified_" + std::to_string(msg.header.stamp.sec) + "_" + std::to_string(msg.header.stamp.nanosec) + ".png";
    // imwrite(filename, image);

    nppiFree(src);
    nppiFree(dst);

    return result;
}

#ifdef ENABLE_OPENCV
OpenCVRectifierCPU::OpenCVRectifierCPU(const CameraInfo &info) {
    cv::Mat camera_intrinsics(3, 3, CV_64F);
    cv::Mat distortion_coefficients(1, 5, CV_64F);

    for (int row=0; row<3; row++) {
        for (int col=0; col<3; col++) {
            camera_intrinsics.at<double>(row, col) = info.k[row * 3 + col];
        }
    }

    for (int col=0; col<5; col++) {
        distortion_coefficients.at<double>(col) = info.d[col];
    }

    cv::Mat m1;
    cv::Mat m2;
    cv::initUndistortRectifyMap(camera_intrinsics,
        distortion_coefficients,
        cv::Mat(),
        camera_intrinsics,
        cv::Size(info.width, info.height),
        CV_32FC1,
        map_x_, map_y_);
}

OpenCVRectifierCPU::~OpenCVRectifierCPU() {}

Image::UniquePtr OpenCVRectifierCPU::rectify(const Image &msg) {
    Image::UniquePtr result = std::make_unique<Image>();
    result->header = msg.header;
    result->height = msg.height;
    result->width = msg.width;
    result->encoding = msg.encoding;
    result->is_bigendian = msg.is_bigendian;
    result->step = msg.step;

    result->data.resize(msg.data.size());

    cv::Mat src(msg.height, msg.width, CV_8UC3, (void *)msg.data.data());
    cv::Mat dst(msg.height, msg.width, CV_8UC3, (void *)result->data.data());

    cv::remap(src, dst, map_x_, map_y_, cv::INTER_LINEAR);

    return result;
}
#endif

#ifdef ENABLE_OPENCV_CUDA
OpenCVRectifierGPU::OpenCVRectifierGPU(const CameraInfo &info) {
    cv::Mat camera_intrinsics(3, 3, CV_64F);
    cv::Mat distortion_coefficients(1, 5, CV_64F);

    for (int row=0; row<3; row++) {
        for (int col=0; col<3; col++) {
            camera_intrinsics.at<double>(row, col) = info.k[row * 3 + col];
        }
    }

    for (int col=0; col<5; col++) {
        distortion_coefficients.at<double>(col) = info.d[col];
    }

    cv::Mat m1;
    cv::Mat m2;
    cv::initUndistortRectifyMap(camera_intrinsics,
        distortion_coefficients,
        cv::Mat(),
        camera_intrinsics,
        cv::Size(info.width, info.height),
        CV_32FC1,
        m1, m2);
    map_x_ = cv::cuda::GpuMat(m1);
    map_y_ = cv::cuda::GpuMat(m2);
}

OpenCVRectifierGPU::~OpenCVRectifierGPU() {}

Image::UniquePtr OpenCVRectifierGPU::rectify(const Image &msg) {
    Image::UniquePtr result = std::make_unique<Image>();
    result->header = msg.header;
    result->height = msg.height;
    result->width = msg.width;
    result->encoding = msg.encoding;
    result->is_bigendian = msg.is_bigendian;
    result->step = msg.step;

    result->data.resize(msg.data.size());

    cv::Mat src(msg.height, msg.width, CV_8UC3, (void *)msg.data.data());
    cv::cuda::GpuMat d_src = cv::cuda::GpuMat(src);
    cv::cuda::GpuMat d_dst = cv::cuda::GpuMat(cv::Size(msg.width, msg.height), src.type());

    cv::cuda::remap(d_src, d_dst, map_x_, map_y_, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

    // copy back to result
    cv::Mat dst(msg.height, msg.width, CV_8UC3, (void *)result->data.data());
    d_dst.download(dst);

    // cv::Mat image(msg.height, msg.width, CV_8UC3, result->data.data(), result->step);
    // cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    // // save with timestamp
    // std::string filename = "rectified_" + std::to_string(msg.header.stamp.sec) + "_" + std::to_string(msg.header.stamp.nanosec) + ".png";
    // imwrite(filename, image);

    return result;
}
#endif

} // namespace Rectifier