#include "v4l2_camera/correction.hpp"
#include <rclcpp/rclcpp.hpp>

#include <nppi_data_exchange_and_initialization.h>
#include <nppi_geometry_transforms.h>
#include <nppi_support_functions.h>

#define CHECK_NPP_STATUS(status) \
    if (status != NPP_SUCCESS) { \
        RCLCPP_ERROR(rclcpp::get_logger("v4l2_camera"), "NPP error: %d (%s:%s)", status, __FILE__, __LINE__); \
    }

#define CHECK_CUDA_STATUS(status) \
    if (status != cudaSuccess) { \
        RCLCPP_ERROR(rclcpp::get_logger("v4l2_camera"), "CUDA error: %s (%s:%s)", cudaGetErrorName(status), __FILE__, __LINE__); \
    }

namespace Correction {

GPUCorrection::GPUCorrection(int width, int height,
                             float *map_x, float *map_y,
                             int interpolation) : pxl_map_x_(nullptr), pxl_map_y_(nullptr) {
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

GPUCorrection::GPUCorrection(int width, int height,
                             double *D, double *K, double *R, double *P,
                             int interpolation) : pxl_map_x_(nullptr), pxl_map_y_(nullptr) {
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

    // Create rectification map
    // TODO: Verify this works
    float *map_x = new float[width * height];
    float *map_y = new float[width * height];

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
            double icdist = 1.0 / cdist;
            double xd = x * icdist;
            double yd = y * icdist;
            double x2 = xd * xd;
            double y2 = yd * yd;
            double xy = xd * yd;
            double kr = 1 + p1 * r2 + p2 * r4;
            map_x[v * width + u] = (float)(fx * (xd * kr + 2 * p1 * xy + p2 * (r2 + 2 * x2)) + cx);
            map_y[v * width + u] = (float)(fy * (yd * kr + p1 * (r2 + 2 * y2) + 2 * p2 * xy) + cy);
        }
    }

    CHECK_CUDA_STATUS(cudaMemcpy2D(pxl_map_x_, pxl_map_x_step_, map_x, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice));
    CHECK_CUDA_STATUS(cudaMemcpy2D(pxl_map_y_, pxl_map_y_step_, map_y, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice));

    interpolation_ = interpolation;

    delete[] map_x;
    delete[] map_y;
}

GPUCorrection::~GPUCorrection() {
    if (pxl_map_x_ != nullptr) {
        nppiFree(pxl_map_x_);
    }

    if (pxl_map_y_ != nullptr) {
        nppiFree(pxl_map_y_);
    }
}

Image::UniquePtr GPUCorrection::correct(Image::UniquePtr &msg) {
    Image::UniquePtr result = std::make_unique<Image>();
    result->header = msg->header;
    result->height = msg->height;
    result->width = msg->width;
    result->encoding = msg->encoding;
    result->is_bigendian = msg->is_bigendian;
    result->step = msg->step;

    result->data.resize(msg->data.size());

    NppiRect src_roi = {0, 0, (int)msg->width, (int)msg->height};
    NppiSize src_size = {(int)msg->width, (int)msg->height};
    NppiSize dst_roi_size = {(int)msg->width, (int)msg->height};
    int src_step;
    int dst_step;

    Npp8u *src = nppiMalloc_8u_C3(msg->width, msg->height, &src_step);
    Npp8u *dst = nppiMalloc_8u_C3(msg->width, msg->height, &dst_step);

    CHECK_CUDA_STATUS(cudaMemcpy2D(src, src_step, msg->data.data(), msg->step, msg->width * 3, msg->height, cudaMemcpyHostToDevice));

    NppiInterpolationMode interpolation = NPPI_INTER_LINEAR;

    CHECK_NPP_STATUS(nppiRemap_8u_C3R(
        src, src_size, src_step, src_roi,
        pxl_map_x_, pxl_map_x_step_, pxl_map_y_, pxl_map_y_step_,
        dst, dst_step, dst_roi_size, interpolation));

    CHECK_CUDA_STATUS(cudaMemcpy2D(result->data.data(), result->step, dst, dst_step, msg->width * 3, msg->height, cudaMemcpyDeviceToHost));

    nppiFree(src);
    nppiFree(dst);

    return result;
}

} // namespace Correction