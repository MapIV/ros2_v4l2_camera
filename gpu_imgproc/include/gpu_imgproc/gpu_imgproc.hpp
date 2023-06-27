#pragma once

#include <rclcpp/rclcpp.hpp>
// #include <rcl_interfaces/msg/parameter.hpp>

#include "accelerator/rectifier.hpp"
#include "accelerator/jpeg_compressor.hpp"
// #include <sensor_msgs/msg/compressed_image.hpp>

namespace gpu_imgproc {

class GpuImgProc : public rclcpp::Node {
public:
    explicit GpuImgProc(const rclcpp::NodeOptions & options);
    virtual ~GpuImgProc();

private:
    std::shared_ptr<Rectifier::NPPRectifier> npp_rectifier_;
    std::shared_ptr<Rectifier::OpenCVRectifierCPU> cv_cpu_rectifier_;
    std::shared_ptr<Rectifier::OpenCVRectifierGPU> cv_gpu_rectifier_;
    std::shared_ptr<JpegCompressor::JetsonCompressor> raw_compressor_;
    std::shared_ptr<JpegCompressor::JetsonCompressor> rect_compressor_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rectified_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr rect_compressed_pub_;

    Rectifier::Implementation rectifier_impl_;
};


} // namespace gpu_imgproc