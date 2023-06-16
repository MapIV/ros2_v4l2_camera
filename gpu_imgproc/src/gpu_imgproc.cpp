#include "gpu_imgproc/gpu_imgproc.hpp"
#include <rclcpp_components/register_node_macro.hpp>

#include <future>

namespace gpu_imgproc {

GpuImgProc::GpuImgProc(const rclcpp::NodeOptions & options)
    : Node("gpu_imgproc", options) {
    RCLCPP_INFO(this->get_logger(), "Initializing node gpu_imgproc");

    raw_compressor_ = std::make_shared<JpegCompressor::JetsonCompressor>("raw_compressor");
    rect_compressor_ = std::make_shared<JpegCompressor::JetsonCompressor>("rect_compressor");

    rectified_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
        "/sensing/camera/test/image_rect", 10);
    compressed_pub_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(
        "/sensing/camera/test/image_raw/compressed", 10);
    rect_compressed_pub_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(
        "/sensing/camera/test/image_rect/compressed", 10);

    img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/sensing/camera/test/image_raw", 10,
        [this](const sensor_msgs::msg::Image::SharedPtr msg) {
            // RCLCPP_INFO(this->get_logger(), "Received image");


            std::future<void> rectified_msg;
            if (npp_rectifier_) {
                rectified_msg =
                    std::async(std::launch::async, [this, msg]() {
                        auto rect_img = npp_rectifier_->rectify(*msg);
                        auto rect_comp_img = rect_compressor_->compress(*rect_img);
                        rectified_pub_->publish(std::move(rect_img));
                        rect_compressed_pub_->publish(std::move(rect_comp_img));
                    });
            }

            std::future<void> compressed_msg =
                std::async(std::launch::async, [this, msg]() {
                    compressed_pub_->publish(std::move(raw_compressor_->compress(*msg)));
                });
            
            if (npp_rectifier_) {
                rectified_msg.wait();
            }
            compressed_msg.wait();
        });

    info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/sensing/camera/test/camera_info", 10,
        [this](const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
            // RCLCPP_INFO(this->get_logger(), "Received camera info");

            if (!npp_rectifier_) {
                npp_rectifier_ = std::make_shared<Rectifier::NPPRectifier>(*msg);
            }
        });
}

GpuImgProc::~GpuImgProc() {
    RCLCPP_INFO(this->get_logger(), "Shutting down node gpu_imgproc");
}

RCLCPP_COMPONENTS_REGISTER_NODE(GpuImgProc)
} // namespace gpu_imgproc