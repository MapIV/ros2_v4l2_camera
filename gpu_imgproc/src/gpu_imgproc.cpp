#include "gpu_imgproc/gpu_imgproc.hpp"
#include <rclcpp_components/register_node_macro.hpp>

#include <future>

namespace gpu_imgproc {

GpuImgProc::GpuImgProc(const rclcpp::NodeOptions & options)
    : Node("gpu_imgproc", options) {
    RCLCPP_INFO(this->get_logger(), "Initializing node gpu_imgproc");

    // std::string image_raw_topic = this->declare_parameter<std::string>("image_raw_topic", "/camera/image_raw");
    // std::string camera_info_topic = this->declare_parameter<std::string>("camera_info_topic", "/camera/camera_info");
    // std::string image_rect_topic = this->declare_parameter<std::string>("image_rect_topic", "/camera/image_rect");
    std::string rect_impl = this->declare_parameter<std::string>("rect_impl", "npp");
    bool use_opencv_map_init = this->declare_parameter<bool>("use_opencv_map_init", false);
    bool active = false;
    Rectifier::MappingImpl mapping_impl;

    // RCLCPP_INFO(this->get_logger(), "Subscribing to %s", image_raw_topic.c_str());
    // RCLCPP_INFO(this->get_logger(), "Subscribing to %s", camera_info_topic.c_str());
    // RCLCPP_INFO(this->get_logger(), "Publishing to %s", image_rect_topic.c_str());

    if (rect_impl == "npp") {
        RCLCPP_INFO(this->get_logger(), "Using NPP implementation for rectification");
        rectifier_impl_ = Rectifier::Implementation::NPP;
#ifdef ENABLE_OPENCV
    } else if (rect_impl == "opencv_cpu") {
        RCLCPP_INFO(this->get_logger(), "Using CPU OpenCV implementation for rectification");
        rectifier_impl_ = Rectifier::Implementation::OpenCV_CPU;
#endif
#ifdef ENABLE_OPENCV_CUDA
    } else if (rect_impl == "opencv_gpu") {
        RCLCPP_INFO(this->get_logger(), "Using GPU OpenCV implementation for rectification");
        rectifier_impl_ = Rectifier::Implementation::OpenCV_GPU;
#endif
    } else {
        RCLCPP_ERROR(this->get_logger(), "Invalid implementation: %s. Available options: npp, opencv_gpu, opencv_cpu", rect_impl.c_str());
        return;
    }

    if (use_opencv_map_init) {
        RCLCPP_INFO(this->get_logger(), "Using OpenCV map initialization");
        mapping_impl = Rectifier::MappingImpl::OpenCV;
    } else {
        RCLCPP_INFO(this->get_logger(), "Using Non-OpenCV map initialization");
        mapping_impl = Rectifier::MappingImpl::NPP;
    }

    raw_compressor_ = std::make_shared<JpegCompressor::JetsonCompressor>("raw_compressor");
    rect_compressor_ = std::make_shared<JpegCompressor::JetsonCompressor>("rect_compressor");

    rectified_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
        "image_rect", 10);
    compressed_pub_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(
        "image_raw/compressed", 10);
    rect_compressed_pub_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(
        "image_rect/compressed", 10);

    img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "image_raw", 10,
        [this, &active](const sensor_msgs::msg::Image::SharedPtr msg) {
            // RCLCPP_INFO(this->get_logger(), "Received image");

            std::future<void> rectified_msg;
            if (active) {
                rectified_msg =
                    std::async(std::launch::async, [this, msg]() {
                        sensor_msgs::msg::Image::UniquePtr rect_img;
                        sensor_msgs::msg::CompressedImage::UniquePtr rect_comp_img;
                        if (rectifier_impl_ == Rectifier::Implementation::NPP) {
                            rect_img = npp_rectifier_->rectify(*msg);
                            rect_comp_img = rect_compressor_->compress(*rect_img, 60);
#ifdef ENABLE_OPENCV                            
                        } else if (rectifier_impl_ == Rectifier::Implementation::OpenCV_CPU) {
                            rect_img = cv_cpu_rectifier_->rectify(*msg);
                            rect_comp_img = rect_compressor_->compress(*rect_img, 60);
#endif
#ifdef ENABLE_OPENCV_CUDA
                        } else if (rectifier_impl_ == Rectifier::Implementation::OpenCV_GPU) {
                            rect_img = cv_gpu_rectifier_->rectify(*msg);
                            rect_comp_img = rect_compressor_->compress(*rect_img, 60);
#endif
                        }
                        rectified_pub_->publish(std::move(rect_img));
                        rect_compressed_pub_->publish(std::move(rect_comp_img));
                    });
            }

            std::future<void> compressed_msg =
                std::async(std::launch::async, [this, msg]() {
                    compressed_pub_->publish(std::move(raw_compressor_->compress(*msg, 60)));
                });
            
            if (active) {
                rectified_msg.wait();
            }
            compressed_msg.wait();
        });

    info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "camera_info", 10,
        [this, mapping_impl, &active](const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
            RCLCPP_INFO(this->get_logger(), "Received camera info");

            if (!active) {
                switch(rectifier_impl_) {
                    case Rectifier::Implementation::NPP:
                        RCLCPP_INFO(this->get_logger(), "Initializing NPP rectifier");
                        npp_rectifier_ = std::make_shared<Rectifier::NPPRectifier>(*msg, mapping_impl);
                        if (npp_rectifier_) {
                            RCLCPP_INFO(this->get_logger(), "Initialized NPP rectifier");
                            active = true;
                        } else {
                            RCLCPP_ERROR(this->get_logger(), "Failed to initialize NPP rectifier");
                            return;
                        }
                        break;
                    case Rectifier::Implementation::OpenCV_CPU:
#ifdef ENABLE_OPENCV
                        RCLCPP_INFO(this->get_logger(), "Initializing OpenCV CPU rectifier");
                        cv_cpu_rectifier_ = std::make_shared<Rectifier::OpenCVRectifierCPU>(*msg, mapping_impl);
                        if (cv_cpu_rectifier_) {
                            RCLCPP_INFO(this->get_logger(), "Initialized OpenCV GPU rectifier");
                            active = true;
                        } else {
                            RCLCPP_ERROR(this->get_logger(), "Failed to initialize OpenCV rectifier");
                            return;
                        }
                        break;
#else
                        RCLCPP_ERROR(this->get_logger(), "OpenCV not enabled");
                        return;
#endif 
                    case Rectifier::Implementation::OpenCV_GPU:
#ifdef ENABLE_OPENCV_CUDA
                        RCLCPP_INFO(this->get_logger(), "Initializing OpenCV GPU rectifier");
                        cv_gpu_rectifier_ = std::make_shared<Rectifier::OpenCVRectifierGPU>(*msg, mapping_impl);
                        if (cv_gpu_rectifier_) {
                            RCLCPP_INFO(this->get_logger(), "Initialized OpenCV GPU rectifier");
                            active = true;
                        } else {
                            RCLCPP_ERROR(this->get_logger(), "Failed to initialize OpenCV rectifier");
                            return;
                        }
                        break;
#else
                        RCLCPP_ERROR(this->get_logger(), "OpenCV CUDA not enabled");
                        return;
#endif
                    default:
                        RCLCPP_ERROR(this->get_logger(), "Invalid rectifier implementation");
                        return;
                }

                // unsubscribe
                info_sub_.reset();
            }
        });
}

GpuImgProc::~GpuImgProc() {
    RCLCPP_INFO(this->get_logger(), "Shutting down node gpu_imgproc");
}

RCLCPP_COMPONENTS_REGISTER_NODE(GpuImgProc)
} // namespace gpu_imgproc