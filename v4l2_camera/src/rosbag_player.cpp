#include <rclcpp/rclcpp.hpp>

#include <memory>

#include "acceleration/jpeg_compressor.hpp"
#include "acceleration/rectifier.hpp"

#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"

class RosbagPlayer : public rclcpp::Node
{
public:
    RosbagPlayer()
    : Node("rosbag_player")
    {
        rect_pub_ = this->create_publisher<sensor_msgs::msg::Image>("image_rect", 1);
        comp_pub_ = this->create_publisher<sensor_msgs::msg::CompressedImage>("image_raw/compressed", 1);
        rect_comp_pub_ = this->create_publisher<sensor_msgs::msg::CompressedImage>("image_rect/compressed", 1);

        raw_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "image_raw", 1, std::bind(&RosbagPlayer::compress, this, std::placeholders::_1));

        cinfo_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "camera_info", 1, std::bind(&RosbagPlayer::on_camera_info, this, std::placeholders::_1));

        // wait for camera info
        compressor_ = std::make_shared<JpegCompressor::JetsonCompressor>("jetson_compressor");
    }

    virtual ~RosbagPlayer() = default;
private:
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rect_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr comp_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr rect_comp_pub_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr raw_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cinfo_sub_;

    std::shared_ptr<Rectifier::NPPRectifier> npp_rectifier_;
    std::shared_ptr<Rectifier::OpenCVRectifierCPU> cv_rectifier_;
    std::shared_ptr<JpegCompressor::JetsonCompressor> compressor_;

    bool use_opencv_ = false;

    void compress(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        if (!npp_rectifier_) {
            RCLCPP_WARN(this->get_logger(), "No camera info received yet");
            return;
        }

        sensor_msgs::msg::Image::UniquePtr rect_msg;
        if (use_opencv_) {
            rect_msg = cv_rectifier_->rectify(*msg);
        } else {
            rect_msg = npp_rectifier_->rectify(*msg);
        }

        auto comp_msg = compressor_->compress(*msg);
        auto rect_comp_msg = compressor_->compress(*rect_msg);

        rect_pub_->publish(std::move(rect_msg));
        comp_pub_->publish(std::move(comp_msg));
        rect_comp_pub_->publish(std::move(rect_comp_msg));
    }

    void on_camera_info(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        if (!npp_rectifier_) {
            npp_rectifier_ = std::make_shared<Rectifier::NPPRectifier>(*msg);
            cv_rectifier_ = std::make_shared<Rectifier::OpenCVRectifierCPU>(*msg);
        }
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RosbagPlayer>();

    while (rclcpp::ok()) {
        rclcpp::spin_some(node);
    }
    rclcpp::shutdown();
    return 0;
}