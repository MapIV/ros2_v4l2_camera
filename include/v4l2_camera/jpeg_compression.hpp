#pragma once

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <turbojpeg.h>

// class NvJPEGEncoder;

namespace JpegCompression {
using Image = sensor_msgs::msg::Image;
using CompressedImage = sensor_msgs::msg::CompressedImage;

// class JpegCompressor {
// public:
//     JpegCompressor();
//     ~JpegCompressor();

//     CompressedImage::UniquePtr compress(Image::UniquePtr &msg);
// private:
//     NvJPEGEncoder *jpegenc;

//     uint32_t in_width;
//     uint32_t in_height;
//     uint32_t in_pixfmt;

//     bool got_error;
//     bool use_fd;

//     bool perf;

//     uint32_t crop_left;
//     uint32_t crop_top;
//     uint32_t crop_width;
//     uint32_t crop_height;
//     int  stress_test;
//     bool scaled_encode;
//     uint32_t scale_width;
//     uint32_t scale_height;
//     int quality;
// };

class JpegCompressor {
public:
    JpegCompressor();
    ~JpegCompressor();

    CompressedImage::UniquePtr compress(Image::UniquePtr &msg, int quality = 95, int sampling = TJ_420, int format = TJPF_RGB);
private:
    tjhandle handle_;
    unsigned char *jpegBuf_;
    unsigned long size_;
};

} // namespace JpegCompression