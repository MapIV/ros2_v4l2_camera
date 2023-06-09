#pragma once

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <turbojpeg.h>

#include <NvJpegEncoder.h>
#include <cuda/api.hpp>

#include <string>

class NvJPEGEncoder;

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

enum Sampling {
    YUV420,
    YUV422,
};

enum ImageFormat {
    RGB,
    BGR
};

class CPUCompressor {
public:
    CPUCompressor();
    ~CPUCompressor();

    CompressedImage::UniquePtr compress(const Image &msg, int quality = 90,int format = TJPF_RGB, int sampling = TJ_420);
private:
    tjhandle handle_;
    unsigned char *jpegBuf_;
    unsigned long size_;
};

// TODO: Make this cleaner (don't want to include NvJpegEncoder.h outside this library)
class JetsonCompressor {
public:
    JetsonCompressor(std::string name);
    ~JetsonCompressor();

    CompressedImage::UniquePtr compress(const Image &msg, int quality = 90, int format = BGR, int sampling = YUV420);
private:
    NvJPEGEncoder *encoder_;
    size_t image_size{};
    size_t yuv_size{};
    cuda::memory::device::unique_ptr<uint8_t[]> dev_image;
    cuda::memory::host::unique_ptr<uint8_t[]> host_yuv;
    cuda::memory::device::unique_ptr<uint8_t[]> dev_yuv;
};

} // namespace JpegCompression