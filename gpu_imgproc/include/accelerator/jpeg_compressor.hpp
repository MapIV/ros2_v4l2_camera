#pragma once

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <turbojpeg.h>

#include <NvJpegEncoder.h>
#include <cuda/api.hpp>

#include <string>

class NvJPEGEncoder;

namespace JpegCompressor {
using Image = sensor_msgs::msg::Image;
using CompressedImage = sensor_msgs::msg::CompressedImage;

enum class ImageFormat {
    RGB,
    BGR
};

class CPUCompressor {
public:
    CPUCompressor();
    ~CPUCompressor();

    CompressedImage::UniquePtr compress(const Image &msg, int quality = 90, int format = TJPF_RGB, int sampling = TJ_420);
private:
    tjhandle handle_;
    unsigned char *jpegBuf_;
    unsigned long size_;
};

// TODO: Make this cleaner (don't want to include NvJpegEncoder.h outside this library)
#ifdef ENABLE_JETSON
class JetsonCompressor {
public:
    JetsonCompressor(std::string name);
    ~JetsonCompressor();

    CompressedImage::UniquePtr compress(const Image &msg, int quality = 90, ImageFormat format = ImageFormat::RGB);
private:
    NvJPEGEncoder *encoder_;
    size_t image_size{};
    size_t yuv_size{};
    cuda::memory::device::unique_ptr<uint8_t[]> dev_image;
    cuda::memory::host::unique_ptr<uint8_t[]> host_yuv;
    cuda::memory::device::unique_ptr<uint8_t[]> dev_yuv;
};
#endif

} // namespace JpegCompressor