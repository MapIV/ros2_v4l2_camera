#include <cstdio>
#include <cstring>
#include <nppi_color_conversion.h>

#include "accelerator/jpeg_compressor.hpp"

#define TEST_ERROR(cond, str) if(cond) { \
                                        fprintf(stderr, "%s\n", str); }

namespace JpegCompressor {

CPUCompressor::CPUCompressor()
    : jpegBuf_(nullptr), size_(0) {
    handle_ = tjInitCompress();
}

CPUCompressor::~CPUCompressor() {
    if (jpegBuf_)
        tjFree(jpegBuf_);
    tjDestroy(handle_);
}

CompressedImage::UniquePtr CPUCompressor::compress(const Image &msg, int quality, int format, int sampling) {
    CompressedImage::UniquePtr compressed_msg = std::make_unique<CompressedImage>();
    compressed_msg->header = msg.header;
    compressed_msg->format = "jpeg";

    if (jpegBuf_) {
        tjFree(jpegBuf_);
        jpegBuf_ = nullptr;
    }

    int tjres = tjCompress2(handle_,
                            msg.data.data(),
                            msg.width,
                            0,
                            msg.height,
                            format,
                            &jpegBuf_,
                            &size_,
                            sampling,
                            quality,
                            TJFLAG_FASTDCT);

    TEST_ERROR(tjres != 0, tjGetErrorStr2(handle_));

    compressed_msg->data.resize(size_);
    memcpy(compressed_msg->data.data(), jpegBuf_, size_);

    return compressed_msg;
}


#ifdef ENABLE_JETSON
JetsonCompressor::JetsonCompressor(std::string name) {
    encoder_ = NvJPEGEncoder::createJPEGEncoder(name.c_str());
}

JetsonCompressor::~JetsonCompressor() {
    delete encoder_;
}

CompressedImage::UniquePtr JetsonCompressor::compress(const Image &msg, int quality, ImageFormat format) {
    CompressedImage::UniquePtr compressed_msg = std::make_unique<CompressedImage>();
    compressed_msg->header = msg.header;
    compressed_msg->format = "jpeg";

    int width = msg.width;
    int height = msg.height;
    const auto &img = msg.data;

    if (image_size < img.size()) {
      dev_image = cuda::memory::device::make_unique<uint8_t[]>(img.size());
      yuv_size =
          width * height +
          (static_cast<size_t>(width / 2) * static_cast<size_t>(height / 2)) * 2;
      host_yuv = cuda::memory::host::make_unique<uint8_t[]>(yuv_size);
      dev_yuv = cuda::memory::device::make_unique<uint8_t[]>(yuv_size);
      image_size = img.size();
    }

    cuda::memory::copy(dev_image.get(), img.data(), img.size());

    if (format == ImageFormat::RGB) {
        TEST_ERROR(cudaRGBToYUV420(dev_image.get(), dev_yuv.get(), width, height) !=
                cudaSuccess, "failed to convert rgb8 to yuv420");
    } else {
        TEST_ERROR(cudaBGRToYUV420(dev_image.get(), dev_yuv.get(), width, height) !=
                cudaSuccess, "failed to convert bgr8 to yuv420");
    }

    cuda::memory::copy(host_yuv.get(), dev_yuv.get(), yuv_size);

    NvBuffer buffer(V4L2_PIX_FMT_YUV420M, width, height, 0);
    TEST_ERROR(buffer.allocateMemory() != 0, "NvBuffer allocation failed");

    auto image_data = reinterpret_cast<int8_t *>(host_yuv.get());

    for (uint32_t i = 0; i < buffer.n_planes; ++i) {
        NvBuffer::NvBufferPlane &plane = buffer.planes[i];
        plane.bytesused = plane.fmt.stride * plane.fmt.height;
        memcpy(plane.data, image_data, plane.bytesused);
        image_data += plane.bytesused;
    }

    size_t out_buf_size = width * height * 3 / 2;
    compressed_msg->data.resize(out_buf_size);
    auto out_data = compressed_msg->data.data();

    TEST_ERROR(
        encoder_->encodeFromBuffer(buffer, JCS_YCbCr, &out_data,
                                   out_buf_size, quality),
        "NvJpeg Encoder Error");

    buffer.deallocateMemory();

    compressed_msg->data.resize(out_buf_size);
    
    return compressed_msg;
}
#endif

} // namespace JpegCompressor