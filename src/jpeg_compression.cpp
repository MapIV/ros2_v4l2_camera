#include <cstdio>
#include <cstring>
#include <v4l2_camera/jpeg_compression.hpp>

// #include <NvJpegEncoder.h>

#define TEST_ERROR(cond, str) if(cond) { \
                                        fprintf(stderr, "%s\n", str); }

namespace JpegCompression {

JpegCompressor::JpegCompressor()
    : jpegBuf_(nullptr), size_(0) {
    handle_ = tjInitCompress();
}

JpegCompressor::~JpegCompressor() {
    if (jpegBuf_)
        tjFree(jpegBuf_);
    tjDestroy(handle_);
}

CompressedImage::UniquePtr JpegCompressor::compress(Image::UniquePtr &msg, int quality, int sampling, int format) {
    CompressedImage::UniquePtr compressed_msg = std::make_unique<CompressedImage>();
    compressed_msg->header = msg->header;
    compressed_msg->format = "jpeg";

    if (jpegBuf_) {
        tjFree(jpegBuf_);
        jpegBuf_ = nullptr;
    }

    int tjres = tjCompress2(handle_,
                            msg->data.data(),
                            msg->width,
                            0,
                            msg->height,
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

} // namespace JpegCompression