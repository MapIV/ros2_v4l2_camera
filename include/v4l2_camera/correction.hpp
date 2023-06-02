#pragma once

#include <sensor_msgs/msg/image.hpp>
#include <nppdefs.h>

using Image = sensor_msgs::msg::Image;

namespace Correction {

class GPUCorrection {
public:
    GPUCorrection(int width, int height,
                  Npp32f *map_x, Npp32f *map_y,
                  int interpolation = NPPI_INTER_LINEAR);
    GPUCorrection(int width, int height,
                  double *D, double *K, double *R, double *P,
                  int interpolation = NPPI_INTER_LINEAR);
    ~GPUCorrection();

    Image::UniquePtr correct(Image::UniquePtr &msg);
private:
    Npp32f *pxl_map_x_;
    Npp32f *pxl_map_y_;
    int pxl_map_x_step_;
    int pxl_map_y_step_;
    int interpolation_;
    cudaStream_t stream_;
};

}