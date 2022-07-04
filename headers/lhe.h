// adding def gaurds
#ifndef LHE_H
#define LHE_H

#ifdef __CUDACC__
#define DEVICE_CALLABLE __device__
#define HOST_CALLABLE __host__
#define GLOBAL_CALLABLE __global__
#else
#define DEVICE_CALLABLE
#define HOST_CALLABLE
#define GLOBAL_CALLABLE
#endif

#define PIXEL_RANGE 256
#define RGB_CHANNELS 3
#define MAX_PIXEL_VAL 255

#include <cuda.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/opencv.hpp>

__global__ void
apply_lhe(uchar* base,
          const uchar* img,
          int window,
          int width,
          int height,
          int steps,
          int channels_c);
// end def gaurds
#endif // HELPERS_HPP