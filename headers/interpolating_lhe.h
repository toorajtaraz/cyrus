// adding def gaurds
#ifndef INTERPOLATING_LHE_H
#define INTERPOLATING_LHE_H

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
lhe_build_luts(double*** all_luts,
               const uchar* img,
               int offset,
               int width,
               int height,
               int channel_c,
               int steps);
__global__ void
apply_interpolating_lhe(uchar* base,
                        const uchar* img,
                        int window,
                        int offset,
                        int width,
                        int height,
                        int channel_n,
                        int steps,
                        double*** all_luts);
// end def gaurds
#endif // HELPERS_HPP