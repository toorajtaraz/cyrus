//adding def gaurds
#ifndef HELPERS_H
#define HELPERS_H

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

#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <cuda.h>

//device function that extracts histogram of an image on gpu
__device__ void extract_histogram_rgb(const uchar* img, int width, int height, int *count, int x_start, int x_end, int y_start, int y_end, int steps, short channel, short channels_c = 3, int *histt = NULL, int sw = 1);
__device__ void extract_histogram(const uchar* img, int width, int height, int *count, int x_start, int x_end, int y_start, int y_end, int steps, int *histt = NULL, int sw = 1);
__device__ double *calculate_probability(int *hist, int total_pixels);
__device__ double *buildLook_up_table(double *prob);
__device__ double *buildLook_up_table_rgb(int *hist_blue, int *hist_green, int *hist_red, int count, bool free_sw = true);
__global__ void test_histogram(const uchar* img, int width, int height, int *count, int x_start, int x_end, int y_start, int y_end, int steps, int block_size, int block_threads, short channel, short channels_c = 3, int *histt = NULL, int sw = 1);
//end def gaurds
#endif // HELPERS_HPP