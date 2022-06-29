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
__device__ void helper_extract_histogram_rgb(const uchar* img,  int *count, int x_start, int x_end, int y_start, int y_end, int width, int height, int steps, short channel, short channels_c = 3, int *histt = NULL, int sw = 1);
__device__ void helper_extract_histogram(const uchar* img, int *count, int x_start, int x_end, int y_start, int y_end, int width, int height, int steps, int *histt = NULL, int sw = 1);
__device__ void helper_calculate_probability(int *hist, int total_pixels, double *prob);
__device__ void helper_buildLook_up_table(double *prob, double *lut);

__global__ void extract_histogram_rgb(const uchar* img,  int *count, int x_start, int x_end, int y_start, int y_end, int width, int height, int steps, short channel, short channels_c = 3, int *histt = NULL, int sw = 1);
__global__ void extract_histogram(const uchar* img, int *count, int x_start, int x_end, int y_start, int y_end, int width, int height, int steps, int *histt = NULL, int sw = 1);
__global__ void calculate_probability(int *hist, int total_pixels, double *prob);
__global__ void buildLook_up_table(double *prob, double *lut);
__global__ void buildLook_up_table_rgb(int *hist_blue, int *hist_green, int *hist_red, int count, bool free_sw = true, double *lut_final = NULL, double *lut_blue = NULL, double *lut_green = NULL, double *lut_red = NULL);


__global__ void apply_LHE(uchar *base, const uchar *img, int window, int width, int height, int steps, int channels_c);
//end def gaurds
#endif // HELPERS_HPP