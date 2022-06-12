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
DEVICE_CALLABLE int *extract_histogram_rgb(cv::cuda::GpuMat &img, int *count, int x_start, int x_end, int y_start, int y_end, short channel, int *histt = NULL, int sw = 1);
DEVICE_CALLABLE int *extract_histogram(cv::cuda::GpuMat &img, int *count, int x_start, int x_end, int y_start, int y_end, int *histt = NULL, int sw = 1);
DEVICE_CALLABLE double *calculate_probability(int *hist, int total_pixels);
DEVICE_CALLABLE double *buildLook_up_table(double *prob);
DEVICE_CALLABLE double *buildLook_up_table_rgb(int *hist_blue, int *hist_green, int *hist_red, int count, bool free_sw = true);
GLOBAL_CALLABLE void test_histogram(cv::cuda::GpuMat &img, int *count, int x_start, int x_end, int y_start, int y_end, short channel, int *histt = NULL, int sw = 1);
//end def gaurds
#endif // HELPERS_HPP