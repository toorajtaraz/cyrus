// adding def gaurds
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

#include <cuda.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/opencv.hpp>

// device function that extracts histogram of an image on gpu
__device__ void
helper_extract_histogram_rgb(const uchar* img,
                             int* count,
                             int x_start,
                             int x_end,
                             int y_start,
                             int y_end,
                             int width,
                             int height,
                             int steps,
                             short channel,
                             short channels_c = 3,
                             int* histt = NULL,
                             int sw = 1);

__device__ void
helper_calculate_probability(int* hist, int total_pixels, double* prob);
__device__ void
helper_buildLook_up_table(double* prob, double* lut);

__global__ void
extract_histogram_rgb(const uchar* img,
                      int* count,
                      int x_start,
                      int x_end,
                      int y_start,
                      int y_end,
                      int width,
                      int height,
                      int steps,
                      short channel,
                      short channels_c = 3,
                      int* histt = NULL,
                      int sw = 1);

__global__ void
calculate_probability(int* hist, int total_pixels, double* prob);
__global__ void
buildLook_up_table(double* prob, double* lut);
__global__ void
buildLook_up_table_rgb(int* hist_blue,
                       int* hist_green,
                       int* hist_red,
                       int count,
                       bool free_sw = true,
                       double* lut_final = NULL,
                       double* lut_blue = NULL,
                       double* lut_green = NULL,
                       double* lut_red = NULL);

// Macro for checking CUDA error codes and exiting if an error occured.
#define CUDA_CHECK(condition)                                                  \
  do {                                                                         \
    cudaError_t error = condition;                                             \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr,                                                          \
              "CUDA error: %s:%d: %s \n",                                      \
              __FILE__,                                                        \
              __LINE__,                                                        \
              cudaGetErrorString(error));                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Macro that has 6 inputs, 3 hists and uses cudaMemset to zero out them and
// checks result using CUDA_CHECK.
#define ZERO_OUT_RGB(histr, histg, histb)                                      \
  do {                                                                         \
    CUDA_CHECK(cudaMemset(histr, 0, sizeof(int) * 256));                       \
    CUDA_CHECK(cudaMemset(histg, 0, sizeof(int) * 256));                       \
    CUDA_CHECK(cudaMemset(histb, 0, sizeof(int) * 256));                       \
  } while (0)

#define ZERO_OUT_COUNTS(countr, countb, countg)                                \
  do {                                                                         \
    CUDA_CHECK(cudaMemset(countr, 0, sizeof(int)));                            \
    CUDA_CHECK(cudaMemset(countb, 0, sizeof(int)));                            \
    CUDA_CHECK(cudaMemset(countg, 0, sizeof(int)));                            \
  } while (0)

#define VALIDATE_KERNEL_CALL()                                                 \
  do {                                                                         \
    cudaError_t error = cudaGetLastError();                                    \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr,                                                          \
              "CUDA error: %s:%d: %s \n",                                      \
              __FILE__,                                                        \
              __LINE__,                                                        \
              cudaGetErrorString(error));                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// end def gaurds
#endif // HELPERS_HPP