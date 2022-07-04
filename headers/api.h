// adding def gaurds
#ifndef API_H
#define API_H

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

#include <helpers.h>

cv::Mat
lhe_api(cv::Mat src, int window, long long* taken_time);
cv::Mat
interpolating_lhe_api(cv::Mat src,
                      int window,
                      long long* taken_time_pure,
                      long long* taken_time_total);
void
clean_up();
// end def gaurds
#endif // HELPERS_HPP