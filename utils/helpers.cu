#include <helpers.h>

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
                             short channels_c,
                             int* histt,
                             int sw)
{
  __shared__ int sh_histt[257];

  // calculate global id
  int index = threadIdx.y * blockDim.y + threadIdx.x;
  int s = 256 / (blockDim.x * blockDim.y);

  if (index == 0) {
    sh_histt[256] = 0;
  }

  if (s < 1)
    s = 1;

  for (int i = index * s; i < ((index + 1) * s); i++)
    if (i < 256) {
      sh_histt[i] = 0;
    }

  __syncthreads();
  // do if boundary is valid
  for (auto i = x_start; i < x_end && i < height; i++)
    for (auto j = y_start; j < y_end && j < width; j++) {
      atomicAdd(&sh_histt[256], sw);
      atomicAdd(&sh_histt[img[j * steps + i * channels_c + channel]], sw);
    }

  __syncthreads();
  if (index == 0) {
    atomicAdd(&count[0], sh_histt[256]);
  }

  for (int i = index * s; i < ((index + 1) * s); i++)
    if (i < 256) {
      atomicAdd(&histt[i], sh_histt[i]);
    }
}

__device__ void
helper_calculate_probability(int* hist, int total_pixels, double* prob)
{
  // in order for this function to perform correctly, total number of threads
  // must equal PIXEL_RANGE
  //  calculate thread id in 1d kernel
  int index = blockDim.x * blockIdx.x + threadIdx.x;

  if (index < 256) {
    prob[index] = (double)hist[index] / total_pixels;
  }
}

__device__ void
helper_buildLook_up_table(double* prob, double* lut)
{
  // shared prob
  __shared__ double sh_prob[256];

  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < 256) {
    sh_prob[index] = prob[index];
  }

  __syncthreads();
  double temp_buffer = 0.0;
  if (index < 256) {
    for (auto j = 0; j <= index; j++) {
      temp_buffer += sh_prob[j] * MAX_PIXEL_VAL;
    }

    lut[index] = temp_buffer;
    }
}

__global__ void
buildLook_up_table_rgb(int* hist_blue,
                       int* hist_green,
                       int* hist_red,
                       int count,
                       bool free_sw,
                       double* lut_final,
                       double* lut_blue,
                       double* lut_green,
                       double* lut_red)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;

  if (index == 0) {
    double* prob_blue = new double[256];
    double* prob_red = new double[256];
    double* prob_green = new double[256];

    calculate_probability<<<1, 256>>>(hist_blue, count, prob_blue);
    calculate_probability<<<1, 256>>>(hist_green, count, prob_green);
    calculate_probability<<<1, 256>>>(hist_red, count, prob_red);
    cudaDeviceSynchronize();

    buildLook_up_table<<<1, 256>>>(prob_blue, lut_blue);
    buildLook_up_table<<<1, 256>>>(prob_green, lut_green);
    buildLook_up_table<<<1, 256>>>(prob_red, lut_red);
    cudaDeviceSynchronize();

    delete[] prob_red;
    delete[] prob_blue;
    delete[] prob_green;
  }

  __syncthreads();
  if (index < 256)
    lut_final[index] =
      (lut_blue[index] + lut_green[index] + lut_red[index]) / 3.0;
}

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
                      short channels_c,
                      int* histt,
                      int sw)
{
  if (x_start < 0) {
    x_start = 0;
  }
  if (x_end > height) {
    x_end = height;
  }

  if (y_start < 0) {
    y_start = 0;
  }
  if (y_end > width) {
    y_end = width;
  }

  int x_range = x_end - x_start;
  int y_range = y_end - y_start;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int x_coverage = gridDim.x * blockDim.x;
  int y_coverage = gridDim.y * blockDim.y;

  int step_x = x_range / x_coverage;
  int step_y = y_range / y_coverage;

  if (step_x < 1) {
    step_x = 1;
  }
  if (step_y < 1) {
    step_y = 1;
  }
  int x_start_ = x * step_x;
  int y_start_ = y * step_y;

  int x_end_ = (x + 1) * step_x;
  int y_end_ = (y + 1) * step_y;

  helper_extract_histogram_rgb(img,
                               count,
                               x_start_,
                               x_end_,
                               y_start_,
                               y_end_,
                               width,
                               height,
                               steps,
                               channel,
                               channels_c,
                               histt,
                               sw);
}
__global__ void
calculate_probability(int* hist, int total_pixels, double* prob)
{
  helper_calculate_probability(hist, total_pixels, prob);
}

__global__ void
buildLook_up_table(double* prob, double* lut)
{
  helper_buildLook_up_table(prob, lut);
}