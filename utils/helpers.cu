#include <helpers.h>

__device__ void
helper_extract_histogram(const uchar* img,
                         int* count,
                         int x_start,
                         int x_end,
                         int y_start,
                         int y_end,
                         int width,
                         int height,
                         int steps,
                         int* histt,
                         int sw)
{
  __shared__ int sh_histt[257];

  int index = threadIdx.y * blockDim.y + threadIdx.x;
  // devid 256 by total number of threads in 2d block
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

  if (x_start < height && y_start < width && x_start >= 0 && y_start >= 0) {

    atomicAdd(&sh_histt[256], sw);
    atomicAdd(&sh_histt[img[y_start * steps + x_start]], sw);
  }

  __syncthreads();
  if (index == 0) {
    atomicAdd(&count[0], sh_histt[256]);
  }

  for (int i = index * s; i < ((index + 1) * s); i++)
    if (index < 256) {
      atomicAdd(&histt[i], sh_histt[i]);
    }
}

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
  // shared lut
  __shared__ double sh_lut[256];

  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < 256) {
    sh_lut[index] = 0.0;
  }

  __syncthreads();
  if (index < 256)
    for (auto j = 0; j <= index; j++) {
      sh_lut[index] += prob[j] * MAX_PIXEL_VAL;
    }
  __syncthreads();

  if (index < 256) {
    lut[index] = sh_lut[index];
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

__device__ void
apply_LHE_helper(uchar* base,
                 const uchar* img,
                 dim3 dimBlock,
                 dim3 dimGrid,
                 int** hists,
                 int* hist,
                 int* count,
                 int* temp,
                 int window,
                 int i_start,
                 int i_end,
                 int width,
                 int height,
                 int steps,
                 int channels_c)
{

  int offset = (int)floor(window / 2.0);
  int sw = 0;
  double** lut = new double*[channels_c + 1];
  for (int i = 0; i <= channels_c; i++) {
    lut[i] = new double[256];
  }

  for (int i = i_start; i < i_end; i++) {
    sw = i % 2 == (i_start % 2) ? 0 : 1;
    if (sw == 1) {
      for (int j = width - 1; j >= 0; j--) {
        if (j == (width - 1)) {
          for (auto k = 0; k < channels_c; k++) {
            *temp = *count;
            extract_histogram_rgb<<<1, window>>>(img,
                                                 temp,
                                                 i - 1 - offset,
                                                 i - 1 - offset + 1,
                                                 j - offset,
                                                 j + offset,
                                                 width,
                                                 height,
                                                 steps,
                                                 k,
                                                 3,
                                                 hists[k],
                                                 -1);
            cudaDeviceSynchronize();

            extract_histogram_rgb<<<1, window>>>(img,
                                                 temp,
                                                 i + window - 1 - offset,
                                                 i + window - 1 - offset + 1,
                                                 j - offset,
                                                 j + offset,
                                                 width,
                                                 height,
                                                 steps,
                                                 k,
                                                 3,
                                                 hists[k],
                                                 1);
            cudaDeviceSynchronize();
          }

          *count = *temp;

        } else if (j < (width - 1)) {

          for (auto k = 0; k < channels_c; k++) {
            *temp = *count;
            extract_histogram_rgb<<<1, window>>>(img,
                                                 temp,
                                                 i - offset,
                                                 i + offset,
                                                 j - offset,
                                                 j - offset + 1,
                                                 width,
                                                 height,
                                                 steps,
                                                 k,
                                                 3,
                                                 hists[k],
                                                 1);
            cudaDeviceSynchronize();

            extract_histogram_rgb<<<1, window>>>(img,
                                                 temp,
                                                 i - offset,
                                                 i + offset,
                                                 j + window - offset,
                                                 j + window - offset + 1,
                                                 width,
                                                 height,
                                                 steps,
                                                 k,
                                                 3,
                                                 hists[k],
                                                 -1);
            cudaDeviceSynchronize();
          }

          *count = *temp;
        }
        *count = *count > 0 ? *count : 1;

        buildLook_up_table_rgb<<<1, 256>>>(hists[0],
                                           hists[1],
                                           hists[2],
                                           *count,
                                           true,
                                           lut[3],
                                           lut[0],
                                           lut[1],
                                           lut[2]);
        cudaDeviceSynchronize();

        for (auto k = 0; k < channels_c; k++) {
          base[j * steps + i * channels_c + k] =
            (uchar)floor(lut[3][img[j * steps + i * channels_c + k]]);
        }
      }
    } else {
      for (int j = 0; j < width; j++) {
        if (j == 0 && i > i_start) {

          for (auto k = 0; k < channels_c; k++) {
            *temp = *count;
            extract_histogram_rgb<<<1, window>>>(img,
                                                 temp,
                                                 i - 1 - offset,
                                                 i - 1 - offset + 1,
                                                 j - offset,
                                                 j + offset,
                                                 width,
                                                 height,
                                                 steps,
                                                 k,
                                                 3,
                                                 hists[k],
                                                 -1);
            cudaDeviceSynchronize();

            extract_histogram_rgb<<<1, window>>>(img,
                                                 temp,
                                                 i + window - 1 - offset,
                                                 i + window - 1 - offset + 1,
                                                 j - offset,
                                                 j + offset,
                                                 width,
                                                 height,
                                                 steps,
                                                 k,
                                                 3,
                                                 hists[k],
                                                 1);
            cudaDeviceSynchronize();
          }

          *count = *temp;

        } else if (j == 0 && i == i_start) {

          for (auto k = 0; k < channels_c; k++) {
            *temp = *count;
            extract_histogram_rgb<<<window, window>>>(img,
                                                      temp,
                                                      i - offset,
                                                      i + offset,
                                                      j - offset,
                                                      j + offset,
                                                      width,
                                                      height,
                                                      steps,
                                                      k,
                                                      3,
                                                      hists[k],
                                                      1);
            cudaDeviceSynchronize();
          }

          *count = *temp;

        } else if (j > 0) {
          for (int n = 0; n < window; n++) {

            for (auto k = 0; k < channels_c; k++) {
              *temp = *count;
              extract_histogram_rgb<<<1, window>>>(img,
                                                   temp,
                                                   i - offset,
                                                   i + offset,
                                                   j - 1 - offset,
                                                   j - 1 - offset + 1,
                                                   width,
                                                   height,
                                                   steps,
                                                   k,
                                                   3,
                                                   hists[k],
                                                   -1);
              cudaDeviceSynchronize();

              extract_histogram_rgb<<<1, window>>>(img,
                                                   temp,
                                                   i - offset,
                                                   i + offset,
                                                   j + window - 1 - offset,
                                                   j + window - 1 - offset + 1,
                                                   width,
                                                   height,
                                                   steps,
                                                   k,
                                                   3,
                                                   hists[k],
                                                   1);
              cudaDeviceSynchronize();
            }

            *count = *temp;
          }
        }
        *count = *count > 0 ? *count : 1;

        buildLook_up_table_rgb<<<1, 256>>>(hists[0],
                                           hists[1],
                                           hists[2],
                                           *count,
                                           true,
                                           lut[3],
                                           lut[0],
                                           lut[1],
                                           lut[2]);
        cudaDeviceSynchronize();

        for (auto k = 0; k < channels_c; k++) {
          base[j * steps + i * channels_c + k] =
            (uchar)floor(lut[3][img[j * steps + i * channels_c + k]]);
        }
      }
    }
  }
  for (auto k = 0; k <= channels_c; k++) {
    delete[] lut[k];
  }
  delete[] lut;
}

__global__ void
apply_LHE(uchar* base,
          const uchar* img,
          int window,
          int width,
          int height,
          int steps,
          int channels_c)
{
  int total_threads = blockDim.x;
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  int _temp = width / total_threads;
  int i_start = thread_id * _temp;
  int i_end = (thread_id + 1) * _temp;

  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid((window - 1) / 32 + 1, (window - 1) / 32 + 1, 1);
  int* temp;
  int* count;
  int** hists;
  int* hist;

  count = new int[1];
  temp = new int[1];
  *count = 0;
  *temp = 0;
  hists = new int*[channels_c];
  for (auto i = 0; i < channels_c; i++) {
    hists[i] = new int[PIXEL_RANGE]();
    memset(hists[i], 0, PIXEL_RANGE * sizeof(int));
  }

  if (thread_id == (total_threads - 1)) {
    i_end = width;
  }

  __syncthreads();
  apply_LHE_helper(base,
                   img,
                   dimBlock,
                   dimGrid,
                   hists,
                   NULL,
                   count,
                   temp,
                   window,
                   i_start,
                   i_end,
                   width,
                   height,
                   steps,
                   channels_c);
  __syncthreads();

  for (auto k = 0; k < channels_c; k++) {
    delete[] hists[k];
  }
  delete[] hists;
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
extract_histogram(const uchar* img,
                  int* count,
                  int x_start,
                  int x_end,
                  int y_start,
                  int y_end,
                  int width,
                  int height,
                  int steps,
                  int* histt,
                  int sww)
{
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

  helper_extract_histogram(img,
                           count,
                           x * step_x,
                           (x + 1) * step_x,
                           y * step_y,
                           (y + 1) * step_y,
                           width,
                           height,
                           steps,
                           histt,
                           sww);
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

__global__ void
lhe_build_luts(double*** all_luts,
               const uchar* img,
               int offset,
               int width,
               int height,
               int channel_c,
               int steps)
{
  int** hists = new int*[channel_c];
  int* count = new int[3];
  for (int i = 0; i < channel_c; i++) {
    hists[i] = new int[PIXEL_RANGE]();
  }
  double *lut_blue, *lut_green, *lut_red;
  lut_blue = new double[PIXEL_RANGE]();
  lut_green = new double[PIXEL_RANGE]();
  lut_red = new double[PIXEL_RANGE]();

  int total_threads = blockDim.x;
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

  int max_i = height + (offset - (height % offset));
  int max_j = width + (offset - (width % offset));

  int local_step = max_i / total_threads;
  int i_start = thread_id * local_step;
  int i_end = (thread_id + 1) * local_step;
  if (thread_id == (total_threads - 1)) {
    i_end = max_i;
  }

  // set i_start to smallest multiple of offset greater than or equal to i_start
  if (i_start % offset != 0) {
    i_start = i_start + offset - (i_start % offset);
  }
  for (int i = i_start; i <= i_end; i += offset) {
    for (int j = 0; j < max_j; j += offset) {
      dim3 block(32, 32, 1);
      dim3 grid((2 * offset - 1) / 32 + 1, (2 * offset - 1) / 32 + 1, 1);

      int i_start_t, i_end_t, j_start, j_end;
      i_start_t = i - offset;
      i_end_t = i + offset;
      j_start = j - offset;
      j_end = j + offset;
      if (i_start < 0) {
        i_start = 0;
      }
      if (i_end > height) {
        i_end = height;
      }
      if (j_start < 0) {
        j_start = 0;
      }
      if (j_end > width) {
        j_end = width;
      }
      for (auto k = 0; k < channel_c; k++) {
        count[k] = 0;
        memset(hists[k], 0, PIXEL_RANGE * sizeof(int));
        // print x and y
        extract_histogram_rgb<<<grid, block>>>(img,
                                               &count[k],
                                               i_start_t,
                                               i_end_t,
                                               j_start,
                                               j_end,
                                               width,
                                               height,
                                               steps,
                                               k,
                                               channel_c,
                                               hists[k],
                                               1);
      }
      cudaDeviceSynchronize();
      buildLook_up_table_rgb<<<1, 256>>>(hists[2],
                                         hists[1],
                                         hists[0],
                                         *count,
                                         true,
                                         all_luts[i / offset][j / offset],
                                         lut_blue,
                                         lut_green,
                                         lut_red);

      cudaDeviceSynchronize();
    }
  }

  for (auto i = 0; i < channel_c; i++) {
    delete[] hists[i];
  }
  delete[] hists;
  delete[] count;
  delete[] lut_blue;
  delete[] lut_green;
  delete[] lut_red;
}

__global__ void
apply_interpolating_lhe(uchar* base,
                        const uchar* img,
                        int window,
                        int offset,
                        int width,
                        int height,
                        int channel_n,
                        int steps,
                        double*** all_luts)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int x1 = i - (i % offset);
  int y1 = j - (j % offset);
  int x2 = x1 + offset;
  int y2 = y1 + offset;
  if (i >= height || j >= width) {
    return;
  }

  double x1_weight = (double)(i - x1) / (double)(x2 - x1);
  double y1_weight = (double)(j - y1) / (double)(y2 - y1);
  double x2_weight = 1.0 - x1_weight;
  double y2_weight = 1.0 - y1_weight;

  int x1_lut = x1 / offset;
  int y1_lut = y1 / offset;
  int x2_lut = x2 / offset;
  int y2_lut = y2 / offset;
#define upper_left_lut all_luts[x1_lut][y1_lut]
#define upper_right_lut all_luts[x1_lut][y2_lut]
#define lower_left_lut all_luts[x2_lut][y1_lut]
#define lower_right_lut all_luts[x2_lut][y2_lut]

  for (auto k = 0; k < channel_n; k++) {
    uchar temp_pixel = img[j * steps + i * channel_n + k];
    uchar new_value = (uchar)floor(
      (float)(upper_left_lut[temp_pixel] * x2_weight * y2_weight +
              upper_right_lut[temp_pixel] * x2_weight * y1_weight +
              lower_left_lut[temp_pixel] * x1_weight * y2_weight +
              lower_right_lut[temp_pixel] * x1_weight * y1_weight));
    base[j * steps + i * channel_n + k] = new_value;
  }
}
