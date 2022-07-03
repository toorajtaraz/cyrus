#include <helpers.h>
__global__ void
__extract_histogram(const uchar* img,
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
  extern __shared__ int sh_histt[];

  int index = threadIdx.y * blockDim.y + threadIdx.x;
  // devid 256 by total number of threads in 2d block
  int s = 256 / (blockDim.x * blockDim.y);

  if (index == 0) {
    sh_histt[256] = 0;
  }

  if (s < 1)
    s = 1;

  for (int i = index * s; i < (index * s + 1); i++)
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

  if (index < 256) {
    atomicAdd(&histt[index], sh_histt[index]);
  }
}
__global__ void
__extract_histogram_rgb(const uchar* img,
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
  extern __shared__ int sh_histt[];

  // calculate global id
  int index = threadIdx.y * blockDim.y + threadIdx.x;
  int s = 256 / (blockDim.x * blockDim.y);

  if (index == 0) {
    sh_histt[256] = 0;
  }

  if (s < 1)
    s = 1;

  for (int i = index * s; i < (index * (s + 1)); i++)
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

  if (index < 256) {
    atomicAdd(&histt[index], sh_histt[index]);
  }
}
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
    // printf("%d %d %d\n", hist_blue[45], hist_green[45], hist_red[45]);

    calculate_probability<<<1, 256>>>(hist_blue, count, prob_blue);
    calculate_probability<<<1, 256>>>(hist_green, count, prob_green);
    calculate_probability<<<1, 256>>>(hist_red, count, prob_red);
    // printf("%d %d %d\n", hist_blue[45], hist_green[45], hist_red[45]);
    cudaDeviceSynchronize();

    // printf("%d %d %d %d %d %d\n", prob_blue[45], prob_green[45], prob_red[45,
    // hist_blue[45], hist_green[45], hist_red[45]]);

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

// __device__ void apply_LHE_helper(uchar *base, const uchar *img, dim3
// dimBlock, dim3 dimGrid, int **hists, int *hist, int *count, int *temp, int
// window, int i_start, int i_end, int width, int height, int steps, int
// channels_c)
// {
//     int offset = (int)floor(window / 2.0);
//     int sw = 0;

//     for (int i = i_start; i < i_end; i++)
//     {
//         sw = i % 2 == (i_start % 2) ? 0 : 1;
//         if (sw == 1)
//         {
//             for (int j = width - 1; j >= 0; j--)
//             {
//                 if (j == (width - 1))
//                 {
//                     for (int n = 0; n < window; n++)
//                     {
//                         if (channels_c > 1)
//                         {
//                             for (auto k = 0; k < channels_c; k++)
//                             {
//                                 *temp = *count;
//                                 extract_histogram_rgb<<<1, 1>>>(img, temp, i
//                                 - 1 - offset, i - 1 - offset + 1, j + n -
//                                 offset, j + n - offset + 1, width, height,
//                                 steps, k, 3, hists[k], -1);
//                                 extract_histogram_rgb<<<1, 1>>>(img, temp, i
//                                 + window - 1 - offset, i + window - 1 -
//                                 offset + 1, j + n - offset, j + n - offset +
//                                 1, width, height, steps, k, 3, hists[k], 1);
//                             }
//                             // cudaDeviceSynchronize();

//                             *count = *temp;
//                         }
//                         else
//                         {
//                             extract_histogram<<<1, 1>>>(img, count, i - 1 -
//                             offset, i - 1 - offset + 1, j + n - offset, j + n
//                             - offset + 1, width, height, steps, hist, -1);
//                             extract_histogram<<<1, 1>>>(img, count, i +
//                             window - 1 - offset, i + window - 1 - offset + 1,
//                             j + n - offset, j + n - offset + 1, width,
//                             height, steps, hist, 1);
//                             // cudaDeviceSynchronize();
//                         }
//                     }
//                 }
//                 else if (j < (width - 1))
//                 {
//                     for (int n = 0; n < window; n++)
//                     {
//                         if (channels_c > 1)
//                         {
//                             for (auto k = 0; k < channels_c; k++)
//                             {
//                                 *temp = *count;
//                                 extract_histogram_rgb<<<1, 1>>>(img, temp, i
//                                 + n - offset, i + n - offset + 1, j - offset,
//                                 j - offset + 1, width, height, steps, k, 3,
//                                 hists[k], 1); extract_histogram_rgb<<<1,
//                                 1>>>(img, temp, i + n - offset, i + n -
//                                 offset + 1, j + window - offset, j + window -
//                                 offset + 1, width, height, steps, k, 3,
//                                 hists[k], -1);
//                             }
//                             // cudaDeviceSynchronize();

//                             *count = *temp;
//                         }
//                         else
//                         {
//                             extract_histogram<<<1, 1>>>(img, count, i + n -
//                             offset, i + n - offset + 1, j - offset, j -
//                             offset + 1, width, height, steps, hist, 1);
//                             extract_histogram<<<1, 1>>>(img, count, i + n -
//                             offset, i + n - offset + 1, j + window - offset,
//                             j + window - offset + 1, width, height, steps,
//                             hist, -1);
//                             // cudaDeviceSynchronize();
//                         }
//                     }
//                 }
//                 *count = *count > 0 ? *count : 1;
//                 if (channels_c > 1)
//                 {
//                     double *lut = new double[256];
//                     buildLook_up_table_rgb<<<1, 256>>>(hists[0], hists[1],
//                     hists[2], *count, true, lut);
//                     // cudaDeviceSynchronize();

//                     for (auto k = 0; k < channels_c; k++)
//                     {
//                         // base[j * steps + i * k] = img[j * steps + i * k];
//                         // base.at<cv::Vec3b>(i, j)[k] =
//                         (uchar)lut[img.at<cv::Vec3b>(i, j)[k]];
//                     }
//                     delete[] lut;
//                 }
//                 else
//                 {
//                     double *prob = new double[256];
//                     calculate_probability<<<1, 256>>>(hist, *count, prob);
//                     // cudaDeviceSynchronize();

//                     double *lut = new double[256];
//                     buildLook_up_table<<<1, 256>>>(prob, lut);
//                     // cudaDeviceSynchronize();

//                     // base[j * steps + i] = img[j * steps + i];
//                     // base.at<uchar>(i, j) = (int)floor(lut[img.at<uchar>(i,
//                     j)]);
//                     // Clean memory
//                     delete[] prob;
//                     delete[] lut;
//                 }
//             }
//         }
//         else
//         {
//             for (int j = 0; j < width; j++)
//             {
//                 if (j == 0 && i > i_start)
//                 {
//                     for (int n = 0; n < window; n++)
//                     {
//                         if (channels_c > 1)
//                         {
//                             for (auto k = 0; k < channels_c; k++)
//                             {
//                                 *temp = *count;
//                                 extract_histogram_rgb<<<1, 1>>>(img, temp, i
//                                 - 1 - offset, i - 1 - offset + 1, j + n -
//                                 offset, j + n - offset + 1, width, height,
//                                 steps, k, 3, hists[k], -1);
//                                 extract_histogram_rgb<<<1, 1>>>(img, temp, i
//                                 + window - 1 - offset, i + window - 1 -
//                                 offset + 1, j + n - offset, j + n - offset +
//                                 1, width, height, steps, k, 3, hists[k], 1);
//                             }
//                             // cudaDeviceSynchronize();

//                             *count = *temp;
//                         }
//                         else
//                         {
//                             extract_histogram<<<1, 1>>>(img, count, i - 1 -
//                             offset, i - 1 - offset + 1, j + n - offset, j + n
//                             - offset + 1, width, height, steps, hist, -1);
//                             extract_histogram<<<1, 1>>>(img, count, i +
//                             window - 1 - offset, i + window - 1 - offset + 1,
//                             j + n - offset, j + n - offset + 1, width,
//                             height, steps, hist, 1);
//                             // cudaDeviceSynchronize();
//                         }
//                     }
//                 }
//                 else if (j == 0 && i == i_start)
//                 {
//                     for (int n = 0; n < window; n++)
//                     {
//                         for (int m = 0; m < window; m++)
//                         {
//                             if (channels_c > 1)
//                             {
//                                 for (auto k = 0; k < channels_c; k++)
//                                 {
//                                     *temp = *count;
//                                     extract_histogram_rgb<<<1, 1>>>(img,
//                                     temp, i + n - offset, i + n - offset + 1,
//                                     j + m - offset, j + m - offset + 1,
//                                     width, height, steps, k, 3, hists[k], 1);
//                                 }
//                                 // cudaDeviceSynchronize();

//                                 *count = *temp;
//                             }
//                             else
//                             {
//                                 extract_histogram<<<1, 1>>>(img, count, i + n
//                                 - offset, i + n - offset + 1, j + m - offset,
//                                 j + m - offset + 1, width, height, steps,
//                                 hist, 1);
//                                 // cudaDeviceSynchronize();
//                             }
//                             printf("extractions goes well and count is %d\n",
//                             *count);
//                         }
//                     }
//                 }
//                 else if (j > 0)
//                 {
//                     for (int n = 0; n < window; n++)
//                     {
//                         if (channels_c > 1)
//                         {
//                             for (auto k = 0; k < channels_c; k++)
//                             {
//                                 *temp = *count;
//                                 extract_histogram_rgb<<<1, 1>>>(img, temp, i
//                                 + n - offset, i + n - offset + 1, j - 1 -
//                                 offset, j - 1 - offset + 1, width, height,
//                                 steps, k, 3, hists[k], -1);
//                                 extract_histogram_rgb<<<1, 1>>>(img, temp, i
//                                 + n - offset, i + n - offset + 1, j + window
//                                 - 1 - offset, j + window - 1 - offset + 1,
//                                 width, height, steps, k, 3, hists[k], 1);
//                             }
//                             // cudaDeviceSynchronize();

//                             *count = *temp;
//                         }
//                         else
//                         {
//                             extract_histogram<<<1, 1>>>(img, count, i + n -
//                             offset, i + n - offset + 1, j - 1 - offset, j - 1
//                             - offset + 1, width, height, steps, hist, -1);
//                             extract_histogram<<<1, 1>>>(img, count, i + n -
//                             offset, i + n - offset + 1, j + window - 1 -
//                             offset, j + window - 1 - offset + 1, width,
//                             height, steps, hist, 1);
//                             // cudaDeviceSynchronize();
//                         }
//                     }
//                 }
//                 *count = *count > 0 ? *count : 1;
//                 if (channels_c > 1)
//                 {
//                     double *lut = new double[256];
//                     buildLook_up_table_rgb<<<1, 256>>>(hists[0], hists[1],
//                     hists[2], *count, true, lut);
//                     // cudaDeviceSynchronize();

//                     for (auto k = 0; k < channels_c; k++)
//                     {
//                         // base[j * steps + i * k] = img[j * steps + i * k];
//                         // base.at<cv::Vec3b>(i, j)[k] =
//                         (uchar)lut[img.at<cv::Vec3b>(i, j)[k]];
//                     }
//                     delete[] lut;
//                 }
//                 else
//                 {
//                     double *prob = new double[256];
//                     calculate_probability<<<1, 256>>>(hist, *count, prob);
//                     // cudaDeviceSynchronize();

//                     double *lut = new double[256];
//                     buildLook_up_table<<<1, 256>>>(prob, lut);
//                     // cudaDeviceSynchronize();

//                     // base[j * steps + i] = img[j * steps + i];
//                     // base.at<uchar>(i, j) = (int)floor(lut[img.at<uchar>(i,
//                     j)]);
//                     // Clean memory
//                     delete[] prob;
//                     delete[] lut;
//                 }
//             }
//         }
//     }
// }

// __global__ void apply_LHE(uchar *base, const uchar *img, int window, int
// width, int height, int steps, int channels_c)
// {
//     int total_threads = blockDim.x;
//     int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
//     int _temp = width / total_threads;
//     int i_start = thread_id * _temp;
//     int i_end = (thread_id + 1) * _temp;
//     dim3 dimBlock(32, 32, 1);
//     dim3 dimGrid((window - 1) / 32 + 1, (window - 1) / 32 + 1, 1);
//     int *temp;
//     int *count;
//     int **hists;
//     int *hist;

//     count = new int[1];
//     temp = new int[1];
//     *count = 0;
//     *temp = 0;
//     if (channels_c > 1)
//     {
//         hists = new int *[channels_c];
//         for (auto i = 0; i < channels_c; i++)
//         {
//             hists[i] = new int[PIXEL_RANGE]();
//         }
//     }
//     else
//     {
//         hist = new int[PIXEL_RANGE]();
//     }

//     if (thread_id == (total_threads - 1))
//     {
//         i_end = width;
//     }
//     // printf("pointer address to hists is %p\n", hists);
//     __syncthreads();
//     apply_LHE_helper(base, img, dimBlock, dimGrid, hists, hist, count, temp,
//     window, i_start, i_end, width, height, steps, channels_c);
//     __syncthreads();

//     if (channels_c > 1)
//     {
//         // delete channels
//         for (auto k = 0; k < channels_c; k++)
//         {
//             delete[] hists[k];
//         }
//     }
//     else
//     {
//         delete[] hist;
//     }
// }

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

  // printf("x_start_ %d x_end_ %d y_start_ %d y_end_ %d\n", x_start_, x_end_,
  // y_start_, y_end_);
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
  int* count = new int[1];
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
    i_end = max_i + 1;
  }

  // set i_start to smallest multiple of offset greater than or equal to i_start
  i_start = i_start + (offset - (i_start % offset));
  for (int i = i_start; i < i_end; i += offset) {
    for (int j = 0; j < width; j += offset) {
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
        *count = 0;
        memset(hists[k], 0, PIXEL_RANGE * sizeof(int));
        // print x and y
        extract_histogram_rgb<<<grid, block>>>(img,
                                               count,
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
      // print one of the hists
      cudaDeviceSynchronize();
      // printf("%d %d %d \n", hists[0][45], hists[1][32], hists[2][12]);
      buildLook_up_table_rgb<<<1, 256>>>(hists[2],
                                         hists[1],
                                         hists[0],
                                         *count,
                                         true,
                                         all_luts[i / offset][j / offset],
                                         lut_blue,
                                         lut_green,
                                         lut_red);
      //   for (auto a = 0; a < PIXEL_RANGE; a++) {
      //     printf("%f ", all_luts[i / offset][j / offset][a]);
      //     }
      cudaDeviceSynchronize();
      //   printf("%d \n", all_luts[i / offset][j / offset][45]);
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
