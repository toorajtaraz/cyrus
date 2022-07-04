#include <helpers.h>
#include <interpolating_lhe.h>


__host__ double***
calculate_luts_for_dynamic_programming(const uchar* img,
                                       int width,
                                       int height,
                                       int channels_c,
                                       int steps,
                                       int window)
{
  int *u_hist_red, *u_hist_green, *u_hist_blue;
  double *u_lut_red, *u_lut_green, *u_lut_blue;
  int *u_count_red, *u_count_green, *u_count_blue;
  double*** unified_mem_luts;
  // allocate unifed memory to hists
  CUDA_CHECK(cudaMallocManaged((void**)&u_hist_red, sizeof(int) * 256));
  CUDA_CHECK(cudaMallocManaged((void**)&u_hist_green, sizeof(int) * 256));
  CUDA_CHECK(cudaMallocManaged((void**)&u_hist_blue, sizeof(int) * 256));

  // allocate unified memory for counts
  CUDA_CHECK(cudaMallocManaged((void**)&u_count_red, sizeof(int)));
  CUDA_CHECK(cudaMallocManaged((void**)&u_count_green, sizeof(int)));
  CUDA_CHECK(cudaMallocManaged((void**)&u_count_blue, sizeof(int)));

  // allocate unified memory for luts
  CUDA_CHECK(cudaMallocManaged((void**)&u_lut_red, sizeof(double) * 256));
  CUDA_CHECK(cudaMallocManaged((void**)&u_lut_green, sizeof(double) * 256));
  CUDA_CHECK(cudaMallocManaged((void**)&u_lut_blue, sizeof(double) * 256));

  // allocate unified memory for luts
  int offset = (int)floor((double)window / 2.0);

  int max_i = height + (offset - (height % offset));
  int max_j = width + (offset - (width % offset));

  CUDA_CHECK(cudaMallocManaged((void**)&unified_mem_luts,
                               sizeof(double**) * (max_i / offset)));
  for (int i = 0; i < (max_i / offset); i++) {
    CUDA_CHECK(cudaMallocManaged((void**)&(unified_mem_luts[i]),
                                 sizeof(double*) * (max_j / offset)));
    for (int j = 0; j < (max_j / offset); j++) {
      CUDA_CHECK(cudaMallocManaged((void**)&(unified_mem_luts[i][j]),
                                   sizeof(double) * 256));
    }
  }
  dim3 block(32, 32, 1);
  dim3 grid((2 * offset - 1) / 32 + 1, (2 * offset - 1) / 32 + 1, 1);

  for (int i = 0; i < height; i += offset) {
    // if (i % offset == 0) {
    for (int j = 0; j < width; j += offset) {
      ZERO_OUT_RGB(u_hist_red, u_hist_green, u_hist_blue);
      ZERO_OUT_COUNTS(u_count_red, u_count_green, u_count_blue);
      int i_start, i_end, j_start, j_end;
      i_start = i - offset;
      i_end = i + offset;
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

      extract_histogram_rgb<<<grid, block>>>(img,
                                             u_count_red,
                                             i_start,
                                             i_end,
                                             j_start,
                                             j_end,
                                             width,
                                             height,
                                             steps,
                                             0,
                                             3,
                                             u_hist_red,
                                             1);
      extract_histogram_rgb<<<grid, block>>>(img,
                                             u_count_green,
                                             i_start,
                                             i_end,
                                             j_start,
                                             j_end,
                                             width,
                                             height,
                                             steps,
                                             1,
                                             3,
                                             u_hist_green,
                                             1);
      extract_histogram_rgb<<<grid, block>>>(img,
                                             u_count_blue,
                                             i_start,
                                             i_end,
                                             j_start,
                                             j_end,
                                             width,
                                             height,
                                             steps,
                                             2,
                                             3,
                                             u_hist_blue,
                                             1);
      cudaDeviceSynchronize();
      VALIDATE_KERNEL_CALL();
      buildLook_up_table_rgb<<<1, 256>>>(
        u_hist_blue,
        u_hist_green,
        u_hist_red,
        *u_count_blue,
        true,
        unified_mem_luts[i / offset][j / offset],
        u_lut_blue,
        u_lut_green,
        u_lut_red);
      cudaDeviceSynchronize();
      VALIDATE_KERNEL_CALL();
    }
    // }
  }

  cudaFree(u_hist_red);
  cudaFree(u_hist_green);
  cudaFree(u_hist_blue);
  cudaFree(u_count_red);
  cudaFree(u_count_green);
  cudaFree(u_count_blue);
  cudaFree(u_lut_red);
  cudaFree(u_lut_green);
  cudaFree(u_lut_blue);
  return unified_mem_luts;
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
