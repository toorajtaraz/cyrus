#include <helpers.h>
#include <lhe.h>
__device__ void
apply_lhe_helper(uchar* base,
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
apply_lhe(uchar* base,
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
  apply_lhe_helper(base,
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
