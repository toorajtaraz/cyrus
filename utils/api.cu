#include <api.h>
#include <interpolating_lhe.h>
#include <iostream>
#include <lhe.h>

double*** d_dp_luts = NULL;
long unsigned int N_DP_LUTS_I = 0;
long unsigned int N_DP_LUTS_J = 0;

void
clean_up()
{
  if (d_dp_luts != NULL) {
    for (int i = 0; i < N_DP_LUTS_I; i++) {
      for (int j = 0; j < N_DP_LUTS_J; j++) {
        delete[] d_dp_luts[i][j];
      }
      delete[] d_dp_luts[i];
    }
    delete[] d_dp_luts;
  }
}

cv::Mat
interpolating_lhe_api(cv::Mat src,
                      int window,
                      long long* taken_time_pure,
                      long long* taken_time_total)
{
  auto start = std::chrono::high_resolution_clock::now();
  cv::cuda::GpuMat d_src;
  d_src.upload(src);
  cv::Mat h_result(src.size().height, src.size().width, src.type());
  cv::cuda::GpuMat d_result(src.size().height, src.size().width, src.type());
  int offset = (int)floor((double)window / 2.0);
  int width = src.rows;
  int height = src.cols;
  int max_i = height + (offset - (height % offset));
  int max_j = width + (offset - (width % offset));

  int x_max = max_i / offset;
  int y_max = max_j / offset;
  if (d_dp_luts == NULL) {
    N_DP_LUTS_I = (max_i / offset);
    N_DP_LUTS_J = (max_j / offset);
    CUDA_CHECK(cudaMallocManaged((void**)&d_dp_luts,
                                 sizeof(double**) * (max_i / offset)));
    for (int i = 0; i <= (max_i / offset); i++) {
      CUDA_CHECK(cudaMallocManaged((void**)&(d_dp_luts[i]),
                                   sizeof(double*) * (max_j / offset)));
      for (int j = 0; j <= (max_j / offset); j++) {
        CUDA_CHECK(
          cudaMallocManaged((void**)&(d_dp_luts[i][j]), sizeof(double) * 256));
      }
    }
  }
  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid((d_src.cols * 2) / 32 + 2, (d_src.rows * 2) / 32, 1);
  auto start_gpu = std::chrono::high_resolution_clock::now();

  lhe_build_luts<<<1, 4>>>(
    d_dp_luts, d_src.data, offset, width, height, d_src.channels(), d_src.step);

  cudaDeviceSynchronize();

  apply_interpolating_lhe<<<dimGrid, dimBlock>>>(d_result.data,
                                                 d_src.data,
                                                 window,
                                                 offset,
                                                 width,
                                                 height,
                                                 d_src.channels(),
                                                 d_src.step,
                                                 d_dp_luts);

  cudaDeviceSynchronize();
  auto end_gpu = std::chrono::high_resolution_clock::now();
  *taken_time_pure =
    std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu)
      .count();
  d_result.download(h_result);
  // free gpumat
  d_src.release();
  d_result.release();
  auto end = std::chrono::high_resolution_clock::now();
  *taken_time_total =
    std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  return h_result;
}

cv::Mat
lhe_api(cv::Mat src, int window, long long* taken_time)
{
  cv::cuda::GpuMat d_src;
  d_src.upload(src);
  cv::Mat h_result(src.size().height, src.size().width, src.type());
  cv::cuda::GpuMat d_result(src.size().height, src.size().width, src.type());
  auto start_gpu = std::chrono::high_resolution_clock::now();

  apply_lhe<<<1, 64>>>(d_result.data,
                       d_src.data,
                       window,
                       d_src.cols,
                       d_src.rows,
                       d_src.step,
                       d_src.channels());
  cudaDeviceSynchronize();
  auto end_gpu = std::chrono::high_resolution_clock::now();
  *taken_time =
    std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu)
      .count();
  d_result.download(h_result);

  // free gpumat
  d_src.release();
  d_result.release();
  return h_result;
}