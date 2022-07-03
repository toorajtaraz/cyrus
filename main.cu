#include <iostream>
using namespace std;

#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv::cuda;

#include <cuda.h>
#include <cuda_runtime.h>
// include headers for cudamallocmanaged
#include <cuda_runtime_api.h>
#include <helpers.h>

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

int
main()
{
  // Load image
  cv::Mat src = cv::imread("E:\\university\\Projects\\cyrus\\data\\2.jpg");
  // print image shape
  cout << "src shape" << src.size() << " " << src.channels() << endl;

  // //resize image to 1/3 of its size
  cv::Mat src_resized;
  cv::resize(src, src_resized, cv::Size(), 0.1, 0.1);

  // //copy the resized image to gpu
  cv::cuda::GpuMat src_resized_gpu;
  src_resized_gpu.upload(src_resized);

  // //create a gpu matrix with the same size as the resized image
  cv::cuda::GpuMat dst_gpu;
  dst_gpu.create(src_resized_gpu.size(), src_resized_gpu.type());
    std::cout << "src_resized_gpu.size()" << src_resized_gpu.size() << endl;
  apply_LHE<<<1, 32>>>(dst_gpu.data,
                      src_resized_gpu.data,
                      31,
                      src_resized_gpu.cols,
                      src_resized_gpu.rows,
                      src_resized_gpu.step,
                      src_resized_gpu.channels());
  cudaDeviceSynchronize();

  // //copy the gpu matrix to cpu
  cv::Mat dst;
  dst_gpu.download(dst);

  //show the result
//   cv::imshow("src", src);
//   cv::imshow("src_resized", src_resized);
  cv::imshow("dst", dst);
  cv::waitKey(0);
  return 0;

  /***********************************************************************************/
  int* hist = new int[256]{ 0 };
  double* h_prob = new double[256]{ 0.0 };
  double* h_lut = new double[256]{ 0.0 };
  // measure how long does it take to calculate histogram on cpu
  auto start = std::chrono::high_resolution_clock::now();
  int h_count = 0;
  // compute histogram on cpu
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      int pix = src.at<cv::Vec3b>(i, j)[0];
      hist[pix]++;
      h_count++;
    }
  }

  for (int i = 0; i < 256; i++) {
    h_prob[i] = (double)hist[i] / h_count;
  }

  for (int i = 0; i < 256; i++) {
    for (int j = 0; j <= i; j++) {
      h_lut[i] += h_prob[j] * 255;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  // // print histogram
  // for (int i = 0; i < 256; i++)
  // {
  //     cout << i << " " << hist[i] << endl;
  // }

  // //print probability
  // for (int i = 0; i < 256; i++)
  // {
  //     cout << i << " " << h_prob[i] << endl;
  // }

  // print lut
  for (int i = 0; i < 256; i++) {
    cout << i << " " << h_lut[i] << endl;
  }

  // Upload image to GPU
  cv::cuda::GpuMat d_src;
  d_src.upload(src);
  // create empty gpu matrix with the same size as the image
  cv::cuda::GpuMat d_dst;
  d_dst.create(d_src.size(), d_src.type());

  // Create histogram array for cpu and gpu
  //   int histSize = 256;
  //   int* h_hist = new int[histSize]{ 0 };
  //   int *d_hist_blue, *d_hist_green, *d_hist_red;

  //   double *d_lut_blue, *d_lut_green, *d_lut_red;

  // allocate memory for histogram on gpu and set it to 0
  //   cudaError_t err;

  //   err = cudaMallocManaged(&d_lut_blue, sizeof(double) * 256);
  //   if (err != cudaSuccess) {
  //     cout << "cudaMallocManaged failed: " << cudaGetErrorString(err) <<
  //     endl; return -1;
  //   }

  //   err = cudaMallocManaged(&d_lut_green, sizeof(double) * 256);
  //   if (err != cudaSuccess) {
  //     cout << "cudaMallocManaged failed: " << cudaGetErrorString(err) <<
  //     endl; return -1;
  //   }

  //   err = cudaMallocManaged(&d_lut_red, sizeof(double) * 256);
  //   if (err != cudaSuccess) {
  //     cout << "cudaMallocManaged failed: " << cudaGetErrorString(err) <<
  //     endl; return -1;
  //   }

  //   // err = cudaMalloc((void **)&d_hist, histSize * sizeof(int));
  //   err = cudaMallocManaged((void**)&d_hist_blue, histSize * sizeof(int));
  //   if (err != cudaSuccess) {
  //     printf("Error allocating device memory!\n");
  //     return -1;
  //   }
  //   err = cudaMallocManaged((void**)&d_hist_green, histSize * sizeof(int));
  //   if (err != cudaSuccess) {
  //     printf("Error allocating device memory!\n");
  //     return -1;
  //   }
  //   err = cudaMallocManaged((void**)&d_hist_red, histSize * sizeof(int));
  //   if (err != cudaSuccess) {
  //     printf("Error allocating device memory!\n");
  //     return -1;
  //   }

  //   err = cudaMemset(d_hist_blue, 0, histSize * sizeof(int));
  //   if (err != cudaSuccess) {
  //     printf("Error setting device memory!\n");
  //     return -1;
  //   }

  //   err = cudaMemset(d_hist_green, 0, histSize * sizeof(int));
  //   if (err != cudaSuccess) {
  //     printf("Error setting device memory!\n");
  //     return -1;
  //   }

  //   err = cudaMemset(d_hist_red, 0, histSize * sizeof(int));
  //   if (err != cudaSuccess) {
  //     printf("Error setting device memory!\n");
  //     return -1;
  //   }

  // Create count pointer on cpu and gpu
  //   int *d_count_blue, *d_count_green, *d_count_red;

  // allocate memory for count on gpu and set it to 0
  // err = cudaMalloc((void **)&d_count, sizeof(int));
  //   err = cudaMallocManaged((void**)&d_count_blue, sizeof(int));
  //   if (err != cudaSuccess) {
  //     printf("Error allocating device memory!\n");
  //     return -1;
  //   }

  //   err = cudaMallocManaged((void**)&d_count_green, sizeof(int));
  //   if (err != cudaSuccess) {
  //     printf("Error allocating device memory!\n");
  //     return -1;
  //   }

  //   err = cudaMallocManaged((void**)&d_count_red, sizeof(int));
  //   if (err != cudaSuccess) {
  //     printf("Error allocating device memory!\n");
  //     return -1;
  //   }

  //   err = cudaMemset(d_count_blue, 0, sizeof(int));
  //   if (err != cudaSuccess) {
  //     printf("Error setting device memory!\n");
  //     return -1;
  //   }

  //   err = cudaMemset(d_count_green, 0, sizeof(int));
  //   if (err != cudaSuccess) {
  //     printf("Error setting device memory!\n");
  //     return -1;
  //   }

  //   err = cudaMemset(d_count_red, 0, sizeof(int));
  //   if (err != cudaSuccess) {
  //     printf("Error setting device memory!\n");
  //     return -1;
  //   }

  //   double* d_prob;
  //   err = cudaMallocManaged((void**)&d_prob, 256 * sizeof(double));
  //   if (err != cudaSuccess) {
  //     printf("Error allocating device memory!\n");
  //     return -1;
  //   }

  //   double* d_lut;
  //   err = cudaMallocManaged((void**)&d_lut, 256 * sizeof(double));
  //   if (err != cudaSuccess) {
  //     printf("Error allocating device memory!\n");
  //     return -1;
  //   }

  //   double* d_final_lut;
  //   err = cudaMallocManaged((void**)&d_final_lut, 256 * sizeof(double));
  //   if (err != cudaSuccess) {
  //     printf("Error allocating device memory!\n");
  //     return -1;
  //   }

  // Dynamic programming for histogram
  // offset is floor of half of window size
  // window size is one of these:
  // 151
  // 51
  // dynamic programming array of luts must be (width/offset) x (height/offset)
  int window = 151;
  int offset = (int)floor((double)window / 2.0);
  int width = src.rows;
  int height = src.cols;
  int max_i = height + (offset - (height % offset));
  int max_j = width + (offset - (width % offset));

  double*** d_dp_luts;
  cudaError_t err;
  int x_max = max_i / offset;
  int y_max = max_j / offset;
  dim3 block(32, 32, 1);
  dim3 grid((x_max - 1) / 32 + 1, (y_max - 1) / 32 + 1, 1);
  CUDA_CHECK(
    cudaMallocManaged((void**)&d_dp_luts, sizeof(double**) * (max_i / offset)));
  for (int i = 0; i <= (max_i / offset); i++) {
    CUDA_CHECK(cudaMallocManaged((void**)&(d_dp_luts[i]),
                                 sizeof(double*) * (max_j / offset)));
    for (int j = 0; j <= (max_j / offset); j++) {
      CUDA_CHECK(
        cudaMallocManaged((void**)&(d_dp_luts[i][j]), sizeof(double) * 256));
    }
  }
  //   err =
  //     cudaMallocManaged((void**)&d_dp_luts, sizeof(double**) * (max_i /
  //     offset));
  //   if (err != cudaSuccess) {
  //     printf("Error allocating device memory!\n");
  //     return -1;
  //   }
  //   for (int i = 0; i < (max_i / offset); i++) {
  //     err = cudaMallocManaged((void**)&(d_dp_luts[i]),
  //                             sizeof(double*) * (max_j / offset));
  //     if (err != cudaSuccess) {
  //       printf("Error allocating device memory!\n");
  //       return -1;
  //     }
  //     for (int j = 0; j < (max_j / offset); j++) {
  //       err = cudaMallocManaged((void**)&(d_dp_luts[i][j]), sizeof(double) *
  //       256); if (err != cudaSuccess) {
  //         printf("Error allocating device memory!\n");
  //         return -1;
  //       }
  //     }
  //   }
  // // Launch the kernel with 1 block and 1 thread
  //   dim3 dimBlock(32, 32, 1);
  //   dim3 dimGrid((d_src.cols - 1) / 32 + 1, (d_src.rows - 1) / 32 + 1, 1);

  //   // dim3 dimBlock(16, 16, 1);
  //   // dim3 dimGrid(2, 2, 1);
  //   int block_count = ((d_src.cols - 1) / 32 + 1) * ((d_src.rows - 1) / 32 +
  //   1);
  auto start_gpu = std::chrono::high_resolution_clock::now();
  //   extract_histogram_rgb<<<dimGrid,
  //                           dimBlock,
  //                           (257 * sizeof(int)) + (256 * sizeof(double))>>>(
  //     d_src.data,
  //     d_count_red,
  //     0,
  //     d_src.cols,
  //     0,
  //     d_src.rows,
  //     d_src.rows,
  //     d_src.cols,
  //     d_src.step,
  //     0,
  //     3,
  //     d_hist_red,
  //     1);
  //   extract_histogram_rgb<<<dimGrid,
  //                           dimBlock,
  //                           (257 * sizeof(int)) + (256 * sizeof(double))>>>(
  //     d_src.data,
  //     d_count_green,
  //     1,
  //     d_src.cols,
  //     0,
  //     d_src.rows,
  //     d_src.rows,
  //     d_src.cols,
  //     d_src.step,
  //     1,
  //     3,
  //     d_hist_green,
  //     1);
  //   extract_histogram_rgb<<<dimGrid,
  //                           dimBlock,
  //                           (257 * sizeof(int)) + (256 * sizeof(double))>>>(
  //     d_src.data,
  //     d_count_blue,
  //     2,
  //     d_src.cols,
  //     0,
  //     d_src.rows,
  //     d_src.rows,
  //     d_src.cols,
  //     d_src.step,
  //     2,
  //     3,
  //     d_hist_blue,
  //     1);
  //   // extract_histogram_rgb<<<dimGrid, dimBlock, (257 * sizeof(int)) + (256
  //   *
  //   // sizeof(double)) >>>(d_src.data, d_src.rows, d_src.cols, d_count, 0,
  //   // d_src.cols, 0, d_src.rows, d_src.step, block_count, 1024, 0,
  //   // d_src.channels(), d_hist); wait for gpu to finish
  //   cudaDeviceSynchronize();

  //   // calculate_probability<<<1, 256>>>(d_hist, *d_count, d_prob);
  //   // cudaDeviceSynchronize();

  //   // buildLook_up_table<<<1, 256>>>(d_prob, d_lut);

  //   buildLook_up_table_rgb<<<1, 256>>>(d_hist_blue,
  //                                      d_hist_green,
  //                                      d_hist_red,
  //                                      *d_count_blue,
  //                                      true,
  //                                      d_final_lut,
  //                                      d_lut_blue,
  //                                      d_lut_green,
  //                                      d_lut_red);
  //   cudaDeviceSynchronize();
//   lhe_build_luts<<<1, 4>>>(
//     d_dp_luts, d_src.data, offset, width, height, d_src.channels(), d_src.step);

//   cudaDeviceSynchronize();
//   dim3 dimBlock(32, 32, 1);
//   dim3 dimGrid((d_src.cols * 2) / 32 + 2, (d_src.rows * 2) / 32, 1);
//   apply_interpolating_lhe<<<dimGrid, dimBlock>>>(d_dst.data,
//                                                  d_src.data,
//                                                  151,
//                                                  offset,
//                                                  width,
//                                                  height,
//                                                  d_src.channels(),
//                                                  d_src.step,
//                                                  d_dp_luts);

//   cudaDeviceSynchronize();

  // double ***d_dp_luts_gpu =
  VALIDATE_KERNEL_CALL();
  // std::cout << "max_i: " << max_i << " max_j: " << max_j << std::endl;

  //   double*** u_dp_luts;
  //   u_dp_luts = calculate_luts_for_dynamic_programming(
  //     d_src.data, d_src.rows, d_src.cols, d_src.channels(), d_src.step, 151);
  auto end_gpu = std::chrono::high_resolution_clock::now();
  // Download the histogram from the GPU
  // err = cudaMemcpy(h_hist, d_hist, histSize * sizeof(int),
  // cudaMemcpyDeviceToHost); if (err != cudaSuccess)
  // {
  //     printf("Error copying device memory to host!\n");
  //     return -1;
  // }

  // // Download the count from the GPU
  // err = cudaMemcpy(h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
  // if (err != cudaSuccess)
  // {
  //     printf("Error copying device memory to host!\n");
  //     return -1;
  // }

  // // Print the histogram
  // for (int i = 0; i < histSize; i++)
  // {
  //     cout << d_hist[i] << " ";
  // }

  // //Print the probability
  // for (int i = 0; i < 256; i++)
  // {
  //     cout << d_prob[i] << " ";
  // }

  // Print the lut
  //   for (int i = 0; i < 256; i++) {
  //     cout << d_final_lut[i] << " ";
  //   }

  //   // Print the count
  //   cout << endl << d_count_blue[0] << endl;

  //   print the very first lut at 0, 0

  // for (int i = 0; i < x_max; i++)
  //   for (int j = 0; j < y_max; j++) {
  //     for (int k = 0; k < 256; k++) {
  //       printf("%f ", d_dp_luts[i][j][k]);
  //     }
  //       printf("\n");
  //   }
  printf("x_max: %d y_max: %d\n", x_max, y_max);
  //   for (int a = 0; a < 256; a++) {
  //     cout << d_dp_luts[0][1][a] << " ";
  //   }
  //   // print time taken on cpu and gpu
  cout << "time taken on cpu: "
       << std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count()
       << " ms" << endl;
  cout << "time taken on gpu: "
       << std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu -
                                                                start_gpu)
            .count()
       << " ms" << endl;

  // download d_dst to host memory
  // and then show the result

  cv::Mat dst_host(d_dst.rows, d_dst.cols, CV_8UC3);
  d_dst.download(dst_host);

//   cv::imshow("src", src);
//   cv::imshow("dst", dst_host);

//   cv::waitKey(0);
  // free the allocated mem
  //   delete[] h_hist;
  //   // free cuda mem
  //   cudaFree(d_count_blue);
  //   cudaFree(d_count_green);
  //   cudaFree(d_count_red);
  //   cudaFree(d_hist_blue);
  //   cudaFree(d_hist_green);
  //   cudaFree(d_hist_red);
  //   cudaFree(d_prob);
  //   cudaFree(d_lut);
  //   cudaFree(d_final_lut);
  //   cudaFree(d_lut_blue);
  //   cudaFree(d_lut_green);
  //   cudaFree(d_lut_red);
  return 0;
}