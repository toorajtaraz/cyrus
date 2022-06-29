#include <iostream>
using namespace std;

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/cudaarithm.hpp>
using namespace cv::cuda;

#include <cuda.h>
#include <cuda_runtime.h>
// include headers for cudamallocmanaged
#include <cuda_runtime_api.h>

#include <helpers.h>

int main()
{
    // Load image
    cv::Mat src = cv::imread("E:\\university\\Projects\\cyrus\\data\\1.jpg");
    // print image shape
    cout << "src shape" << src.size() << " " << src.channels() << endl;
    int *hist = new int[256]{0};
    double *h_prob = new double[256]{0.0};
    double *h_lut = new double[256]{0.0};
    // measure how long does it take to calculate histogram on cpu
    auto start = std::chrono::high_resolution_clock::now();
    int h_count = 0;
    // compute histogram on cpu
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            int pix = src.at<cv::Vec3b>(i, j)[0];
            hist[pix]++;
            h_count++;
        }
    }

    for (int i = 0; i < 256; i++)
    {
        h_prob[i] = (double)hist[i] / h_count;
    }

    for (int i = 0; i < 256; i++)
    {
        for (int j = 0; j <= i; j++)
        {
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

    //print lut
    for (int i = 0; i < 256; i++)
    {
        cout << i << " " << h_lut[i] << endl;
    }

    // Upload image to GPU
    cv::cuda::GpuMat d_src;
    d_src.upload(src);

    // Create histogram array for cpu and gpu
    int histSize = 256;
    int *h_hist = new int[histSize]{0};
    int *d_hist_blue, *d_hist_green, *d_hist_red;

    double *d_lut_blue, *d_lut_green, *d_lut_red;

    // allocate memory for histogram on gpu and set it to 0
    cudaError_t err;

    err = cudaMallocManaged(&d_lut_blue, sizeof(double) * 256);
    if (err != cudaSuccess)
    {
        cout << "cudaMallocManaged failed: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    err = cudaMallocManaged(&d_lut_green, sizeof(double) * 256);
    if (err != cudaSuccess)
    {
        cout << "cudaMallocManaged failed: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    err = cudaMallocManaged(&d_lut_red, sizeof(double) * 256);
    if (err != cudaSuccess)
    {
        cout << "cudaMallocManaged failed: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    // err = cudaMalloc((void **)&d_hist, histSize * sizeof(int));
    err = cudaMallocManaged((void **)&d_hist_blue, histSize * sizeof(int));
    if (err != cudaSuccess)
    {
        printf("Error allocating device memory!\n");
        return -1;
    }
    err = cudaMallocManaged((void **)&d_hist_green, histSize * sizeof(int));
    if (err != cudaSuccess)
    {
        printf("Error allocating device memory!\n");
        return -1;
    }
    err = cudaMallocManaged((void **)&d_hist_red, histSize * sizeof(int));
    if (err != cudaSuccess)
    {
        printf("Error allocating device memory!\n");
        return -1;
    }

    err = cudaMemset(d_hist_blue, 0, histSize * sizeof(int));
    if (err != cudaSuccess)
    {
        printf("Error setting device memory!\n");
        return -1;
    }

    err = cudaMemset(d_hist_green, 0, histSize * sizeof(int));
    if (err != cudaSuccess)
    {
        printf("Error setting device memory!\n");
        return -1;
    }

    err = cudaMemset(d_hist_red, 0, histSize * sizeof(int));
    if (err != cudaSuccess)
    {
        printf("Error setting device memory!\n");
        return -1;
    }

    // Create count pointer on cpu and gpu
    int *d_count_blue, *d_count_green, *d_count_red;

    // allocate memory for count on gpu and set it to 0
    // err = cudaMalloc((void **)&d_count, sizeof(int));
    err = cudaMallocManaged((void **)&d_count_blue, sizeof(int));
    if (err != cudaSuccess)
    {
        printf("Error allocating device memory!\n");
        return -1;
    }

    err = cudaMallocManaged((void **)&d_count_green, sizeof(int));
    if (err != cudaSuccess)
    {
        printf("Error allocating device memory!\n");
        return -1;
    }

    err = cudaMallocManaged((void **)&d_count_red, sizeof(int));
    if (err != cudaSuccess)
    {
        printf("Error allocating device memory!\n");
        return -1;
    }

    err = cudaMemset(d_count_blue, 0, sizeof(int));
    if (err != cudaSuccess)
    {
        printf("Error setting device memory!\n");
        return -1;
    }

    err = cudaMemset(d_count_green, 0, sizeof(int));
    if (err != cudaSuccess)
    {
        printf("Error setting device memory!\n");
        return -1;
    }

    err = cudaMemset(d_count_red, 0, sizeof(int));
    if (err != cudaSuccess)
    {
        printf("Error setting device memory!\n");
        return -1;
    }

    double *d_prob;
    err = cudaMallocManaged((void **)&d_prob, 256 * sizeof(double));
    if (err != cudaSuccess)
    {
        printf("Error allocating device memory!\n");
        return -1;
    }

    double *d_lut;
    err = cudaMallocManaged((void **)&d_lut, 256 * sizeof(double));
    if (err != cudaSuccess)
    {
        printf("Error allocating device memory!\n");
        return -1;
    }

    double *d_final_lut;
    err = cudaMallocManaged((void **)&d_final_lut, 256 * sizeof(double));
    if (err != cudaSuccess)
    {
        printf("Error allocating device memory!\n");
        return -1;
    }
    // // Launch the kernel with 1 block and 1 thread
    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid((d_src.cols - 1) / 32 + 1, (d_src.rows - 1) / 32 + 1, 1);

    // dim3 dimBlock(16, 16, 1);
    // dim3 dimGrid(2, 2, 1);
    int block_count = ((d_src.cols - 1) / 32 + 1) * ((d_src.rows - 1) / 32 + 1);
    auto start_gpu = std::chrono::high_resolution_clock::now();
    extract_histogram_rgb<<<dimGrid, dimBlock, (257 * sizeof(int)) + (256 * sizeof(double))>>>(d_src.data, d_count_red, 0, d_src.cols, 0, d_src.rows, d_src.rows, d_src.cols, d_src.step, 0, 3, d_hist_red, 1);
    extract_histogram_rgb<<<dimGrid, dimBlock, (257 * sizeof(int)) + (256 * sizeof(double))>>>(d_src.data, d_count_green, 1, d_src.cols, 0, d_src.rows, d_src.rows, d_src.cols, d_src.step, 1, 3, d_hist_green, 1);
    extract_histogram_rgb<<<dimGrid, dimBlock, (257 * sizeof(int)) + (256 * sizeof(double))>>>(d_src.data, d_count_blue, 2, d_src.cols, 0, d_src.rows, d_src.rows, d_src.cols, d_src.step, 2, 3, d_hist_blue, 1);
    // extract_histogram_rgb<<<dimGrid, dimBlock, (257 * sizeof(int)) + (256 * sizeof(double)) >>>(d_src.data, d_src.rows, d_src.cols, d_count, 0, d_src.cols, 0, d_src.rows, d_src.step, block_count, 1024, 0, d_src.channels(), d_hist);
    // wait for gpu to finish
    cudaDeviceSynchronize();

    // calculate_probability<<<1, 256>>>(d_hist, *d_count, d_prob);
    // cudaDeviceSynchronize();

    // buildLook_up_table<<<1, 256>>>(d_prob, d_lut);

    buildLook_up_table_rgb<<<1, 256>>>(d_hist_blue, d_hist_green, d_hist_red, *d_count_blue, true, d_final_lut, d_lut_blue, d_lut_green, d_lut_red);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    // Download the histogram from the GPU
    // err = cudaMemcpy(h_hist, d_hist, histSize * sizeof(int), cudaMemcpyDeviceToHost);
    // if (err != cudaSuccess)
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

    //Print the lut
    for (int i = 0; i < 256; i++)
    {
        cout << d_final_lut[i] << " ";
    }

    // Print the count
    cout << endl
         << d_count_blue[0] << endl;

    // print time taken on cpu and gpu
    cout << "time taken on cpu: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << endl;
    cout << "time taken on gpu: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count() << " ms" << endl;
    // free the allocated mem
    delete[] h_hist;
    // free cuda mem
    cudaFree(d_count_blue);
    cudaFree(d_count_green);
    cudaFree(d_count_red);
    cudaFree(d_hist_blue);
    cudaFree(d_hist_green);
    cudaFree(d_hist_red);
    cudaFree(d_prob);
    cudaFree(d_lut);
    cudaFree(d_final_lut);
    cudaFree(d_lut_blue);
    cudaFree(d_lut_green);
    cudaFree(d_lut_red);
    return 0;
}