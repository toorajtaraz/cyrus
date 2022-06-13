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
    //print image shape
    cout << "src shape" << src.size() << " " << src.channels() << endl;
    int *hist = new int[256]{0};
    //measure how long does it take to calculate histogram on cpu
    auto start = std::chrono::high_resolution_clock::now();
    //compute histogram on cpu
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            int pix = src.at<cv::Vec3b>(i, j)[0];
            hist[pix]++;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    //print histogram
    for (int i = 0; i < 256; i++)
    {
        cout << i << " " << hist[i] << endl;
    }
    // Upload image to GPU
    cv::cuda::GpuMat d_src;
    d_src.upload(src);

    // Create histogram array for cpu and gpu
    int histSize = 256;
    int *h_hist = new int[histSize]{0};
    int *d_hist;

    // allocate memory for histogram on gpu and set it to 0
    cudaError_t err;

    err = cudaMalloc((void **)&d_hist, histSize * sizeof(int));
    if (err != cudaSuccess)
    {
        printf("Error allocating device memory!\n");
        return -1;
    }

    err = cudaMemset(d_hist, 0, histSize * sizeof(int));
    if (err != cudaSuccess)
    {
        printf("Error setting device memory!\n");
        return -1;
    }

    // Create count pointer on cpu and gpu
    int *h_count = new int[1]{0};
    int *d_count;

    // allocate memory for count on gpu and set it to 0
    err = cudaMalloc((void **)&d_count, sizeof(int));
    if (err != cudaSuccess)
    {
        printf("Error allocating device memory!\n");
        return -1;
    }

    err = cudaMemset(d_count, 0, sizeof(int));
    if (err != cudaSuccess)
    {
        printf("Error setting device memory!\n");
        return -1;
    }

    // Launch the kernel with 1 block and 1 thread
    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid((d_src.cols - 1) / 32 + 1, (d_src.rows - 1) / 32 + 1, 1);
    int block_count = ((d_src.cols - 1) / 32 + 1) * ((d_src.rows - 1) / 32 + 1);
    auto start_gpu = std::chrono::high_resolution_clock::now();
    test_histogram<<<dimGrid, dimBlock, 257*sizeof(int)>>>(d_src.data, d_src.rows, d_src.cols, d_count, 0, d_src.cols, 0, d_src.rows, d_src.step, block_count, 1024, 0, d_src.channels(), d_hist);
    // wait for gpu to finish
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    // Download the histogram from the GPU
    err = cudaMemcpy(h_hist, d_hist, histSize * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("Error copying device memory to host!\n");
        return -1;
    }

    // Download the count from the GPU
    err = cudaMemcpy(h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("Error copying device memory to host!\n");
        return -1;
    }

    // Print the histogram
    for (int i = 0; i < histSize; i++)
    {
        cout << h_hist[i] << " ";
    }

    // Print the count
    cout << endl
         << h_count[0] << endl;


    //print time taken on cpu and gpu
    cout << "time taken on cpu: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << endl;
    cout << "time taken on gpu: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count() << " ms" << endl;
    return 0;
}