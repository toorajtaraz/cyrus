#include <iostream>
using namespace std;

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


#include <opencv2/cudaarithm.hpp>
using namespace cv::cuda;

#include <cuda.h>
#include <cuda_runtime.h>

#include <helpers.h>

//add two int arrays into one array on gpu
__device__
void add_int_arrays(int *d_arr1, int *d_arr2, int *d_arr3, int size)
{
    for (int i = 0; i < size; i++)
    {
        d_arr3[i] = d_arr1[i] + d_arr2[i];
    }
}

//kernel for addition
__global__
void add_int_arrays_kernel(int *d_arr1, int *d_arr2, int *d_arr3, int size)
{
    add_int_arrays(d_arr1, d_arr2, d_arr3, size);
}

int main()
{
    // //create two arrays on host
    int size = 10;
    int *h_arr1 = new int[size];
    int *h_arr2 = new int[size];
    int *h_arr3 = new int[size];

    //fill arrays with random values
    for (int i = 0; i < size; i++)
    {
        h_arr1[i] = rand() % 10;
        h_arr2[i] = rand() % 10;
        cout << "arr 1 = " << h_arr1[i] << endl;
        cout << "arr 2 = " << h_arr2[i] << endl;
    }

    //create two arrays on gpu
    int *d_arr1;
    int *d_arr2;
    int *d_arr3;
    cudaMalloc(&d_arr1, size * sizeof(int));
    cudaMalloc(&d_arr2, size * sizeof(int));

    //copy arrays to gpu
    cudaMemcpy(d_arr1, h_arr1, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr2, h_arr2, size * sizeof(int), cudaMemcpyHostToDevice);

    //create array on gpu for result
    cudaMalloc(&d_arr3, size * sizeof(int));

    //call kernel
    add_int_arrays_kernel<<<1, 1>>>(d_arr1, d_arr2, d_arr3, size);

    //copy result from gpu to host
    cudaMemcpy(h_arr3, d_arr3, size * sizeof(int), cudaMemcpyDeviceToHost);

    //print result
    for (int i = 0; i < size; i++)
    {
        cout << h_arr3[i] << " ";
    }

    printShortCudaDeviceInfo(getDevice());
    int cuda_devices_number = getCudaEnabledDeviceCount();
    cout << "CUDA Device(s) Number: "<< cuda_devices_number << endl;
    DeviceInfo _deviceInfo;
    bool _isd_evice_compatible = _deviceInfo.isCompatible();
    cout << "CUDA Device(s) Compatible: " << _isd_evice_compatible << endl;
    cout << "Hello, World!" << endl;
    //open 1.jpg and upload it to gpu then download it and show it
    cv::Mat _src_img = cv::imread("E:\\university\\Projects\\cyrus\\data\\1.jpg");
    cv::Mat _dst_img;
    cv::cuda::GpuMat _src_gpu_img;
    _src_gpu_img.upload(_src_img);
    cout << "gpumat image shape : " << _src_gpu_img.channels() << " " << _src_gpu_img.rows << " " << _src_gpu_img.cols << endl;
    int *d_hist;
    int *h_hist = new int[256]{0};
    int _hist_size = 256;
    //allocate memory for histogram on gpu
    cudaError_t err;
    err = cudaMalloc((void **) &d_hist, _hist_size * sizeof(int));
    if (err != cudaSuccess)
    {
        cout << "Error allocating device memory!" << endl;
        return -1;
    }
    err = cudaMemset(d_hist, 0, _hist_size * sizeof(int));
    if (err != cudaSuccess)
    {
        cout << "Error setting device memory!" << endl;
        return -1;
    }
    //allocate memory for histogram on cpu
    int *d_count;
    int *h_count = new int[1]{0};
    //allocate memory for count on gpu
    err = cudaMalloc((void **) &d_count, sizeof(int));
    if (err != cudaSuccess)
    {
        cout << "Error allocating device memory!" << endl;
        return -1;
    }
    err = cudaMemset(d_count, 0, sizeof(int));
    if (err != cudaSuccess)
    {
        cout << "Error setting device memory!" << endl;
        return -1;
    }
    for (int i = 0; i < _hist_size; i++)
    {
        cout << h_hist[i] << " ";
    }
    
    cout << "count is " << h_count[0] << endl;
    //allocate memory for count on cpu
    //calculate histogram on gpu
    test_histogram<<<1,1>>>(_src_gpu_img, d_count, 0, 0, 0, 0, 0, d_hist, 1);
    cudaDeviceSynchronize();
    //copy count from gpu to cpu
    cudaMemcpy(h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    //copy hist from gpu to cpu
    cudaError_t errr =  cudaMemcpy(h_hist, d_hist, _hist_size * sizeof(int) - 1, cudaMemcpyDeviceToHost);
    if (errr != cudaSuccess)
    {
        cout << "Error copying device memory!" << cudaGetErrorString(errr) << __FILE__ <<
__LINE__ << endl;
        return -1;
    }
    //print histogram on cpu
    for (int i = 0; i < _hist_size; i++)
    {
        cout << h_hist[i] << " ";
    }
    
    cout << "count is " << h_count[0] << endl;

    cv::imshow("src_img", _src_img);
    cv::waitKey(0);
    return 0;
}