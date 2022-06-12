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

int main()
{
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
    int *d_hist;
    int *h_hist;
    int _hist_size = 256;
    //allocate memory for histogram on gpu
    cudaMalloc(&d_hist, _hist_size * sizeof(int));
    cudaMemset(d_hist, 0, _hist_size * sizeof(int));
    //allocate memory for histogram on cpu
    h_hist = (int*)malloc(_hist_size * sizeof(int));
    int *d_count;
    int *h_count;
    //allocate memory for count on gpu
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));

    //allocate memory for count on cpu
    h_count = (int*)malloc(sizeof(int));
    //calculate histogram on gpu
    test_histogram<<<1,1>>>(_src_gpu_img, d_count, 0, 2, 0, 2, 0, d_hist, 1);
    //copy count from gpu to cpu
    cudaMemcpy(h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    //copy hist from gpu to cpu
    cudaMemcpy(h_hist, d_hist, _hist_size * sizeof(int), cudaMemcpyDeviceToHost);

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