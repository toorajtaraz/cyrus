#include <iostream>
using namespace std;

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


#include <opencv2/cudaarithm.hpp>
using namespace cv::cuda;

#include <cuda.h>
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
    cv::cuda::GpuMat _dst_gpu_img;
    _src_gpu_img.upload(_src_img);
    cv::cuda::add(_src_gpu_img, _src_gpu_img, _dst_gpu_img);
    _dst_gpu_img.download(_dst_img);
    cv::imshow("src_img", _src_img);
    cv::imshow("dst_img", _dst_img);
    cv::waitKey(0);
    return 0;
}