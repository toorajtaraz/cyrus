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

#include <api.h>
#include <helpers.h>
#include <interpolating_lhe.h>
#include <lhe.h>

int
main()
{
  // Load image
  cv::Mat src = cv::imread("E:\\university\\Projects\\cyrus\\data\\1.jpg");

  long long time_pure;
  long long time_total;

  cv::Mat dst = interpolating_lhe_api(src, 151, &time_pure, &time_total);

  std::cout << "time pure: " << time_pure << endl;
  std::cout << "time total: " << time_total << endl;
  cv::Mat dst_ = interpolating_lhe_api(src, 151, &time_pure, &time_total);
  std::cout << "time pure: " << time_pure << endl;
  std::cout << "time total: " << time_total << endl;
  clean_up();
  cv::imshow("src", src);
  cv::imshow("dst", dst);
  cv::waitKey(0);
  return 0;
}