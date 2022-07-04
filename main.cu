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
main(int argc, char** argv)
{
  // // Load image
  cv::Mat src = cv::imread("E:\\university\\Projects\\cyrus\\data\\he4.jpg");

  long long time_pure;
  long long time_total;

  //create resized versions of src by 0.1, 0.3, 0.5, 0.7
  cv::Mat src_0_1;
  cv::Mat src_0_3;
  cv::Mat src_0_5;
  cv::Mat src_0_7;

  cv::resize(src, src_0_1, cv::Size(), 0.1, 0.1);
  cv::resize(src, src_0_3, cv::Size(), 0.3, 0.3);
  cv::resize(src, src_0_5, cv::Size(), 0.5, 0.5);
  cv::resize(src, src_0_7, cv::Size(), 0.7, 0.7);
  

  //call interpolating_lhe_api on original image and resized ones
  //print results in this format 
  // [pure time] [width] [height]
  cv::Mat dst = interpolating_lhe_api(src, 151, &time_pure, &time_total);
  cout << time_pure << " " << src.cols << " " << src.rows << endl;
  cv::imshow("src", src);
  cv::imshow("dst", dst);
  cv::Mat dst_0_1 = interpolating_lhe_api(src_0_1, 151, &time_pure, &time_total);
  cout << time_pure << " " << src_0_1.cols << " " << src_0_1.rows << endl;
  cv::imshow("src_0_1", src_0_1);
  cv::imshow("dst_0_1", dst_0_1);
  cv::Mat dst_0_3 = interpolating_lhe_api(src_0_3, 151, &time_pure, &time_total);
  cout << time_pure << " " << src_0_3.cols << " " << src_0_3.rows << endl;
  cv::imshow("src_0_3", src_0_3);
  cv::imshow("dst_0_3", dst_0_3);
  cv::Mat dst_0_5 = interpolating_lhe_api(src_0_5, 151, &time_pure, &time_total);
  cout << time_pure << " " << src_0_5.cols << " " << src_0_5.rows << endl;
  cv::imshow("src_0_5", src_0_5);
  cv::imshow("dst_0_5", dst_0_5);
  cv::Mat dst_0_7 = interpolating_lhe_api(src_0_7, 151, &time_pure, &time_total);
  cout << time_pure << " " << src_0_7.cols << " " << src_0_7.rows << endl;
  cv::imshow("src_0_7", src_0_7);
  cv::imshow("dst_0_7", dst_0_7);
  cv::waitKey(0);
  // std::cout << "time pure: " << time_pure << endl;
  // std::cout << "time total: " << time_total << endl;
  // cv::Mat dst_ = interpolating_lhe_api(src, 151, &time_pure, &time_total);
  // std::cout << "time pure: " << time_pure << endl;
  // std::cout << "time total: " << time_total << endl;
  // // clean_up();
  // cv::imshow("src", src);
  // cv::imshow("dst", dst);
  // cv::waitKey(0);

  // Path args as strings
  // handle_video(
  //   std::string(argv[1]), std::string(argv[2]), 201, &time_pure, &time_total);

  // print taken times in seconds
  // std::cout << "time pure: " << time_pure / 1000 << " s" << endl;
  // std::cout << "time total: " << time_total / 1000 << " s" << endl;

  return 0;
}