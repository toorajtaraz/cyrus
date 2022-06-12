#include <helpers.h>

#include <thrust/device_vector.h>
__device__ int *extract_histogram(cv::cuda::GpuMat &img, int *count, int x_start, int x_end, int y_start, int y_end, int *histt, int sw) {
    if (histt == NULL)
    {
        histt = new int[PIXEL_RANGE]();
    }
    else
    {
        if (x_start < 0 || x_end > img.rows || y_start < 0 || y_end > img.cols)
        {
            return NULL;
        }
    }
    //get image height and width from device
    int height = img.rows;
    int width = img.cols;

    if (x_start < 0)
        x_start = 0;
    if (x_end > height)
        x_end = height;
    if (y_start < 0)
        y_start = 0;
    if (y_end > width)
        y_end = width;

    for (auto i = x_start; i < x_end; i++)
    {
        for (auto j = y_start; j < y_end; j++)
        {
            *count += sw;
            histt[img.data[i * width + j]] += sw;
        }
    }
    return histt;
}
__device__ void extract_histogram_rgb(cv::cuda::GpuMat &img, int *count, int x_start, int x_end, int y_start, int y_end, short channel, int *histt, int sw) {
    if (x_start < 0 || x_end > img.rows || y_start < 0 || y_end > img.cols)
        return;

    int height = img.rows;
    int width = img.cols;
    if (x_start < 0)
        x_start = 0;
    if (x_end > height)
        x_end = height;
    if (y_start < 0)
        y_start = 0;
    if (y_end > width)
        y_end = width;
    

    for (auto i = x_start; i < x_end; i++)
        for (auto j = y_start; j < y_end; j++)
        {
            // int temp = *count;
            *count = 1;
            // *count += 1;
            // histt[img.data[i * width + j * RGB_CHANNELS + channel]] += sw;
            //             histt[img.data[i * width + j * RGB_CHANNELS + channel]] += sw;
        }
    return;    
}
__device__ double *calculate_probability(int *hist, int total_pixels) {
    double *prob = new double[PIXEL_RANGE]();
    for (auto i = 0; i < PIXEL_RANGE; i++)
    {
        prob[i] = (double)hist[i] / total_pixels;
    }
    return prob;
}
__device__ double *buildLook_up_table(double *prob) {
    double *lut = new double[PIXEL_RANGE]();

    for (auto i = 0; i < PIXEL_RANGE; i++)
    {
        for (auto j = 0; j <= i; j++)
        {
            lut[i] += prob[j] * MAX_PIXEL_VAL;
        }
    }
    return lut;
}
__device__ double *buildLook_up_table_rgb(int *hist_blue, int *hist_green, int *hist_red, int count, bool free_sw) {
    double *prob_blue = calculate_probability(hist_blue, count);
    double *lut_blue = buildLook_up_table(prob_blue);
    delete[] prob_blue;

    double *prob_green = calculate_probability(hist_green, count);
    double *lut_green = buildLook_up_table(prob_green);
    delete[] prob_green;

    double *prob_red = calculate_probability(hist_red, count);
    double *lut_red = buildLook_up_table(prob_red);
    delete[] prob_red;

    double *lut_final = new double[PIXEL_RANGE]();

    for (auto i = 0; i < PIXEL_RANGE; i++)
    {
        lut_final[i] = (lut_blue[i] + lut_green[i] + lut_red[i]) / 3.0;
    }
    if (free_sw)
    {
        delete[] lut_blue;
        delete[] lut_green;
        delete[] lut_red;
    }
    return lut_final;    
}
__global__ void test_histogram(cv::cuda::GpuMat &img, int *count, int x_start, int x_end, int y_start, int y_end, short channel, int *histt, int sw) {
    //call extract histogram rgb
    if (x_start < 0 || x_end > img.rows || y_start < 0 || y_end > img.cols)
        return;

    int height = img.rows;
    int width = img.cols;
    if (x_start < 0)
        x_start = 0;
    if (x_end > height)
        x_end = height;
    if (y_start < 0)
        y_start = 0;
    if (y_end > width)
        y_end = width;
    

    for (auto i = x_start; i < x_end; i++)
        for (auto j = y_start; j < y_end; j++)
        {
            count[0] += 1;
            // histt[img.data[i * width + j * RGB_CHANNELS + channel]] += sw;
            //             histt[img.data[i * width + j * RGB_CHANNELS + channel]] += sw;
        }
}