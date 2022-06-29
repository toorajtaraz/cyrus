#include <helpers.h>
__global__ void __extract_histogram(const uchar *img, int *count, int x_start, int x_end, int y_start, int y_end, int width, int height, int steps, int *histt, int sw)
{
    extern __shared__ int sh_histt[];

    int index = threadIdx.y * blockDim.y + threadIdx.x;
    // devid 256 by total number of threads in 2d block
    int s = 256 / (blockDim.x * blockDim.y);

    if (index == 0)
    {
        sh_histt[256] = 0;
    }

    if (s < 1)
        s = 1;

    for (int i = index * s; i < (index * s + 1); i++)
        if (i < 256)
        {
            sh_histt[i] = 0;
        }

    __syncthreads();

    if (x_start < height && y_start < width && x_start >= 0 && y_start >= 0)
    {

        atomicAdd(&sh_histt[256], sw);
        atomicAdd(&sh_histt[img[y_start * steps + x_start]], sw);
    }

    __syncthreads();
    if (index == 0)
    {
        atomicAdd(&count[0], sh_histt[256]);
    }

    if (index < 256)
    {
        atomicAdd(&histt[index], sh_histt[index]);
    }
}
__global__ void __extract_histogram_rgb(const uchar *img, int *count, int x_start, int x_end, int y_start, int y_end, int width, int height, int steps, short channel, short channels_c, int *histt, int sw)
{
    extern __shared__ int sh_histt[];

    // calculate global id
    int index = threadIdx.y * blockDim.y + threadIdx.x;
    int s = 256 / (blockDim.x * blockDim.y);

    if (index == 0)
    {
        sh_histt[256] = 0;
    }

    if (s < 1)
        s = 1;

    for (int i = index * s; i < (index * (s + 1)); i++)
        if (i < 256)
        {
            sh_histt[i] = 0;
        }

    __syncthreads();
    // do if boundary is valid
    for (auto i = x_start; i < x_end && i < height; i++)
        for (auto j = y_start; j < y_end && j < width; j++)
        {
            atomicAdd(&sh_histt[256], sw);
            atomicAdd(&sh_histt[img[j * steps + i * channels_c + channel]], sw);
        }

    __syncthreads();
    if (index == 0)
    {
        atomicAdd(&count[0], sh_histt[256]);
    }

    if (index < 256)
    {
        atomicAdd(&histt[index], sh_histt[index]);
    }
}
__device__ void helper_extract_histogram(const uchar *img, int *count, int x_start, int x_end, int y_start, int y_end, int width, int height, int steps, int *histt, int sw)
{
    extern __shared__ int sh_histt[];

    int index = threadIdx.y * blockDim.y + threadIdx.x;
    // devid 256 by total number of threads in 2d block
    int s = 256 / (blockDim.x * blockDim.y);

    if (index == 0)
    {
        sh_histt[256] = 0;
    }

    if (s < 1)
        s = 1;

    for (int i = index * s; i < ((index + 1) * s); i++)
        if (i < 256)
        {
            sh_histt[i] = 0;
        }

    __syncthreads();

    if (x_start < height && y_start < width && x_start >= 0 && y_start >= 0)
    {

        atomicAdd(&sh_histt[256], sw);
        atomicAdd(&sh_histt[img[y_start * steps + x_start]], sw);
    }

    __syncthreads();
    if (index == 0)
    {
        atomicAdd(&count[0], sh_histt[256]);
    }

    for (int i = index * s; i < ((index + 1) * s); i++)
        if (index < 256)
        {
            atomicAdd(&histt[i], sh_histt[i]);
        }
}
__device__ void helper_extract_histogram_rgb(const uchar *img, int *count, int x_start, int x_end, int y_start, int y_end, int width, int height, int steps, short channel, short channels_c, int *histt, int sw)
{
    extern __shared__ int sh_histt[];

    // calculate global id
    int index = threadIdx.y * blockDim.y + threadIdx.x;
    int s = 256 / (blockDim.x * blockDim.y);

    if (index == 0)
    {
        sh_histt[256] = 0;
    }

    if (s < 1)
        s = 1;

    for (int i = index * s; i < ((index + 1) * s); i++)
        if (i < 256)
        {
            sh_histt[i] = 0;
        }

    __syncthreads();
    // do if boundary is valid
    for (auto i = x_start; i < x_end && i < height; i++)
        for (auto j = y_start; j < y_end && j < width; j++)
        {
            atomicAdd(&sh_histt[256], sw);
            atomicAdd(&sh_histt[img[j * steps + i * channels_c + channel]], sw);
        }

    __syncthreads();
    if (index == 0)
    {
        atomicAdd(&count[0], sh_histt[256]);
    }

    for (int i = index * s; i < ((index + 1) * s); i++)
        if (i < 256)
        {
            atomicAdd(&histt[i], sh_histt[i]);
        }
}
__device__ void helper_calculate_probability(int *hist, int total_pixels, double *prob)
{
    // in order for this function to perform correctly, total number of threads must equal PIXEL_RANGE
    //  calculate thread id in 1d kernel
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < 256)
    {
        prob[index] = (double)hist[index] / total_pixels;
    }
}
__device__ void helper_buildLook_up_table(double *prob, double *lut)
{
    // shared lut
    __shared__ double sh_lut[256];

    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < 256)
    {
        sh_lut[index] = 0.0;
    }

    __syncthreads();
    if (index < 256)
        for (auto j = 0; j <= index; j++)
        {
            sh_lut[index] += prob[j] * MAX_PIXEL_VAL;
        }
    __syncthreads();

    if (index < 256)
    {
        lut[index] = sh_lut[index];
    }
}
__global__ void buildLook_up_table_rgb(int *hist_blue, int *hist_green, int *hist_red, int count, bool free_sw, double *lut_final, double *lut_blue, double *lut_green, double *lut_red)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index == 0)
    {
        double *prob_blue = new double[256];
        double *prob_red = new double[256];
        double *prob_green = new double[256];

        calculate_probability<<<1, 256>>>(hist_blue, count, prob_blue);
        calculate_probability<<<1, 256>>>(hist_green, count, prob_green);
        calculate_probability<<<1, 256>>>(hist_red, count, prob_red);
        cudaDeviceSynchronize();

        buildLook_up_table<<<1, 256>>>(prob_blue, lut_blue);
        buildLook_up_table<<<1, 256>>>(prob_green, lut_green);
        buildLook_up_table<<<1, 256>>>(prob_red, lut_red);

        delete[] prob_red;
        delete[] prob_blue;
        delete[] prob_green;
        cudaDeviceSynchronize();
    }

    __syncthreads();
    if (index < 256)
        lut_final[index] = (lut_blue[index] + lut_green[index] + lut_red[index]) / 3.0;
}

// __global__ void apply_LHE_helper(uchar *base, const uchar *img, int window, int i_start, int i_end, int width, int height, int steps, int channels_c)
// {
//     int offset = (int)floor(window / 2.0);
//     int count = 0;
//     int sw = 0;
//     int **hists;
//     int *hist;
//     int temp;
//     if (channels_c > 1)
//     {
//         hists = new int *[channels_c];
//         for (auto i = 0; i < channels_c; i++)
//         {
//             hists[i] = new int[PIXEL_RANGE]();
//         }
//     }
//     else
//     {
//         hist = new int[PIXEL_RANGE]();
//     }
//     for (int i = i_start; i < i_end; i++)
//     {
//         sw = i % 2 == (i_start % 2) ? 0 : 1;
//         if (sw == 1)
//         {
//             for (int j = width - 1; j >= 0; j--)
//             {
//                 if (j == (width - 1))
//                 {
//                     for (int n = 0; n < window; n++)
//                     {
//                         if (channels_c > 1)
//                         {
//                             for (auto k = 0; k < channels_c; k++)
//                             {
//                                 temp = count;
//                                 extract_histogram_rgb(img, &temp, i - 1 - offset, i - 1 - offset + 1, j + n - offset, j + n - offset + 1, width, height, steps, k, 3, hists[k], -1);
//                                 extract_histogram_rgb(img, &temp, i + window - 1 - offset, i + window - 1 - offset + 1, j + n - offset, j + n - offset + 1, width, height, steps, k, 3, hists[k], 1);
//                             }
//                             count = temp;
//                         }
//                         else
//                         {
//                             extract_histogram(img, &count, i - 1 - offset, i - 1 - offset + 1, j + n - offset, j + n - offset + 1, width, height, steps, hist, -1);
//                             extract_histogram(img, &count, i + window - 1 - offset, i + window - 1 - offset + 1, j + n - offset, j + n - offset + 1, width, height, steps, hist, 1);
//                         }
//                     }
//                 }
//                 else if (j < (width - 1))
//                 {
//                     for (int n = 0; n < window; n++)
//                     {
//                         if (channels_c > 1)
//                         {
//                             for (auto k = 0; k < channels_c; k++)
//                             {
//                                 temp = count;
//                                 extract_histogram_rgb(img, &temp, i + n - offset, i + n - offset + 1, j - offset, j - offset + 1, width, height, steps, k, 3, hists[k], 1);
//                                 extract_histogram_rgb(img, &temp, i + n - offset, i + n - offset + 1, j + window - offset, j + window - offset + 1, width, height, steps, k, 3, hists[k], -1);
//                             }
//                             count = temp;
//                         }
//                         else
//                         {
//                             extract_histogram(img, &count, i + n - offset, i + n - offset + 1, j - offset, j - offset + 1, width, height, steps, hist, 1);
//                             extract_histogram(img, &count, i + n - offset, i + n - offset + 1, j + window - offset, j + window - offset + 1, width, height, steps, hist, -1);
//                         }
//                     }
//                 }
//                 count = count > 0 ? count : 1;
//                 if (channels_c > 1)
//                 {
//                     double *lut;
//                     buildLook_up_table_rgb(hists[0], hists[1], hists[2], count, true, lut);
//                     for (auto k = 0; k < channels_c; k++)
//                     {
//                         base[j * steps + i * k] = img[j * steps + i * k];
//                         // base.at<cv::Vec3b>(i, j)[k] = (uchar)lut[img.at<cv::Vec3b>(i, j)[k]];
//                     }
//                     delete[] lut;
//                 }
//                 else
//                 {
//                     double *prob;
//                     calculate_probability(hist, count, prob);
//                     double *lut;
//                     buildLook_up_table(prob, lut);
//                     base[j * steps + i] = img[j * steps + i];
//                     // base.at<uchar>(i, j) = (int)floor(lut[img.at<uchar>(i, j)]);
//                     // Clean memory
//                     delete[] prob;
//                     delete[] lut;
//                 }
//             }
//         }
//         else
//         {
//             for (int j = 0; j < width; j++)
//             {
//                 if (j == 0 && i > i_start)
//                 {
//                     for (int n = 0; n < window; n++)
//                     {
//                         if (channels_c > 1)
//                         {
//                             for (auto k = 0; k < channels_c; k++)
//                             {
//                                 temp = count;
//                                 extract_histogram_rgb(img, &temp, i - 1 - offset, i - 1 - offset + 1, j + n - offset, j + n - offset + 1, width, height, steps, k, 3, hists[k], -1);
//                                 extract_histogram_rgb(img, &temp, i + window - 1 - offset, i + window - 1 - offset + 1, j + n - offset, j + n - offset + 1, width, height, steps, k, 3, hists[k], 1);
//                             }
//                             count = temp;
//                         }
//                         else
//                         {
//                             extract_histogram(img, &count, i - 1 - offset, i - 1 - offset + 1, j + n - offset, j + n - offset + 1, width, height, steps, hist, -1);
//                             extract_histogram(img, &count, i + window - 1 - offset, i + window - 1 - offset + 1, j + n - offset, j + n - offset + 1, width, height, steps, hist, 1);
//                         }
//                     }
//                 }
//                 else if (j == 0 && i == i_start)
//                 {
//                     for (int n = 0; n < window; n++)
//                     {
//                         for (int m = 0; m < window; m++)
//                         {
//                             if (channels_c > 1)
//                             {
//                                 for (auto k = 0; k < channels_c; k++)
//                                 {
//                                     temp = count;
//                                     extract_histogram_rgb(img, &temp, i + n - offset, i + n - offset + 1, j + m - offset, j + m - offset + 1, width, height, steps, k, 3, hists[k], 1);
//                                 }
//                                 count = temp;
//                             }
//                             else
//                             {
//                                 extract_histogram(img, &count, i + n - offset, i + n - offset + 1, j + m - offset, j + m - offset + 1, width, height, steps, hist, 1);
//                             }
//                         }
//                     }
//                 }
//                 else if (j > 0)
//                 {
//                     for (int n = 0; n < window; n++)
//                     {
//                         if (channels_c > 1)
//                         {
//                             for (auto k = 0; k < channels_c; k++)
//                             {
//                                 temp = count;
//                                 extract_histogram_rgb(img, &temp, i + n - offset, i + n - offset + 1, j - 1 - offset, j - 1 - offset + 1, width, height, steps, k, 3, hists[k], -1);
//                                 extract_histogram_rgb(img, &temp, i + n - offset, i + n - offset + 1, j + window - 1 - offset, j + window - 1 - offset + 1, width, height, steps, k, 3, hists[k], 1);
//                             }
//                             count = temp;
//                         }
//                         else
//                         {
//                             extract_histogram(img, &count, i + n - offset, i + n - offset + 1, j - 1 - offset, j - 1 - offset + 1, width, height, steps, hist, -1);
//                             extract_histogram(img, &count, i + n - offset, i + n - offset + 1, j + window - 1 - offset, j + window - 1 - offset + 1, width, height, steps, hist, 1);
//                         }
//                     }
//                 }
//                 count = count > 0 ? count : 1;
//                 if (channels_c > 1)
//                 {
//                     double *lut;
//                     buildLook_up_table_rgb(hists[0], hists[1], hists[2], count, true, lut);
//                     for (auto k = 0; k < channels_c; k++)
//                     {
//                         base[j * steps + i * k] = img[j * steps + i * k];
//                         // base.at<cv::Vec3b>(i, j)[k] = (uchar)lut[img.at<cv::Vec3b>(i, j)[k]];
//                     }
//                     delete[] lut;
//                 }
//                 else
//                 {
//                     double *prob;
//                     calculate_probability(hist, count, prob);
//                     double *lut;
//                     buildLook_up_table(prob, lut);
//                     base[j * steps + i] = img[j * steps + i];
//                     // base.at<uchar>(i, j) = (int)floor(lut[img.at<uchar>(i, j)]);
//                     // Clean memory
//                     delete[] prob;
//                     delete[] lut;
//                 }
//             }
//         }
//     }
//     if (channels_c > 1)
//     {
//         // delete channels
//         for (auto k = 0; k < channels_c; k++)
//         {
//             delete[] hists[k];
//         }
//     }
//     else
//     {
//         delete[] hist;
//     }
// }

void apply_LHE(cv::Mat &base, cv::Mat img, int window)
{
    // #pragma omp parallel
    //     {
    //         int n_threads = omp_get_num_threads();
    //         int thread_id = omp_get_thread_num();
    //         int i_start = thread_id * (base.rows / n_threads);
    //         int i_end = (thread_id + 1) * (base.rows / n_threads);

    //         if (thread_id == n_threads - 1)
    //         {
    //             i_end = base.rows;
    //         }
    //         ApplyLHEHelper(base, img, window, i_start, i_end);
    //     }
}

__global__ void extract_histogram_rgb(const uchar *img, int *count, int x_start, int x_end, int y_start, int y_end, int width, int height, int steps, short channel, short channels_c, int *histt, int sw)
{
    int x_range = x_end - x_start;
    int y_range = y_end - y_start;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int x_coverage = gridDim.x * blockDim.x;
    int y_coverage = gridDim.y * blockDim.y;

    int step_x = x_range / x_coverage;
    int step_y = y_range / y_coverage;

    if (step_x < 1)
    {
        step_x = 1;
    }
    if (step_y < 1)
    {
        step_y = 1;
    }

    helper_extract_histogram_rgb(img, count, x * step_x, (x + 1) * step_x, y * step_y, (y + 1) * step_y, width, height, steps, channel, channels_c, histt, sw);
}

__global__ void extract_histogram(const uchar *img, int *count, int x_start, int x_end, int y_start, int y_end, int width, int height, int steps, int *histt, int sww)
{
    int x_range = x_end - x_start;
    int y_range = y_end - y_start;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int x_coverage = gridDim.x * blockDim.x;
    int y_coverage = gridDim.y * blockDim.y;

    int step_x = x_range / x_coverage;
    int step_y = y_range / y_coverage;

    if (step_x < 1)
    {
        step_x = 1;
    }

    if (step_y < 1)
    {
        step_y = 1;
    }

    helper_extract_histogram(img, count, x * step_x, (x + 1) * step_x, y * step_y, (y + 1) * step_y, width, height, steps, histt, sww);
}

__global__ void calculate_probability(int *hist, int total_pixels, double *prob)
{
    helper_calculate_probability(hist, total_pixels, prob);
}

__global__ void buildLook_up_table(double *prob, double *lut)
{
    helper_buildLook_up_table(prob, lut);
}