#include <helpers.h>

__device__ void extract_histogram(const uchar *img, int width, int height, int *count, int x_start, int x_end, int y_start, int y_end, int steps, int *histt, int sw)
{
    extern __shared__ int sh_histt[];

    int index = threadIdx.y * blockDim.y + threadIdx.x;

    if (index == 0)
    {
        sh_histt[256] = 0;
    }

    if (index < 256)
    {
        sh_histt[index] = 0;
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
__device__ void extract_histogram_rgb(const uchar *img, int width, int height, int *count, int x_start, int x_end, int y_start, int y_end, int steps, short channel, short channels_c, int *histt, int sw)
{
    extern __shared__ int sh_histt[];

    // calculate global id
    int index = threadIdx.y * blockDim.y + threadIdx.x;

    if (index == 0)
    {
        sh_histt[256] = 0;
    }

    if (index < 256)
    {
        sh_histt[index] = 0;
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
__device__ double *calculate_probability(int *hist, int total_pixels)
{
    double *prob = new double[PIXEL_RANGE]{0.0};

    // // calculate global id
    // int index = threadIdx.y * blockDim.y + threadIdx.x;

    // if (index < 256)
    // {
    //     prob[i] = (double)hist[i] / total_pixels;
    // }

    // for (auto i = 0; i < PIXEL_RANGE; i++)
    // {
    //     prob[i] = (double)hist[i] / total_pixels;
    // }
    return prob;
}
__device__ double *buildLook_up_table(double *prob)
{
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
__device__ double *buildLook_up_table_rgb(int *hist_blue, int *hist_green, int *hist_red, int count, bool free_sw)
{
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
__global__ void test_histogram(const uchar *img, int width, int height, int *count, int x_start, int x_end, int y_start, int y_end, int steps, int block_size, int block_threads, short channel, short channels_c, int *histt, int sw)
{
    // calculate x and y
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    extract_histogram_rgb(img, width, height, count, x, x + 1, y, y + 1, steps, channel, channels_c, histt, sw);
}