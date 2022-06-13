#include <helpers.h>

#include <thrust/device_vector.h>
__device__ void extract_histogram(const uchar *img, int width, int height, int *count, int x_start, int x_end, int y_start, int y_end, int steps, int *histt, int sw)
{
    extern __shared__ int sh_histt[];

    if (threadIdx.x == 0)
    {
        sh_histt[256] = 0;
    }

    if (threadIdx.x < 256) {
        sh_histt[threadIdx.x] = 0;
    }

    __syncthreads();

    if (x_start < height && y_start < width && x_start >= 0 && y_start >= 0)
    {

        atomicAdd(&sh_histt[256], sw);
        atomicAdd(&sh_histt[img[y_start * steps + x_start]], sw);
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        count[0] = sh_histt[256];
    }

    if (threadIdx.x < 256)
    {
        atomicAdd(&histt[threadIdx.x], sh_histt[threadIdx.x]);
    }
}
__device__ void extract_histogram_rgb(const uchar *img, int width, int height, int *count, int x_start, int x_end, int y_start, int y_end, int steps, short channel, short channels_c, int *histt, int sw)
{
    extern __shared__ int sh_histt[];

    //calculate global id 
    int index = threadIdx.y * blockDim.y + threadIdx.x;

    if (index == 0)
    {
        sh_histt[256] = 0;
    }

    if (index < 256) {
        sh_histt[index] = 0;
    }

    __syncthreads();
    //do if boundary is valid
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
    double *prob = new double[PIXEL_RANGE]();
    for (auto i = 0; i < PIXEL_RANGE; i++)
    {
        prob[i] = (double)hist[i] / total_pixels;
    }
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
    // //calculate range for each block
    // int x_range = (x_end - x_start + 1) / block_size;
    // int y_range = (y_end - y_start + 1) / block_size;

    // //calculate range for each thread
    // int x_thread_range = (x_range + 1) / block_threads;
    // int y_thread_range = (y_range + 1) / block_threads;

    // //get block id
    // int block_id_x = blockIdx.x * blockDim.x;
    // int block_id_y = blockIdx.y * blockDim.y;

    // //get thread id
    // int thread_id = threadIdx.x;

    // int this_thread_x_start =  x_start + block_id_x * x_range + thread_id * x_thread_range;
    // int this_thread_y_start = y_start + block_id_y * y_range + thread_id * y_thread_range;

    // int this_thread_x_end = this_thread_x_start + x_thread_range;
    // int this_thread_y_end = this_thread_y_start + y_thread_range;

    // //print ids
    // printf("block_id_X: %d, block_id_Y: %d, thread_id: %d\n", block_id_x, block_id_y, thread_id);
    // printf("x_start : %d, x_end : %d, y_start : %d, y_end : %d\n", this_thread_x_start, this_thread_x_end, this_thread_y_start, this_thread_y_end);
    // extract_histogram_rgb(img, width, height, count, this_thread_x_start, this_thread_x_end, this_thread_y_start, this_thread_y_end, steps, channel, channels_c, histt, sw);
}