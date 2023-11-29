//===================matrix-unroll.cu===========
#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define TILE_WIDTH_2 16
#define M_max 16
#define C_max 6
#define K_max 5
#define S_max 2

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      std::cout<<"CUDA error: "<<cudaGetErrorString(err)<<std::endl;   \
      exit(-1);                                                          \
    }                                                                     \
  } while (0)

__constant__ float Mc[M_max * C_max * K_max * K_max];

__global__ void unroll(const int B, const int C, const int H, const int W, const int K, const int S, const float* X, float* X_unroll) 
{
    int H_out = (H - K) / S + 1; // calculate H_out, W_out
    int W_out = (W - K) / S + 1;

    int num_of_block_w = 0;
    if(W_out % TILE_WIDTH == 0)
        num_of_block_w = W_out / TILE_WIDTH;
    else
        num_of_block_w = W_out / TILE_WIDTH + 1;
    // int num_of_block_h = ceil(H_out / TILE_WIDTH);
    int b = blockIdx.x;
    // int m = blockIdx.y;
    int th = (blockIdx.y / num_of_block_w) * TILE_WIDTH + threadIdx.y;
    int tw = (blockIdx.y % num_of_block_w) * TILE_WIDTH + threadIdx.x;

    int col = H_out * W_out;
    // #define X_in(i3, i2, i1, i0) X[(i3) * (X * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    // #define X_out(i2, i1, i0) X_unroll[(i2) * (K * K * C * ((H - K) / S + 1) * ((H - K) / S + 1)) + (i1) * (K * K * C) + i0]

    // for (int b = 0; b < B; ++b) // for each image
    if(tw < W_out && th < H_out)
    {
        for (int c = 0; c < C; c++) { // for each input channel
            int w_base = c * (K * K); // per-channel offset for smallest X_unroll index
            for (int p = 0; p < K; p++) // for each element of KxK filter (two loops)
                for (int q = 0; q < K; q++) { 
                    int h_unroll = w_base + p * K + q; // data needed by one thread
                    int w_unroll = th * W_out + tw; // smallest index--across threads (output values)
                    X_unroll[b * K * K * C * col + h_unroll * col + w_unroll] = X[b * C * H * W + c * (H * W) + (th * S + p) * (W) + tw * S + q]; // copy input pixels
                }
                    // }
        }
    }
}

__global__ void matrixMultiplyShared(const float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    //@@ You have to use shared memory for this MP
    __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];
    int batch = blockIdx.z;
    int block_x = blockIdx.x; 
    int block_y = blockIdx.y;
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;
    int row = block_y * TILE_WIDTH + thread_y;
    int col = block_x * TILE_WIDTH + thread_x;
    float pValue = 0;
    for(int q = 0; q < ceil((float)numAColumns / TILE_WIDTH); q++) {
        if(q * TILE_WIDTH + thread_x < numAColumns && row < numARows) {
            subTileA[thread_y][thread_x] = A[row * numAColumns + q * TILE_WIDTH + thread_x];
        }
        else {
            subTileA[thread_y][thread_x] = 0;
        }
        if(q * TILE_WIDTH + thread_y < numBRows && col < numBColumns) {
            subTileB[thread_y][thread_x] = B[(numBRows * numBColumns) * batch + (q * TILE_WIDTH + thread_y) * numBColumns + col];
        }
        else {
            subTileB[thread_y][thread_x] = 0;
        }
        __syncthreads();

        for(int k = 0; k < TILE_WIDTH; k++) {
            // pValue += Mc[row * numAColumns + q * TILE_WIDTH + k] * subTileB[k][thread_x];
            pValue += subTileA[thread_y][k] * subTileB[k][thread_x];
        }
        __syncthreads();
    }
    if(row < numCRows && col < numCColumns) {
        C[(numCRows * numCColumns) * batch + row * numCColumns + col] = pValue;
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    int num_of_input = B * C * H * W;
    int num_of_output = B * M * ((H - K) / S + 1) * ((W - K) / S + 1);
    std::cout<<"num of input "<<num_of_input<<std::endl;
    std::cout<<"num of output "<<num_of_output<<std::endl;
    wbCheck(cudaMalloc((void **)device_input_ptr, num_of_input * sizeof(float)));
    wbCheck(cudaMalloc((void **)device_output_ptr, num_of_output * sizeof(float)));
    wbCheck(cudaMalloc((void **)device_mask_ptr, M * C * K * K * sizeof(float)));

    cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<" (error in prolog memalloc)"<<std::endl;
    //     exit(-1);
    // }

    wbCheck(cudaMemcpy(*device_input_ptr, host_input, num_of_input * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(*device_mask_ptr, host_mask, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemset(*device_output_ptr, 0, num_of_output * sizeof(float)));
    // wbCheck(cudaMemcpyToSymbol(Mc, host_mask, M * C * K * K * sizeof(float)));

    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<" (error in prolog memcpy)"<<std::endl;
        exit(-1);
    }
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel
    int output_w = (W - K) / S + 1;
    int output_h = (H - K) / S + 1;
    int num_of_block_w = ceil((float)output_w / TILE_WIDTH);
    int num_of_block_h = ceil((float)output_h / TILE_WIDTH);
    int num_of_block = num_of_block_w * num_of_block_h;
    int unrolled_col = output_h * output_w;
    
    float* unrolled_input;
    cudaMalloc((void **)&unrolled_input, B * unrolled_col * C * K * K * sizeof(float));
    
    // wbLog(TRACE, "The number of blocks in width is ", num_of_block_w);
    // wbLog(TRACE, "The number of blocks in height is ", num_of_block_h);
    std::cout<<"num of block w "<<num_of_block_w<<std::endl;
    std::cout<<"num of block h "<<num_of_block_h<<std::endl;

    // int shared_width = (TILE_WIDTH - 1) * S + K;
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(B, num_of_block, 1);
    // conv_with_constant_mask<<<dimGrid, dimBlock>>>(device_output, device_input, B, M, C, H, W, K, S);
    unroll<<<dimGrid, dimBlock>>>(B, C, H, W, K, S, device_input, unrolled_input);

    int mat_block_w = ceil((float)unrolled_col / TILE_WIDTH_2);
    int mat_block_h = ceil((float)M / TILE_WIDTH_2);
    dim3 dimGrid2(mat_block_w, mat_block_h, B);
    dim3 dimBlock2(TILE_WIDTH_2, TILE_WIDTH_2, 1);
    matrixMultiplyShared<<<dimGrid2, dimBlock2>>>(device_mask, unrolled_input, device_output, M, C * K * K, C * K * K, unrolled_col, M, unrolled_col);
    // std::cout<<"tiled shared memory"<<std::endl;
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<" (error in kernel call)"<<std::endl;
        exit(-1);
    }
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Copy the output back to host
    int num_of_output = B * M * ((H - K) / S + 1) * ((W - K) / S + 1);
    cudaMemcpy(host_output, device_output, num_of_output * sizeof(float), cudaMemcpyDeviceToHost);
   
    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}