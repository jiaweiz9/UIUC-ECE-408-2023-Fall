//==========constant-memery.cu============
#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16

#define M_max 16
#define C_max 8
#define K_max 8
#define S_max 5
#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      std::cout<<"CUDA error: "<<cudaGetErrorString(err)<<std::endl;   \
      exit(-1);                                                          \
    }                                                                     \
  } while (0)

__constant__ float Mc[M_max * C_max * K_max * K_max];

__global__ void conv_with_constant_mask(float *output, const float *input, const int B, const int M, const int C, const int H, const int W, const int K, const int S) 
{
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) Mc[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int num_of_block_w = 0;
    if(W_out % TILE_WIDTH == 0)
        num_of_block_w = W_out / TILE_WIDTH;
    else
        num_of_block_w = W_out / TILE_WIDTH + 1;
    // int num_of_block_h = ceil(H_out / TILE_WIDTH);
    int b = blockIdx.x;
    int m = blockIdx.y;
    // int c = threadIdx.z;
    int h = (blockIdx.z / num_of_block_w) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.z % num_of_block_w) * TILE_WIDTH + threadIdx.x;
    
    if(h < H_out && w < W_out) {
        // for(int b = 0; b < B; b++) {
            float acc = 0.0f;
            for (int c = 0; c < C; c++) { // sum over all input channels
                for (int p = 0; p < K; p++) // loop over KxK filter
                    for (int q = 0; q < K; q++)
                            acc += in_4d(b, c, h * S + p, w * S + q) * mask_4d(m, c, p, q);
            }
            out_4d(b, m, h, w) = acc;
        // }
    }
    #undef out_4d
    #undef in_4d
    #undef mask_4d
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
    // wbCheck(cudaMemcpy(*device_mask_ptr, host_mask, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemset(*device_output_ptr, 0, num_of_output * sizeof(float)));
    wbCheck(cudaMemcpyToSymbol(Mc, host_mask, M * C * K * K * sizeof(float)));

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

    
    // wbLog(TRACE, "The number of blocks in width is ", num_of_block_w);
    // wbLog(TRACE, "The number of blocks in height is ", num_of_block_h);
    std::cout<<"num of block w "<<num_of_block_w<<std::endl;
    std::cout<<"num of block h "<<num_of_block_h<<std::endl;
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(B, M, num_of_block);
    // conv_with_constant_mask<<<dimGrid, dimBlock>>>(device_output, device_input, B, M, C, H, W, K, S);
    conv_with_constant_mask<<<dimGrid, dimBlock>>>(device_output, device_input, B, M, C, H, W, K, S);
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
