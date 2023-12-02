//==============shared-memory.cu============
#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 8

#define M_max 16
#define C_max 6
#define K_max 5
#define S_max 4
#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      std::cout<<"CUDA error: "<<cudaGetErrorString(err)<<std::endl;   \
      exit(-1);                                                          \
    }                                                                     \
  } while (0)

// __constant__ float Mc[M_max * C_max * K_max * K_max];

__global__ void conv_tiled_shared_memory(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    int shared_width = (TILE_WIDTH - 1) * S + K;
    __shared__ float shared_tile[C_max][45][45];
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
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    // #define mask_4d(i3, i2, i1, i0) Mc[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    
    // Insert your GPU convolution kernel code here
    int num_of_block_w = 0;
    if(W_out % TILE_WIDTH == 0)
        num_of_block_w = W_out / TILE_WIDTH;
    else
        num_of_block_w = W_out / TILE_WIDTH + 1;

    int b = blockIdx.x;
    int m = blockIdx.y;
    int th = threadIdx.y;
    int tw = threadIdx.x;
    int block_idx_w = blockIdx.z % num_of_block_w;
    int block_idx_h = blockIdx.z / num_of_block_w;
    int h_o = (block_idx_h) * TILE_WIDTH + threadIdx.y; // h_o is the target output row
    int w_o = (block_idx_w) * TILE_WIDTH + threadIdx.x; // w_o is the target output column
    // int h_o_top = (block_idx_h) * TILE_WIDTH;
    // int w_o_left = (block_idx_w) * TILE_WIDTH;
    
    // int h_i = block_idx_h * shared_width + th * S;
    // int w_i = block_idx_w * shared_width + tw * S;
    
    // for(int b = 0; b < B; b++)
    // {
    for(int c = 0; c < C; c++)
    {
        for(int p = 0; p < S; p++)
        {
            for(int q = 0; q < S; q++)
            {
                for(int i = 0; (th * S + p) + TILE_WIDTH * S * i < shared_width; i++)
                {
                    for(int j = 0; (tw * S + q) + TILE_WIDTH * S * j < shared_width; j++)
                    {
                        if(h_o * S + p + TILE_WIDTH * S * i >= 0 && h_o * S + p + TILE_WIDTH * S * i < H 
                            && w_o * S + q + TILE_WIDTH * S * j >= 0 && w_o * S + q + TILE_WIDTH * S * j < W)
                        // if(h_o >=0 && h_o < H_out && w_o >=0 && w_o < W_out)
                        {
                            shared_tile[c][(th * S + p) + TILE_WIDTH * S * i][(tw * S + q) + TILE_WIDTH * S * j] = 
                                    in_4d(b, c, h_o * S + p + TILE_WIDTH * S * i, w_o * S + q + TILE_WIDTH * S * j);
                        }
                        else
                        {
                            shared_tile[c][(th * S + p) + TILE_WIDTH * S * i][(tw * S + q) + TILE_WIDTH * S * j] = 0.0f;
                        }
                    }
                }
            }
        }
    }
    __syncthreads();

    float acc = 0.0f;
    for (int c = 0; c < C; c++) { // sum over all input channels
        for (int p = 0; p < K; p++) { // loop over KxK filter
            for (int q = 0; q < K; q++) {
                acc += shared_tile[c][th * S + p][tw * S + q] * mask_4d(m, c, p, q);
            }
        }
    }
    if(h_o < H_out && w_o < W_out) {
        out_4d(b, m, h_o, w_o) = acc;
    }
    // }
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
    // GPUInterface::get_device_properties();
    int num_of_input = B * C * H * W;
    int num_of_output = B * M * ((H - K) / S + 1) * ((W - K) / S + 1);
    std::cout<<"num of input "<<num_of_input<<std::endl;
    std::cout<<"num of output "<<num_of_output<<std::endl;
    cudaMalloc((void **)device_input_ptr, num_of_input * sizeof(float));
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<" (error in prolog input alloc)"<<std::endl;
        exit(-1);
    }

    cudaMalloc((void **)device_output_ptr, num_of_output * sizeof(float));
    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<" (error in prolog output alloc)"<<std::endl;
        exit(-1);
    }

    cudaMalloc((void **)device_mask_ptr, M * C * K * K * sizeof(float));
    // cudaDeviceSynchronize();
    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<" (error in prolog mask alloc)"<<std::endl;
        exit(-1);
    }

    cudaMemcpy(*device_input_ptr, host_input, num_of_input * sizeof(float), cudaMemcpyHostToDevice);
    //wbCheck(cudaMemcpy(*device_mask_ptr, host_mask, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice));
    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<" (error in prolog input cpy)"<<std::endl;
        exit(-1);
    }
    
    cudaMemset(*device_output_ptr, 0, num_of_output * sizeof(float));
    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<" (error in prolog output cpy)"<<std::endl;
        exit(-1);
    }
    
    // cudaMemcpyToSymbol(Mc, host_mask, M * C * K * K * sizeof(float));
    wbCheck(cudaMemcpy(*device_mask_ptr, host_mask, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice));
    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<" (error in prolog mask cpy)"<<std::endl;
        exit(-1);
    }

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

    // int shared_width = (TILE_WIDTH - 1) * S + K;
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(B, M, num_of_block);
    // conv_with_constant_mask<<<dimGrid, dimBlock>>>(device_output, device_input, B, M, C, H, W, K, S);
    conv_tiled_shared_memory<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);
    std::cout<<"tiled shared memory"<<std::endl;
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
  