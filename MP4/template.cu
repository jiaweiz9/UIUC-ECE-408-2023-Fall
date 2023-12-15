#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define Mask_Width 3
#define Tile_Width 8

//@@ Define constant memory for device kernel here
__constant__ float Mc[Mask_Width][Mask_Width][Mask_Width];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  __shared__ float input_tile[Tile_Width + Mask_Width - 1][Tile_Width + Mask_Width - 1][Tile_Width + Mask_Width - 1];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  int z_out = blockIdx.z * Tile_Width + tz;
  int y_out = blockIdx.y * Tile_Width + ty;
  int x_out = blockIdx.x * Tile_Width + tx;
  int z_in = z_out - Mask_Width / 2;
  int y_in = y_out - Mask_Width / 2;
  int x_in = x_out - Mask_Width / 2;

  // load global memory to shared memory with halo data
  if(z_in >= 0 && z_in < z_size &&
     y_in >= 0 && y_in < y_size &&
     x_in >= 0 && x_in < x_size)
  {
    input_tile[tz][ty][tx] = input[(x_size * y_size) * z_in + x_size * y_in + x_in];
  }
  else
  {
    input_tile[tz][ty][tx] = 0.0f;
  }
  __syncthreads();

  // calculate convolution result
  float Pvalue = 0.0f;
  if(tz < Tile_Width && ty < Tile_Width && tx < Tile_Width)
  {
    for(int i = 0; i < Mask_Width; i++)
    {
      for(int j = 0; j < Mask_Width; j++)
      {
        for(int k = 0; k < Mask_Width; k++)
        {
          Pvalue += input_tile[tz + i][ty + j][tx + k] * Mc[i][j][k];
        }
      }
    }
    if(z_out < z_size && y_out < y_size && x_out < x_size)
      output[(x_size * y_size) * z_out + x_size * y_out + x_out] = Pvalue;
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  int size_input = z_size * y_size * x_size * sizeof(float);
  cudaMalloc((void **)&deviceInput, size_input);
  cudaMalloc((void **)&deviceOutput, size_input);
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput + 3, size_input, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Mc, hostKernel, kernelLength * sizeof(float));
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 dimGrid(ceil(x_size * 1.0 / Tile_Width), ceil(y_size * 1.0 / Tile_Width), ceil(z_size * 1.0 / Tile_Width));
  dim3 dimBlock(Tile_Width + Mask_Width -1, Tile_Width + Mask_Width - 1, Tile_Width + Mask_Width -1);
  
  //@@ Launch the GPU kernel here
  cudaDeviceSynchronize();
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput + 3, deviceOutput, size_input, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbLog(TRACE, "The Output Data is:");
  // for(int i = 0; i < z_size; i++)
  // {
  //   for(int j = 0; j < y_size; j++)
  //   {
  //     for(int k = 0; k < x_size; k++)
  //     {
  //       printf(" %f", hostOutput[x_size * y_size * i + x_size * j + k + 3]);
  //     }
  //     printf("\n");
  //   }
  //   printf("\n\n");
  // }
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
