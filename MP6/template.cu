// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 1024 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


__global__ void add(float *aux_array, float *output, int len) {
  int index_first = 2 * blockDim.x * blockIdx.x + threadIdx.x;
  int index_second = 2 * blockDim.x * blockIdx.x + blockDim.x + threadIdx.x;
  if(blockIdx.x > 0)
  {
    if(index_first < len)
    {
      output[index_first] += aux_array[blockIdx.x - 1];
    }
    if(index_second < len)
    {
      output[index_second] += aux_array[blockIdx.x - 1];
    }
  }
}

__global__ void scan_aux(float *aux, int len){
  __shared__ float T[2 * BLOCK_SIZE];
  if(threadIdx.x < len)
  {
    T[threadIdx.x] = aux[threadIdx.x];
  }
  else
  {
    T[threadIdx.x] = 0;
  }
  if(threadIdx.x + BLOCK_SIZE < len)
  {
    T[threadIdx.x + BLOCK_SIZE] = aux[threadIdx.x + BLOCK_SIZE];
  }
  else
  {
    T[threadIdx.x + BLOCK_SIZE] = 0;
  }
  int stride = 1;
  while(stride < 2 * BLOCK_SIZE)
  {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if(index < 2 * BLOCK_SIZE && (index - stride) >= 0)
    {
      T[index] += T[index - stride];
    }
    stride = stride * 2;
  }

  stride = BLOCK_SIZE / 2;
  while(stride > 0)
  {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if(index + stride < 2 * BLOCK_SIZE)
    {
      T[index + stride] += T[index];
    }
    stride = stride / 2;
  }
  __syncthreads();
  if(threadIdx.x < len)
  {
    aux[threadIdx.x] = T[threadIdx.x];
  }
  if(threadIdx.x + BLOCK_SIZE < len)
  {
    aux[threadIdx.x + BLOCK_SIZE] = T[threadIdx.x + BLOCK_SIZE];
  }
}

__global__ void scan(float *input, float *output, int len, float *aux_array) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T[2 * BLOCK_SIZE];
  int i = 2 * blockDim.x * blockIdx.x + threadIdx.x;
  if(i < len)
  {
    T[threadIdx.x] = input[i];
  }
  else
  {
    T[threadIdx.x] = 0;
  }
  if(i + BLOCK_SIZE < len)
  {
    T[threadIdx.x + BLOCK_SIZE] = input[i + BLOCK_SIZE];
  }
  else
  {
    T[threadIdx.x + BLOCK_SIZE] = 0;
  }

  int stride = 1;
  while(stride < 2 * BLOCK_SIZE)
  {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if(index < 2 * BLOCK_SIZE && (index - stride) >= 0)
    {
      T[index] += T[index - stride];
    }
    stride = stride * 2;
  }

  stride = BLOCK_SIZE / 2;
  while(stride > 0)
  {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if(index + stride < 2 * BLOCK_SIZE)
    {
      T[index + stride] += T[index];
    }
    stride = stride / 2;
  }

  __syncthreads();
  if(i < len)
  {
    output[i] = T[threadIdx.x];
  }
  if(i + BLOCK_SIZE < len)
  {
    output[i + BLOCK_SIZE] = T[threadIdx.x + BLOCK_SIZE];
  }
  if(threadIdx.x == blockDim.x - 1)
  {
    aux_array[blockIdx.x] = T[BLOCK_SIZE * 2 - 1];
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(ceil((float)numElements / (BLOCK_SIZE * 2)), 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce

  float *device_aux_array;
  cudaMalloc((void **)&device_aux_array, BLOCK_SIZE * 2);
  cudaMemset(device_aux_array, 0, BLOCK_SIZE * 2);
  scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numElements, device_aux_array);
  scan_aux<<<1, 1024>>>(device_aux_array, ceil((float)numElements / (BLOCK_SIZE * 2)));
  add<<<dimGrid, dimBlock>>>(device_aux_array, deviceOutput, numElements);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
