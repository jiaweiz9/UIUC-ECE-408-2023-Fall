// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 1024
//@@ insert code here
__global__ void cast(float* floatImage, unsigned char* ucharImage, int width, int height, int channels) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int i = index; i < width * height * channels; i += stride) {
    ucharImage[i] = (unsigned char)(255 * floatImage[i]);
    //i = i + stride;
  }
}

__global__ void rgb2gray(unsigned char* ucharImage, unsigned char* grayImage, int width, int height, int channels) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int i = index; i < width * height; i += stride) {
    unsigned char r = ucharImage[3 * i];
    unsigned char g = ucharImage[3 * i + 1];
    unsigned char b = ucharImage[3 * i + 2];
    grayImage[i] = (unsigned char)(0.21 * r + 0.71 * g + 0.07 * b);
  }
}

__global__ void histogram(unsigned char* grayImage, int width, int height, int* histo) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x; // stride = total # of threads
  // All threads in the grid collectively handle 
  // blockDim.x * gridDim.x consecutive elements
  while (i < width * height) {
    atomicAdd(&(histo[grayImage[i]]), 1);
    i += stride;
  }
}

__global__ void scan_cdf(int *input, float *cdf, int len, int width, int height) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ int T[2 * BLOCK_SIZE];
  int i = 2 * blockDim.x * blockIdx.x + threadIdx.x;

  //=============================init process===============================
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
  // ==========================calculate step1========================
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
  // ==========================calculate step2============================
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
  //=======================copy back to output and aux====================
  __syncthreads();
  if(i < len)
  {
    cdf[i] = (float)T[threadIdx.x] / (width * height);
  }
  if(i + BLOCK_SIZE < len)
  {
    cdf[i + BLOCK_SIZE] = (float)T[threadIdx.x + BLOCK_SIZE] / (width * height);
  }
  // if(threadIdx.x == blockDim.x - 1)
  // {
  //   aux_array[blockIdx.x] = T[BLOCK_SIZE * 2 - 1];
  // }
}

__global__ void cdf(int* histo, float* cdf, int width, int height) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int i = index; i < 256; i += stride) {
    cdf[i] = (float)histo[i] / (width * height);
  }
}

__global__ void correct_color_output(unsigned char* ucharImage, float* outputImage, float* cdf, int cdfmin, int width, int height, int channel) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  
  for(int i = index; i < width * height * channel; i += stride) {
    int value = ucharImage[i];
    int corrected_value = 255 * (cdf[value] - cdfmin) / (1.0 - cdfmin);
    outputImage[i] = (float) (corrected_value / 255.0);
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float* deviceInputFloatImage;
  float* deviceOutputFloatImage;
  unsigned char* ucharImage;
  unsigned char* grayImage;
  //int *aux_array;
  int* histo;
  float* device_cdf;
  float* host_cdf;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  cudaMalloc((void **)&ucharImage, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
  cudaMemset(ucharImage, 0, imageWidth * imageHeight * imageChannels);
  
  cudaMalloc((void **)&grayImage, imageWidth * imageHeight * sizeof(unsigned char));
  cudaMemset(grayImage, 0, imageWidth * imageHeight);
  
  cudaMalloc((void **)&histo, HISTOGRAM_LENGTH * sizeof(int));
  cudaMemset(histo, 0, HISTOGRAM_LENGTH);
  //cudaMalloc((void **)&aux_array, BLOCK_SIZE * sizeof(int));
  cudaMalloc((void **)&device_cdf, HISTOGRAM_LENGTH * sizeof(float));
  cudaMemset(device_cdf, 0, HISTOGRAM_LENGTH);
  host_cdf = (float *)malloc(HISTOGRAM_LENGTH * sizeof(float));

  cudaMalloc((void **)&deviceInputFloatImage, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMemcpy(deviceInputFloatImage, hostInputImageData, 
                  imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&deviceOutputFloatImage, imageWidth * imageHeight * imageChannels * sizeof(float));
  
  dim3 dimGrid(ceil((float)imageWidth * imageHeight / BLOCK_SIZE), 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  cast<<<dimGrid, dimBlock>>>(deviceInputFloatImage, ucharImage, imageWidth, imageHeight, imageChannels);
  rgb2gray<<<dimGrid, dimBlock>>>(ucharImage, grayImage, imageWidth, imageHeight, imageChannels);
  histogram<<<dimGrid, dimBlock>>>(grayImage, imageWidth, imageHeight, histo);
  scan_cdf<<<dimGrid, dimBlock>>>(histo, device_cdf, HISTOGRAM_LENGTH, imageWidth, imageHeight);

  cudaMemcpy(host_cdf, device_cdf, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyDeviceToHost);
  
  int cdfmin = 0;
  for(int i = 0; i < HISTOGRAM_LENGTH; i++) {
    if(host_cdf[i] > 0) {
      cdfmin = host_cdf[i];
      break;
    }
  }

  correct_color_output<<<dimGrid, dimBlock>>>(ucharImage, deviceOutputFloatImage, device_cdf, cdfmin, imageWidth, imageHeight, imageChannels);
  cudaMemcpy(hostOutputImageData, deviceOutputFloatImage, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceInputFloatImage);
  cudaFree(deviceOutputFloatImage);
  cudaFree(ucharImage);
  cudaFree(grayImage);
  cudaFree(histo);
  cudaFree(device_cdf);
  free(hostInputImageData);
  free(hostOutputImageData);
  free(host_cdf);
  return 0;
}
