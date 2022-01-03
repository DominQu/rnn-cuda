#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#define CUDA_CALL(x)							\
  {                                                                            \
    cudaError_t cuda_error__ = (x);                                            \
    if (cuda_error__)                                                          \
      std::cout << "CUDA error: " #x " returned "                              \
                << cudaGetErrorString(cuda_error__) << std::endl;              \
  }


__global__
void testKernel() {
  return;
}


void runKernel() {
  testKernel<<<1, 1>>>();
  CUDA_CALL(cudaDeviceSynchronize());
}
