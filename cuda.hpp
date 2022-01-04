// This file is default header for all cuda files

#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CUDA_CALL(x)							       \
  {                                                                            \
    cudaError_t cuda_error__ = (x);                                            \
    if (cuda_error__)                                                          \
      std::cout << "CUDA error: " #x " returned "                              \
                << cudaGetErrorString(cuda_error__) << std::endl;              \
  }
