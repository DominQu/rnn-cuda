// This class is cuda-accelerated Matrix
// It is needed for all linear algebra operations
#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstddef>
#include <iostream>

#define CUDA_CALL(x)                                                           \
  {                                                                            \
    cudaError_t cuda_error__ = (x);                                            \
    if (cuda_error__)                                                          \
      std::cout << "CUDA error: " #x " returned "                              \
                << cudaGetErrorString(cuda_error__) << std::endl;              \
  }

class MatrixSize {
public:
  MatrixSize() : width(0), height(0), total(0) {}
  MatrixSize(std::size_t height, std::size_t width)
      : height(height), width(width), total(width * height){};

  
  std::size_t height;
  std::size_t width;
  std::size_t total;
};

using MatrixValType = float;

/* GPU */
__global__ void init(MatrixValType *matrix, const MatrixSize size,
                     const MatrixValType val);

/* CPU */
class Matrix {
public:
  /// Initialize matrix of some size
  Matrix(const MatrixSize &size);

  /// Initialize matrix of some size, with values specified in the constructor
  Matrix(const MatrixSize &size, const MatrixValType val);

  ~Matrix();
  
  /// Show matrix to the stdout (requires copying to CPU)
  void show() const;

  Matrix multiply(const Matrix &other);
  Matrix multiply(const MatrixValType other);
  Matrix add(const Matrix &other);
  Matrix add(const MatrixValType other);

  inline MatrixSize getSize() const { return size; }

private:
  MatrixSize size;
  MatrixValType *gpuData; // Pointer to global memory on GPU
};
