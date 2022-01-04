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

/* CPU */
class Matrix {
public:
  /// Copy constructor
  Matrix(const Matrix& copied);
  /// Move constructor
  Matrix(Matrix&& moved);
  
  /// Initialize matrix of some size
  Matrix(const MatrixSize &size);

  /// Initialize matrix of some size, with values specified in the constructor
  Matrix(const MatrixSize &size, const MatrixValType val);

  /// Create matrix of size like provided one
  static Matrix like(const Matrix& other) {
    return Matrix(other.getSize());
  }
  
  ~Matrix();
  
  /// Show matrix to the stdout (requires copying to CPU)
  void show() const;

  // These functions do not modify the object
  Matrix multiply(const Matrix &other) const;
  Matrix multiply(const Matrix &other, Matrix& result) const;
  
  // These functions modify the object
  void multiply(const MatrixValType scalar);
  void add(const Matrix &other);
  void add(const MatrixValType other);

  inline const MatrixSize getSize() const { return size; }
private:
  MatrixSize size;
  MatrixValType *gpuData; // Pointer to global memory on GPU
};
