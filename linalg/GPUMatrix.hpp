// This class is cuda-accelerated Matrix
// It is needed for all linear algebra operations
#pragma once

#include <cstddef>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CPUMatrix.hpp"

class GPUMatrix {
public:
  /// Copy constructor
  GPUMatrix(const GPUMatrix &copied);
  /// Move constructor
  GPUMatrix(GPUMatrix &&moved);

  /// Initialize matrix of some size
  GPUMatrix(const MatrixSize &size);

  /// Initialize matrix of some size, with values specified in the constructor
  GPUMatrix(const MatrixSize &size, const MatrixValType val);

  /// Create matrix of size like provided one
  static GPUMatrix like(const GPUMatrix &other) {
    return GPUMatrix(other.getSize());
  }

  /// Create matrix of size like provided one (and fill it with some value)
  static GPUMatrix like(const GPUMatrix &other, const MatrixValType val) {
    return GPUMatrix(other.getSize(), val);
  }

  /// Create matrix from cpu based data (std vectors)
  static GPUMatrix from(const std::initializer_list<std::initializer_list<MatrixValType>> &);
  static GPUMatrix from(const CPUMatrix &);

  static GPUMatrix random(const MatrixSize &size);

  /// Destructor
  ~GPUMatrix();

  /// Convert GPU based data to CPU based
  CPUMatrix toCPU() const;

  void show(std::ostream &outStream) { this->toCPU().show(outStream); }

  // These functions do not modify the object
  GPUMatrix multiply(const MatrixValType scalar) const;
  void multiply(const MatrixValType scalar, GPUMatrix &result) const;
  GPUMatrix multiply(const GPUMatrix &other) const;
  void multiply(const GPUMatrix &other, GPUMatrix &result) const;

  GPUMatrix multiplyelementwise(const GPUMatrix &other) const;
  void multiplyelementwise(const GPUMatrix &other, GPUMatrix &result) const;

  GPUMatrix add(const GPUMatrix &other) const;
  void add(const GPUMatrix &other, GPUMatrix &result) const;
  GPUMatrix add(const MatrixValType scalar) const;
  void add(const MatrixValType scalar, GPUMatrix &result) const;

  /// Return transposed matrix
  GPUMatrix transpose() const;
  /// Transpose matrix into passed result GPUMatrix
  void transpose(GPUMatrix &result) const;

  // TODO: Use function below instead of accesing data directly
  inline const MatrixSize getSize() const { return size; }

  inline MatrixValType *gpuHandle() { return gpuData; }
  inline const MatrixValType *gpuHandle() const { return gpuData; }

  /// Synchronizes GPU, should be called before every operation on the array
  inline void syncGPU() const;
private:
  MatrixSize size;
  MatrixValType *gpuData; // Pointer to global memory on GPU
};
