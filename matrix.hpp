// This class is cuda-accelerated Matrix
// It is needed for all linear algebra operations
#pragma once

#include <cstddef>
#include <iostream>
#include <vector>
#include <exception>
#include <string>

class InvalidMatrixSize : std::exception {
private:
  std::string message;
public:
  InvalidMatrixSize(std::string message) : message(message) {}
  virtual char const* what() const throw() {
    return message.c_str();
  }
};

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
using CPUMatrix = std::vector<std::vector<MatrixValType>>;

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

  /// Create matrix of size like provided one (and fill it with some value)
  static Matrix like(const Matrix& other, const MatrixValType val) {
    return Matrix(other.getSize(), val);
  }

  /// Create matrix from cpu based data (std vectors)
  static Matrix fromCPU(const CPUMatrix&);

  /// Destructor
  ~Matrix();

  /// Convert GPU based data to CPU based
  CPUMatrix toCPU() const;
  
  /// Show matrix to the stdout (requires copying to CPU)
  void show() const;

  // These functions do not modify the object
  Matrix multiply(const Matrix &other) const;
  void multiply(const Matrix &other, Matrix& result) const;

  Matrix transpose() const;
  
  // These functions modify the object
  void multiply(const MatrixValType scalar);
  void add(const Matrix &other);
  void add(const MatrixValType other);

  inline const MatrixSize getSize() const { return size; }
  MatrixValType* gpu_handle() { return gpuData; }
private:
  MatrixSize size;
  MatrixValType *gpuData; // Pointer to global memory on GPU
};
