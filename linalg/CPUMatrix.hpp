// This class is CPU Matrix
// It is needed for all linear algebra operations
#pragma once

#include <cstddef>
#include <exception>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include "common.hpp"

class CPUMatrix {
public:
  /// Copy constructor
  CPUMatrix(const CPUMatrix &copied);
  /// Move constructor
  CPUMatrix(CPUMatrix &&moved);

  /// Initialize matrix of some size
  CPUMatrix(const MatrixSize &size);

  /// Initialize matrix of some size, with values specified in the constructor
  CPUMatrix(const MatrixSize &size, const MatrixValType val);

  /// Destructor
  ~CPUMatrix();

  /// Create matrix of size like provided one
  static CPUMatrix like(const CPUMatrix &other) {
    return CPUMatrix(other.getSize());
  }

  /// Create matrix of size like provided one (and fill it with some value)
  static CPUMatrix like(const CPUMatrix &other, const MatrixValType val) {
    return CPUMatrix(other.getSize(), val);
  }

  /// Create matrix from cpu based data (std vectors)
  static CPUMatrix from(const std::vector<std::vector<MatrixValType>> &);

  static CPUMatrix random(const MatrixSize size, const MatrixValType min=0, const MatrixValType max=1);
  
  /// Accessor
  MatrixValType& at(const std::size_t y, const std::size_t x);
  const MatrixValType at(const std::size_t y, const std::size_t x) const;

  /// Comparing matricies
  bool operator==(const CPUMatrix& other) const;
  bool operator!=(const CPUMatrix& other) const {
    return !(*this == other);
  };

  /// Show matrix to the stdout (requires copying to CPU)
  void show(std::ostream&) const;

  // Serialization and deserialization
  /// Serialize object into file
  void serialize(std::ostream&) const;
  /// Deserialize object from file (must have same Matrix size)
  void deSerialize(std::istream&);

  // These functions do not modify the object
  CPUMatrix multiply(const MatrixValType scalar) const;
  void multiply(const MatrixValType scalar, CPUMatrix &result) const;
  CPUMatrix multiply(const CPUMatrix &other) const;
  void multiply(const CPUMatrix &other, CPUMatrix &result) const;

  CPUMatrix add(const CPUMatrix &other) const;
  void add(const CPUMatrix &other, CPUMatrix &result) const;
  CPUMatrix add(const MatrixValType scalar) const;
  void add(const MatrixValType scalar, CPUMatrix &result) const;

  int argmax() const;

  /// Return transposed matrix
  CPUMatrix transpose() const;
  /// Transpose matrix into passed result CPUMatrix
  void transpose(CPUMatrix &result) const;

  inline const MatrixSize getSize() const { return size; }

  inline MatrixValType *cpuHandle() { return cpuData; }
  inline const MatrixValType *cpuHandle() const { return cpuData; }
private:
  MatrixSize size;
  MatrixValType *cpuData; // Pointer to global memory on GPU

  inline std::size_t coordsFrom(const std::size_t y, const std::size_t x) const {
    return y * this->getSize().width + x;
  }
};
