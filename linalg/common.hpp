// This class is cuda-accelerated Matrix
// It is needed for all linear algebra operations
#pragma once

#include <cstddef>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

class InvalidMatrixSize : std::exception {
private:
  std::string message;

public:
  InvalidMatrixSize(std::string message) : message(message) {}
  virtual char const *what() const throw() { return message.c_str(); }
};

class MatrixSize {
public:
  MatrixSize() : width(0), height(0), total(0) {}
  MatrixSize(std::size_t height, std::size_t width)
      : height(height), width(width), total(width * height){};

  bool operator==(const MatrixSize& other) const {
    return this->height == other.height && this->width == other.width;
  }
  
  bool operator!=(const MatrixSize& other) const {
    return !(*this == other);
  }
  
  std::size_t height;
  std::size_t width;
  std::size_t total;
};

using MatrixValType = float;
