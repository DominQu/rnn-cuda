#pragma once

#include "linalg/GPUMatrix.hpp"
#include "softmax.hpp"

class CCEloss {
  public:
  CCEloss() {};
  ~CCEloss() {};
  MatrixValType forward(const GPUMatrix &softmax,const GPUMatrix &label);
  GPUMatrix backward(const GPUMatrix &softmax, const GPUMatrix &label);
};