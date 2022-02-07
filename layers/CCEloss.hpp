#pragma once

#include "linalg/GPUMatrix.hpp"
#include "softmax.hpp"

class CCEloss {
  public:
  CCEloss() {};
  ~CCEloss() {};
  MatrixValType forward(const GPUMatrix &input,const GPUMatrix &label);
  GPUMatrix backward(const GPUMatrix &softmax, const GPUMatrix &label);
};