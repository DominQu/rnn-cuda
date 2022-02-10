#pragma once

#include "linalg/GPUMatrix.hpp"
#include "softmax.hpp"

class CCEloss {
  public:
  CCEloss() {};
  ~CCEloss() {};
  MatrixValType forward(const GPUMatrix &softmax,const GPUMatrix &label);
  std::vector<MatrixValType> forward(const std::vector<GPUMatrix> &softmax,
                                     const GPUMatrix &label, 
                                     const std::vector<GPUMatrix> &batch);
  GPUMatrix backward(const GPUMatrix &softmax, const GPUMatrix &label);
  std::vector<GPUMatrix> backward(std::vector<GPUMatrix> &softmax,
                                  GPUMatrix &label, 
                                  std::vector<GPUMatrix> &batch);

};