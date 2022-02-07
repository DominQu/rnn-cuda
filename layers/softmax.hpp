#pragma once

#include "linalg/GPUMatrix.hpp"

class Softmax {
  private:
  int input_dim;
  GPUMatrix normalize(const GPUMatrix &input);
  public:
  Softmax(int input_dim);
  ~Softmax() {};
  GPUMatrix forward(const GPUMatrix &input);
  GPUMatrix backward(const GPUMatrix &upstream, const GPUMatrix &forward_result);
};