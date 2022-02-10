#pragma once

#include "linalg/GPUMatrix.hpp"

class Softmax {
  private:
  int input_dim;
  GPUMatrix normalize(const GPUMatrix &input);
  public:
  Softmax(int input_dim);
  ~Softmax() {};
  /// Calculate probabilities of single input 
  GPUMatrix forward(const GPUMatrix &input);
  /// Calculate probabilities of every Matrix from a sequence 
  std::vector<GPUMatrix> forward(const std::vector<GPUMatrix> &input, bool return_sequence);

  GPUMatrix backward(const GPUMatrix &upstream, const GPUMatrix &forward_result);
};