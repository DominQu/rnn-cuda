#pragma once

#include <vector>
#include "linalg/GPUMatrix.hpp"


class SGD {
  private:
  const float learning_rate;
  public:
  SGD(float learning_rate) : learning_rate(learning_rate) {};
  ~SGD() {};
  std::vector<GPUMatrix> calculateUpdate(std::vector<GPUMatrix> &gradients);
};