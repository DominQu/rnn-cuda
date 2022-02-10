#pragma once

#include "linalg/GPUMatrix.hpp"
#include <vector>

class RMSprop {
  private:
  float learning_rate;
  float beta;
  std::vector<GPUMatrix> square_gradients; 
  int gradient_num;
  float epsilon;
  public:
  RMSprop(float learning_rate, float beta, int input_size, int state_size , float epsilon);
  ~RMSprop() {};
  std::vector<GPUMatrix> calculateUpdate(std::vector<GPUMatrix> &gradients);

};