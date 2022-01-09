#pragma once

#include "layer.hpp"
#include "matrix.hpp"

class LstmLayer {
private:
  int state_dim;
  int timesteps;
  Matrix c; /// Carry matrix
  Matrix h; /// State matrix
public:
  LstmLayer(int state_dim);
  ~LstmLayer();

  Matrix& forward(Matrix& input);
  void backward();
};