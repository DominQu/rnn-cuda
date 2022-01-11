#pragma once

#include "layer.hpp"
#include "linalg/matrix.hpp"

class LstmLayer : public Layer{
private:
  int state_dim;
  int timesteps;
  Matrix c; /// Carry matrix
  Matrix h; /// State matrix which is the output of the cell
public:
  LstmLayer(int state_dim, int timesteps);
  ~LstmLayer();

  Matrix& forward(Matrix& input);
  void backward();
};