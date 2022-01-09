#pragma once

#include "layer.hpp"
#include "matrix.hpp"

class SigmoidLayer : public Layer {
private:
  Matrix input;
  Matrix output;
public:
  SigmoidLayer();
  ~SigmoidLayer();

  Matrix& forward(Matrix& input);
  void backward();
};