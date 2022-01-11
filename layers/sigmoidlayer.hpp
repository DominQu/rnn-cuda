#pragma once

#include "layer.hpp"
#include "linalg/matrix.hpp"

class SigmoidLayer : public Layer {
public:
  SigmoidLayer();
  ~SigmoidLayer();

  Matrix& forward(Matrix& input);
  void backward();
};