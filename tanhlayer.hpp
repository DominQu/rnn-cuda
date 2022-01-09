#pragma once

#include "layer.hpp"
#include "matrix.hpp"

class TanhLayer : public Layer {
public:
  TanhLayer();
  ~TanhLayer();

  Matrix& forward(Matrix& input);
  void backward();
};