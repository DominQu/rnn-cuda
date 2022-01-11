#pragma once

#include "layer.hpp"
#include "linalg/matrix.hpp"

class TanhLayer : public Layer {
public:
  TanhLayer();
  ~TanhLayer();

  Matrix& forward(Matrix& input);
  void backward();
};