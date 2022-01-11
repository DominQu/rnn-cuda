#pragma once

#include "linalg/matrix.hpp"

/// Abstract class for neural network layers
class Layer {
public:
  
  virtual ~Layer() = 0; 

  virtual Matrix& forward() = 0;
  virtual void backward() = 0;

};