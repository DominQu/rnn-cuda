#pragma once

#include "layer.hpp"
#include "linalg/matrix.hpp"

class LinearLayer : public Layer {
  private:
  Matrix weights;

  public:
  LinearLayer(MatrixSize& weights_size)
    : weights(Matrix(MatrixSize(weights_size.height, weights_size.width))) { };
  ~LinearLayer();

  void setRandomWeights(float mean,
                        float standard_deviation,
                        int num_input_units,
                        int num_output_units);
  Matrix& forward(Matrix& input);
  void backward();
};