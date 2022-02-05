#pragma once

#include <vector>
#include "linalg/matrix.hpp"

class LstmLayer{
private:
  const int input_dim;
  const int state_dim;
  const int timesteps;
  std::vector<GPUMatrix> c; /// Vector of carry matrices
  std::vector<GPUMatrix> tanh_c; /// Vector of tanh applied elementwise on carry matrices
  std::vector<GPUMatrix> h; /// Vector of state matrices which is the output of the cell
  std::vector<GPUMatrix> f; /// Vector of forget gate values
  std::vector<GPUMatrix> g; /// Vector of g gate values
  std::vector<GPUMatrix> i; /// Vecror of input gate values
  std::vector<GPUMatrix> o; /// Vector of output gate values
  GPUMatrix input_weights_f;
  GPUMatrix input_weights_g;
  GPUMatrix input_weights_i;
  GPUMatrix input_weights_o;
  GPUMatrix state_weights_f;
  GPUMatrix state_weights_g;
  GPUMatrix state_weights_i;
  GPUMatrix state_weights_o;
  GPUMatrix output_weights;
  void applySigmoid(GPUMatrix &input, GPUMatrix &result);
  void applyTanh(GPUMatrix &input, GPUMatrix &result);
public:
  /// Constructor with specified sizes of input and state and number of timesteps
  LstmLayer(int input_dim, int state_dim, int timesteps, int weight_min, int weight_max);
  /// Destructor
  ~LstmLayer() {};
  ///Forward pass of lstm layer on given batch
  Matrix forward(std::vector<CPUMatrix> batch);

};