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
  GPUMatrix input_bias_f;
  GPUMatrix input_bias_g;
  GPUMatrix input_bias_i;
  GPUMatrix input_bias_o;
  GPUMatrix output_bias;
  void applySigmoid(GPUMatrix &input, GPUMatrix &result);
  void applyTanh(GPUMatrix &input, GPUMatrix &result);
public:
  /// Constructor with specified sizes of input and state and number of timesteps
  LstmLayer(int input_dim, int state_dim, int timesteps, 
            MatrixValType weight_min, MatrixValType weight_max, 
            MatrixValType bias_init=0);
  /// Constructor for loading a model from file
  LstmLayer(int input_dim, int state_dim, int timesteps, std::string filepath);
  /// Destructor
  ~LstmLayer() {};
  ///Forward pass of lstm layer on given batch, return output of the last hidden cell.
  Matrix forward(std::vector<GPUMatrix> batch);
  ///Forward pass of lstm layer on given batch, returns outputs of all hidden cells.
  std::vector<GPUMatrix> forward(std::vector<GPUMatrix> batch, bool return_sequence);
  ///Backward pass of lstm layer, returns vector of gradients
  std::vector<GPUMatrix> backward(GPUMatrix upstream, std::vector<GPUMatrix> batch);
  /// Backward pass of lstm layer, takes as input upstream gradients for every timestep, returns vector of gradients
  std::vector<GPUMatrix> backward(std::vector<GPUMatrix> upstream, std::vector<GPUMatrix> batch);

  /** Update weights of the network, this function doesn't implement an optimizer.
  * scaled_gradients should contain gradients in this order:
  * f gate input weights gradient
  * g gate input weights gradient
  * i gate input weights gradient
  * o gate input weights gradient
  * f gate state weights gradient
  * g gate state weights gradient
  * i gate state weights gradient
  * o gate state weights gradient
  * output linear layer weights gradient
  */
  void updateWeights(std::vector<GPUMatrix> scaled_gradients);

  void saveWeights(std::fstream &output);
  void showWeights();

};