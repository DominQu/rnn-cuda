#include "cuda.hpp"
#include "lstmlayer.hpp"

LstmLayer::LstmLayer(int state_dim) : state_dim(state_dim) {}

Matrix& LstmLayer::forward(Matrix& input) {
  this->timesteps = input.size.height;
}