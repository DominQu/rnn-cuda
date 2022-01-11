#include "cuda.hpp"
#include "lstmlayer.hpp"

LstmLayer::LstmLayer(int state_dim, int timesteps) 
  : state_dim(state_dim), 
    timesteps(timesteps),
    c(Matrix(MatrixSize(timesteps, state_dim))),
    h(Matrix(MatrixSize(timesteps, state_dim))) 
{
  
}

Matrix& LstmLayer::forward(Matrix& input) {
  this->timesteps = input.getSize().height;
}