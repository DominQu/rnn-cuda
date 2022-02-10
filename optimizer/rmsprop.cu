#include "rmsprop.hpp"

RMSprop::RMSprop(float learning_rate, float beta, int input_dim, int state_dim) 
  : learning_rate(learning_rate), beta(beta), gradient_num(14) {
  square_gradients.emplace_back(MatrixSize(state_dim, input_dim), 0);
  square_gradients.emplace_back(MatrixSize(state_dim, input_dim), 0);
  square_gradients.emplace_back(MatrixSize(state_dim, input_dim), 0);
  square_gradients.emplace_back(MatrixSize(state_dim, input_dim), 0);
  
  square_gradients.emplace_back(MatrixSize(state_dim, state_dim), 0);
  square_gradients.emplace_back(MatrixSize(state_dim, state_dim), 0);
  square_gradients.emplace_back(MatrixSize(state_dim, state_dim), 0);
  square_gradients.emplace_back(MatrixSize(state_dim, state_dim), 0);
  
  square_gradients.emplace_back(MatrixSize(input_dim, state_dim), 0);
  
  square_gradients.emplace_back(MatrixSize(state_dim, 1), 0);
  square_gradients.emplace_back(MatrixSize(state_dim, 1), 0);
  square_gradients.emplace_back(MatrixSize(state_dim, 1), 0);
  square_gradients.emplace_back(MatrixSize(state_dim, 1), 0);

  square_gradients.emplace_back(MatrixSize(input_dim, 1), 0);
}

std::vector<GPUMatrix> RMSprop::calculateUpdate(std::vector<GPUMatrix> &gradients) {
  std::vector<GPUMatrix> result;
  for(int i = 0 ; i < this->gradient_num; i++) {
    square_gradients[i].multiply(this->beta, square_gradients[i]);
    square_gradients[i].add(gradients[i].multiplyelementwise(gradients[i]).multiply(1-this->beta), square_gradients[i]);
    result.push_back(gradients[i].divideelementwise(square_gradients[i].sqrt()).multiply(this->learning_rate));
  }  
}
