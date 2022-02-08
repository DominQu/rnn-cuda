#include "linalg/GPUMatrix.hpp"
#include "layers/lstmlayer.hpp"
#include "layers/softmax.hpp"
#include "layers/CCEloss.hpp"
#include "loader/loader.hpp"
#include "optimizer/sgd.hpp"
#include <iostream>

using Matrix = GPUMatrix;

int main() {
  Matrix a = Matrix::from({{ 1, 3 }, { 4, 5 }});
  Matrix b = Matrix::from({{ 1 }, { 1 }});

  std::cout << "A:" << "\n";

  a.show(std::cout);

  std::cout << "\n";

  std::cout << "B:" << "\n";

  b.show(std::cout);

  std::cout << "\n";

  std::cout << "A * B = C:" << "\n";

  Matrix c = a.multiply(b);

  c.show(std::cout);

  std::cout << "\n";

  std::cout << "E" << std::endl;

  Matrix e = Matrix::from({{ 2, 2 }, { 2, 2 }});
  
  e.show(std::cout);

  std::cout << "\n";

  std::cout << "A * E (elementwise) = D:" << std::endl;

  Matrix d = a.multiplyelementwise(e);
  
  d.show(std::cout);

  std::cout << "\n";

  DataLoader dl("data/dziady-ascii.txt");
  dl.show(std::cout);
  std::vector<GPUMatrix> batch = dl.getTrainBatch(3);
  GPUMatrix label = batch.back();
  batch.pop_back();

  // std::cout << "label size " << label.getSize().height << std::endl;

  std::cout << "LSTM forward pass:" << std::endl;

  int input_size = 64;
  int state_size = 512;
  int timesteps = 2;

  LstmLayer layer(input_size, state_size, timesteps, 0, 1);
  Softmax softmax(input_size);
  CCEloss cceloss;
  SGD sgd(0.01);


  GPUMatrix output = layer.forward(batch);
  GPUMatrix probabilities = softmax.forward(output);
  MatrixValType loss = cceloss.forward(probabilities, label);

  std::cout << "Loss: " << loss << std::endl;
  std::cout << "Forward pass finished" << std::endl;

  std::cout << "Backward pass: " << std::endl;

  Matrix lossgrad = cceloss.backward(probabilities, label);
  // probabilities.show(std::cout);
  std::cout << "--------" << std::endl;
  // lossgrad.show(std::cout);

  std::vector<GPUMatrix> gradients = layer.backward(lossgrad, batch);

  std::cout << "Backward pass finished" << std::endl;
  
  std::cout << "Optimizing using stochastic gradient descent" << std::endl;

  std::vector<GPUMatrix> scaled_gradients = sgd.calculateUpdate(gradients);
  
  std::cout << "Updating weights" << std::endl;

  layer.updateWeights(scaled_gradients);

  return 0;
}
