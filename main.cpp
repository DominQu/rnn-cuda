#include "linalg/GPUMatrix.hpp"
#include "layers/lstmlayer.hpp"
#include "layers/softmax.hpp"
#include "layers/CCEloss.hpp"
#include "loader/loader.hpp"
#include <iostream>

using Matrix = GPUMatrix;

std::vector<GPUMatrix> testDataset() {
  try {
    DataLoader dl("data/dziady-ascii.txt");
    dl.show(std::cout);
    auto batch = dl.getBatch(1024);
    return batch;
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << std::endl;
  }
}

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


  std::vector<GPUMatrix> batch = testDataset();

  std::cout << "LSTM forward pass:" << std::endl;

  int input_size = 64;
  int state_size = 512;
  int timesteps = 2;

  LstmLayer layer(input_size, state_size, timesteps, 0, 1);
  Softmax softmax(input_size);
  CCEloss cceloss;


  GPUMatrix output = layer.forward(batch);
  GPUMatrix probabilities = softmax.forward(output);
  MatrixValType loss = cceloss.forward(probabilities, batch[1]); //Not an actual label

  std::cout << "Loss: " << loss << std::endl;
  std::cout << "Forward pass finished" << std::endl;

  // Matrix grad = loss.backward(result, label);
  // grad.show(std::cout);
  // std::vector<GPUMatrix> gradients = layer.backward(cost, batch);

  return 0;
}
