#include "linalg/GPUMatrix.hpp"
#include "layers/lstmlayer.hpp"
#include "loader/loader.hpp"
#include <iostream>

using Matrix = GPUMatrix;

void testDataset() {
  try {
    DataLoader dl("data/dziady-ascii.txt");
    dl.show(std::cout);
    auto batch = dl.getBatch(32);
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

  std::cout << "LSTM forward pass:" << std::endl;

  LstmLayer layer(64, 512, 2, 0, 1);
  std::vector<CPUMatrix> batch;
  batch.emplace_back(MatrixSize(64,1), 1);
  batch.emplace_back(MatrixSize(64,1), 1);

  GPUMatrix output = layer.forward(batch);
  GPUMatrix cost(output.getSize(), 1);
  cost.add(output.multiply(-1), cost);
  std::vector<GPUMatrix> gradients = layer.backward(cost, batch);


  return 0;
}
