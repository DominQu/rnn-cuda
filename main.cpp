#include <iostream>
#include "linalg/GPUMatrix.hpp"
#include "layers/lstmlayer.hpp"

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

  std::cout << "LSTM forward pass:" << std::endl;

  LstmLayer layer(64, 512, 2, 0, 1);
  std::vector<CPUMatrix> batch;
  batch.emplace_back(MatrixSize(64,1), 1);
  batch.emplace_back(MatrixSize(64,1), 1);

  GPUMatrix output = layer.forward(batch);
  // output.show(std::cout);
  std::cout << output.getSize().height << std::endl;

  return 0;
}
