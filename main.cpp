#include <iostream>
#include "linalg/GPUMatrix.hpp"

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

  Matrix::from({{1, 2}, {3, 4}}).transpose().show(std::cout);

  Matrix e = Matrix::from({{ 2, 2 }, { 2, 2 }});

  Matrix d = a.multiplyelementwise(e);
  
  d.show(std::cout);

  std::cout << "\n";
  
  return 0;
}
