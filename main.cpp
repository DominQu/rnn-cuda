#include <iostream>
#include "linalg/matrix.hpp"

int main() {
  Matrix a = Matrix::fromCPU({{ 1, 3 }, { 4, 5 }});
  Matrix b = Matrix::fromCPU({{ 1 }, { 1 }});

  std::cout << "A:" << "\n";
  
  a.show();

  std::cout << "\n";

  std::cout << "B:" << "\n";
  
  b.show();

  std::cout << "\n";

  std::cout << "A * B = C:" << "\n";
  
  Matrix c = a.multiply(b);

  c.show();

  std::cout << "\n";

  Matrix::fromCPU({{1, 2}, {3, 4}}).transpose().show();
  
  return 0;
}
