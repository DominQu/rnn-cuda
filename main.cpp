#include <iostream>
#include "matrix.hpp"

int main() {
  Matrix a(MatrixSize(4, 2), 1);
  Matrix b(MatrixSize(2, 3), 1);

  a.show();

  std::cout << "\n";
  
  b.show();

  std::cout << "\n";
  
  Matrix c = a.multiply(b);

  c.show();
  
  std::cout << "Hopefully runned kernel !" << std::endl;
  return 0;
}
