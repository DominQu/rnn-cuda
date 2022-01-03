#include <iostream>
#include "matrix.hpp"

int main() {
  Matrix m(MatrixSize(8, 32), 5);
  m.show();
  std::cout << "Hopefully runned kernel !" << std::endl;
  return 0;
}
