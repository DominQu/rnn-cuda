#include "matrix.hpp"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

bool matEq(const CPUMatrix& m1, const CPUMatrix& m2) {
  if (m1.size() != m2.size()) return false;
  if (m1[0].size() != m2[0].size()) return false;

  for (int y = 0; y < m1.size(); y++) {
    for (int x = 0; x < m1[0].size(); x++) {
      if (m1[y][x] != m2[y][x]) return false;
    }
  }

  return true;
}

TEST_CASE("CPU <-> GPU") {
  CPUMatrix _a = {{3, 6}, {1, 2}};
  Matrix a = Matrix::fromCPU(_a);
  CPUMatrix b = a.toCPU();

  CHECK(matEq(b, _a));
}

TEST_CASE("Simple Multiplication") {
  Matrix a = Matrix::fromCPU({{ 1, 3 }, { 4, 5 }});
  Matrix b = Matrix::fromCPU({{ 1 }, { 1 }});

  Matrix c = a.multiply(b);

  CHECK_EQ(c.toCPU().size(), 2);
  CHECK_EQ(c.toCPU()[0].size(), 1);
  CHECK(matEq(c.toCPU(), {{4}, {9}}));
}

TEST_CASE("More multiplication") {
  Matrix a = Matrix::fromCPU({{6, 6, 6}, {2, 2, 2}});
  Matrix b = Matrix::fromCPU({{1, 1, 1}, {1, 1, 1}, {1, 1, 1}});

  Matrix c = a.multiply(b);

  CHECK(matEq(c.toCPU(), {{18, 18, 18}, {6, 6, 6}}));
}

