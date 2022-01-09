#include "linalg/matrix.hpp"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

bool matEq(const CPUMatrix &m1, const CPUMatrix &m2) {
  if (m1.size() != m2.size())
    return false;
  if (m1[0].size() != m2[0].size())
    return false;

  for (int y = 0; y < m1.size(); y++) {
    for (int x = 0; x < m1[0].size(); x++) {
      if (m1[y][x] != m2[y][x])
        return false;
    }
  }

  return true;
}

bool matEq(const Matrix &m1, const CPUMatrix &m2) {
  return matEq(m1.toCPU(), m2);
}

bool matEq(const CPUMatrix &m1, const Matrix &m2) {
  return matEq(m1, m2.toCPU());
}

bool matEq(const Matrix &m1, const Matrix &m2) {
  return matEq(m1.toCPU(), m2.toCPU());
}

TEST_CASE("CPU <-> GPU") {
  CPUMatrix a = {{3, 6}, {1, 2}};
  Matrix b = Matrix::fromCPU(a);

  CHECK(matEq(a, b));
}

TEST_CASE("Adding scalar to the Matrix") {
  const auto case1 = Matrix(MatrixSize(8, 8), 1).add(5);

  CHECK(matEq(case1, Matrix(MatrixSize(8, 8), 6)));
}

TEST_CASE("Adding two Matrices") {
  const auto case1 =
      Matrix::fromCPU({{1, 2}, {3, 4}}).add(Matrix::fromCPU({{3, 2}, {1, 0}}));
  CHECK(matEq(case1, {{4, 4}, {4, 4}}));

  CHECK_THROWS(Matrix::fromCPU({{1, 2}}).add(Matrix::fromCPU({{1}, {2}})));

  const auto case2 =
      Matrix(MatrixSize(2, 2), 1).add(Matrix::fromCPU({{1, 8}, {8, 1}}));

  CHECK(matEq(case2, {{2, 9}, {9, 2}}));
}

TEST_CASE("Simple Multiplication") {
  Matrix a = Matrix::fromCPU({{1, 3}, {4, 5}});
  Matrix b = Matrix::fromCPU({{1}, {1}});

  Matrix c = a.multiply(b);

  CHECK(matEq(c, {{4}, {9}}));

  Matrix d = c.multiply(10);

  CHECK(matEq(d, {{40}, {90}}));
}

TEST_CASE("More multiplication") {
  Matrix a = Matrix::fromCPU({{6, 6, 6}, {2, 2, 2}});
  Matrix b = Matrix::fromCPU({{1, 1, 1}, {1, 1, 1}, {1, 1, 1}});

  Matrix c = a.multiply(b);

  CHECK(matEq(c, {{18, 18, 18}, {6, 6, 6}}));
}

TEST_CASE("Invalid multiplication") {
  Matrix a = Matrix::fromCPU({{1, 2}, {3, 4}});
  Matrix b = Matrix::fromCPU({{1, 2}, {3, 4}, {5, 6}});

  CHECK_THROWS(a.multiply(b));
}

TEST_CASE("Transposition") {
  CHECK(matEq(Matrix::fromCPU({{1, 2}}).transpose(), {{1}, {2}}));
  CHECK(matEq(Matrix::fromCPU({{1, 2}, {3, 4}}).transpose(), {{1, 3}, {2, 4}}));
}
