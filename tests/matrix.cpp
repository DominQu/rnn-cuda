#include "linalg/CPUMatrix.hpp"
#include "linalg/GPUMatrix.hpp"

#include "doctest.h"
#include <sstream>

TEST_CASE("Constructors") {
  // TODO
}

TEST_CASE("CPU <-> GPU") {
  const auto a = CPUMatrix::from({{3, 6}, {1, 2}});
  const auto b = GPUMatrix::from(a);

  CHECK(a == b.toCPU());
}


TEST_CASE("Adding scalar to the Matrix") {
  const auto case1 = GPUMatrix(MatrixSize(8, 8), 1).add(5);

  CHECK(case1.toCPU() == CPUMatrix(MatrixSize(8, 8), 6));
}


TEST_CASE("Adding two Matrices") {
  const auto case1 =
    GPUMatrix::from({{1, 2}, {3, 4}}).add(GPUMatrix::from({{3, 2}, {1, 0}}));
  CHECK(case1.toCPU() == CPUMatrix::from({{4, 4}, {4, 4}}));

  CHECK_THROWS(GPUMatrix::from({{1, 2}}).add(GPUMatrix::from({{1}, {2}})));

  const auto case2 =
      GPUMatrix(MatrixSize(2, 2), 1).add(GPUMatrix::from({{1, 8}, {8, 1}}));

  CHECK(case2.toCPU() == CPUMatrix::from({{2, 9}, {9, 2}}));
}


TEST_CASE("Simple Multiplication") {
  auto a = GPUMatrix::from({{1, 3}, {4, 5}});
  auto b = GPUMatrix::from({{1}, {1}});

  GPUMatrix c = a.multiply(b);

  CHECK(c.toCPU() == CPUMatrix::from({{4}, {9}}));

  GPUMatrix d = c.multiply(10);

  CHECK(d.toCPU() == CPUMatrix::from({{40}, {90}}));
}

TEST_CASE("More multiplication") {
  GPUMatrix a = GPUMatrix::from({{6, 6, 6}, {2, 2, 2}});
  GPUMatrix b = GPUMatrix::from({{1, 1, 1}, {1, 1, 1}, {1, 1, 1}});

  GPUMatrix c = a.multiply(b);

  CHECK(c.toCPU() == CPUMatrix::from({{18, 18, 18}, {6, 6, 6}}));
}

TEST_CASE("Invalid multiplication") {
  auto a = GPUMatrix::from({{1, 2}, {3, 4}});
  auto b = GPUMatrix::from({{1, 2}, {3, 4}, {5, 6}});

  CHECK_THROWS(a.multiply(b));
}

TEST_CASE("Transposition") {
  CHECK(GPUMatrix::from({{1, 2}, {3, 4}}).transpose().transpose().toCPU() == CPUMatrix::from({{1, 2}, {3, 4}}));
  CHECK(GPUMatrix::from({{1, 2}}).transpose().toCPU() == CPUMatrix::from({{1}, {2}}));
  CHECK(GPUMatrix::from({{1, 2}, {3, 4}}).transpose().toCPU() == CPUMatrix::from({{1, 3}, {2, 4}}));
}

TEST_CASE("Elementwise multiplication") {
  GPUMatrix a = GPUMatrix::from({{3,4}, { 3, 4}});
  GPUMatrix b = GPUMatrix::from({{2,2}, { 2, 2}});

  GPUMatrix c = a.multiplyelementwise(b);

  CHECK(c.toCPU() == CPUMatrix::from({{6,8}, { 6, 8}}));
}

TEST_CASE("Serialization") {
  std::stringstream ss; // IRL this would be a file stream
  auto mat1 = CPUMatrix::from({{3, 4}, {8, 12}});
  auto mat2 = CPUMatrix::like(mat1);
  auto mat3 = CPUMatrix(MatrixSize(1, 2));
  mat1.serialize(ss);
  mat2.deSerialize(ss);

  CHECK_EQ(mat1, mat2);
  CHECK_THROWS(mat3.deSerialize(ss));
}
