#include "doctest.h"
#include "linalg/CPUMatrix.hpp"
#include "linalg/GPUMatrix.hpp"
#include "loader/loader.hpp"
#include <cstddef>
#include <iostream>
#include <sstream>

TEST_CASE("Onehot") {
  std::stringstream str("dbacsadcasd");

  OneHot oh(str);
  
  CPUMatrix m1 = oh.encode('b');
  CPUMatrix m2 = oh.encode('c');
  CPUMatrix m3 = oh.encode('a');
  CPUMatrix m4 = oh.encode('d');
  
  CHECK_NE(m1, m2);
  CHECK_NE(m1, m3);
  CHECK_NE(m2, m3);

  CHECK_EQ('d', oh.decode(oh.encode('d')));
  CHECK_EQ('s', oh.decode(oh.encode('s')));
  CHECK_EQ('b', oh.decode(oh.encode('b')));
  
  CHECK_THROWS(oh.encode('x'));
  CHECK_THROWS(oh.decode(CPUMatrix::from({{0}, {0}, {0}, {1}})));
}

TEST_CASE("Loading dataset") {
  const auto FILENAME = "/tmp/dataset.txt";
  const std::string TEST_STR = "Hello world !\nLine one\nLine two";
  
  std::ofstream outStream(FILENAME);
  outStream << TEST_STR;
  outStream.close();

  DataLoader dl(FILENAME);
  CHECK_EQ(dl.getDatasetSize(), TEST_STR.size());
  
  const auto trainBatch = dl.getTrainBatch(12);
  const auto testBatch  = dl.getTestBatch();

  CHECK_LE(trainBatch.size(), 12);

  // DeEncode batch and check if it is subset of TEST_STR
  std::string acc = "";
  for (const GPUMatrix& c: trainBatch) {
    acc += dl.getOneHot().decode(c.toCPU());
  }

  CHECK_NE(TEST_STR.find(acc), std::string::npos);

  acc = "";
  for (const GPUMatrix& c: testBatch) {
    acc += dl.getOneHot().decode(c.toCPU());
  }

  for (std::size_t i = 0; i < acc.size(); i++) {
    CHECK_EQ(acc[acc.size() - 1 - i], TEST_STR[TEST_STR.size() - 1 - i]);
  }
}
