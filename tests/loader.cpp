#include "doctest.h"
#include "linalg/CPUMatrix.hpp"
#include "loader/loader.hpp"
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

  m4.show(std::cout);
  
  CHECK_EQ('d', oh.decode(oh.encode('d')));
  CHECK_EQ('s', oh.decode(oh.encode('s')));
  CHECK_EQ('b', oh.decode(oh.encode('b')));
  
  CHECK_THROWS(oh.encode('x'));
  CHECK_THROWS(oh.decode(CPUMatrix::from({{0}, {0}, {0}, {1}})));
}
