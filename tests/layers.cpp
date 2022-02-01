#include "doctest.h"
#include "layers/tanhlayer.hpp"

TEST_CASE("tanh") {
  SUBCASE("tanh forward") {
    const auto tanh = TanhLayaer::Tanhlayer();
    const auto a = GPUMatrix::from({{0, 0}, {0, 0}});
    const auto b = CPUMatrix::from({{0, 0}, {0, 0}});
    const auto c = GPUMatrix::from({{1, 1}, {1, 1}});

    CHECK(tanh.forward(a).toCPU() == b);
    CHECK(tanh.forward(a).toCPU()[0] > 0 and tanh.forward(a).toCPU()[0] < 1);
  }
}