#include "doctest.h"
#include "layers/lstmlayer.hpp"
#include "linalg/CPUMatrix.hpp"
#include "linalg/GPUMatrix.hpp"

TEST_CASE("lstm layer forward") {
  const int input_dim = 64;
  const int state_dim = 512;
  const int timesteps = 2;
  const int minweight = 0;
  const int maxweight = 1;
  LstmLayer layer(input_dim, state_dim, timesteps, minweight, maxweight);
  std::vector<CPUMatrix> batch;
  for(int t = 0; t < timesteps; t++) {
    batch.emplace_back(MatrixSize(input_dim,1), 1);
  }
  GPUMatrix output = layer.forward(batch);
  CHECK(output.getSize().height == input_dim);
}