#include "sgd.hpp"

std::vector<GPUMatrix> SGD::calculateUpdate(std::vector<GPUMatrix> &gradients) {
  std::vector<GPUMatrix> result;
  for ( auto i : gradients) {
      result.push_back(i.multiply(this->learning_rate));
  }
  return result;
}