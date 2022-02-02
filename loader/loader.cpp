#include "loader.hpp"
#include "linalg/CPUMatrix.hpp"
#include "linalg/common.hpp"
#include <cstddef>
#include <istream>
#include <stdexcept>

OneHot::OneHot(std::istream& inputStream) {
  while (!inputStream.eof()) {
    const auto character = inputStream.get();

    if (character != -1 && this->onehotMap[character] == 0)
      this->onehotMap[character] = this->onehotMap.size();
  }
}

CPUMatrix OneHot::encode(const char input) {
  CPUMatrix result(MatrixSize(this->onehotMap.size(), 1), 0);

  for (auto it = this->onehotMap.begin(); it != this->onehotMap.end(); ++it) {
    if (it->first == input) {
      result.at(it->second - 1, 0) = 1;
      return result;
    }
  }

  throw new std::runtime_error("Invalid Onehot input to encode");
}

char OneHot::decode(const CPUMatrix& input) {
  if (input.getSize().height != this->onehotMap.size())
    throw new std::runtime_error("Onehot input to decode has invalid size");
  
  for (std::size_t y = 0; y < input.getSize().height; y++) {
    if (input.at(y, 0) == 1) {
      // This is some crazy map shit
      return std::next(this->onehotMap.begin(), y)->first;
    }
  }

  throw new std::runtime_error("Invalid Onehot input to decode");
}

