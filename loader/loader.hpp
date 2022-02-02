#pragma once

#include <istream>
#include <string>
#include <vector>
#include <map>

#include "linalg/CPUMatrix.hpp"

std::map<unsigned int, char> loadOnehotAssoc(const std::string&);

class OneHot {
public:
  OneHot(std::istream& inputStream);

  /// Encodes single character into Onehot GPUMatrix
  CPUMatrix encode(const char);
  /// Decodes from GPUMatrix into single character
  char decode(const CPUMatrix&);
private:
  std::map<char, unsigned int> onehotMap;
};
