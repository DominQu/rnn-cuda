#pragma once

#include <cstddef>
#include <fstream>
#include <istream>
#include <ostream>
#include <string>
#include <vector>
#include <map>

#include "linalg/CPUMatrix.hpp"
#include "linalg/GPUMatrix.hpp"

class OneHot {
public:
  OneHot(std::istream& inputStream);

  /// Encodes single character into Onehot GPUMatrix
  CPUMatrix encode(const char) const;
  /// Decodes from GPUMatrix into single character
  char decode(const CPUMatrix&) const;

  inline std::size_t getCharacterAmount() const {
    return this->onehotMap.size();
  }
  
private:
  std::map<char, unsigned int> onehotMap;
};

class DataLoader {
public:
  DataLoader(const char* path);

  std::vector<GPUMatrix> getBatch(const std::size_t N);

  inline const char* getPath() const {
    return this->path;
  }

  std::size_t getDatasetSize() const {
    return this->datasetSize;
  }

  void show(std::ostream&);
private:
  void loadDatasetSize();
  
  const char* path;
  std::ifstream inputFile;
  std::size_t datasetSize;
  OneHot oh;
};
