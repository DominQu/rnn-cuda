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

  /// Loads batch of preferable (maximum) size of N.
  /// Resulting vector may be smaller than N.
  std::vector<GPUMatrix> getTrainBatch(const std::size_t N);

  /// Loads batch of test data (size depends on train percentage)
  std::vector<GPUMatrix> getTestBatch();
  
  inline const char* getPath() const {
    return this->path;
  }

  std::size_t getDatasetSize() const {
    return this->datasetSize;
  }

  const OneHot& getOneHot() const {
    return this->oh;
  }

  /// Print DataLoader in data friendly way
  void show(std::ostream&);

  /// Set train percentage, between <0, 1>
  void setTrainPercentage(const float trainPercentage) {
    this->trainPercentage = trainPercentage;
  }
private:
  void loadDatasetSize();
  
  const char* path;
  std::ifstream inputFile;
  std::size_t datasetSize;
  OneHot oh;

  float trainPercentage = 0.9;
};
