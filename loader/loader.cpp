#include "loader.hpp"
#include "linalg/CPUMatrix.hpp"
#include "linalg/GPUMatrix.hpp"
#include "linalg/common.hpp"
#include <cstddef>
#include <fstream>
#include <istream>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>

OneHot::OneHot(std::istream &inputStream) {
  while (!inputStream.eof() && !inputStream.fail()) {
    char character;
    inputStream.get(character);
    if (this->onehotMap[character] == 0)
      this->onehotMap[character] = this->onehotMap.size();
  }
}

CPUMatrix OneHot::encode(const char input) const {
  CPUMatrix result(MatrixSize(this->onehotMap.size(), 1), 0);

  for (auto it = this->onehotMap.begin(); it != this->onehotMap.end(); ++it) {
    if (it->first == input) {
      result.at(it->second - 1, 0) = 1;
      return result;
    }
  }

  std::string message;
  message += "Invalid OneHot input '";
  message += input;
  message += "' to encode !";
  throw std::runtime_error(message.c_str());
}

char OneHot::decode(const CPUMatrix &input) const {
  if (input.getSize().height != this->onehotMap.size())
    throw std::runtime_error("Onehot input to decode has invalid size");

  for (std::size_t y = 0; y < input.getSize().height; y++) {
    if (input.at(y, 0) == 1) {
      // This is some crazy map shit
      for (auto it = this->onehotMap.begin(); it != this->onehotMap.end();
           it++) {
        if (it->second == y + 1)
          return it->first;
      }
    }
  }

  throw std::runtime_error("Invalid Onehot input to decode");
}

DataLoader::DataLoader(const char *path)
  : path(path), inputFile(path, std::ios::binary), oh(inputFile) {
  this->loadDatasetSize();
}

void DataLoader::loadDatasetSize() {
  this->inputFile.clear();
  this->inputFile.seekg(0);
  this->datasetSize = 0;

  while (this->inputFile.get() != std::char_traits<char>::eof()) {
    this->datasetSize++;
  }
}

std::vector<GPUMatrix> DataLoader::getTrainBatch(const std::size_t N) {
  std::vector<GPUMatrix> result;
  result.reserve(N);
  
  std::random_device r;
  std::default_random_engine e1(r());
  std::uniform_int_distribution<int> uniform_dist(0, (this->datasetSize * this->trainPercentage) - N);
  std::size_t index = uniform_dist(e1);

  this->inputFile.clear();
  this->inputFile.seekg(index);

  // Back up for paragraph to start
  for (char c = ' '; this->inputFile.peek() != '\n' && this->inputFile.tellg() != 0; this->inputFile.unget()) { }
  // When found the newline, skip it
  if (this->inputFile.peek() == '\n') this->inputFile.get();
  
  // Load data into batch
  for (std::size_t i = 0; i < N; i++) {
    char c = this->inputFile.get();
    if (this->inputFile.eof()) break;
    result.emplace_back(GPUMatrix::from(oh.encode(c)));
  }

  result.shrink_to_fit();
  return result;
}

std::vector<GPUMatrix> DataLoader::getTestBatch() {
  std::vector<GPUMatrix> result;
  result.reserve((1.0 - this->trainPercentage) * this->datasetSize);

  const std::size_t index = this->trainPercentage * this->datasetSize + 1;

  this->inputFile.clear();
  this->inputFile.seekg(index);

  // Back up for paragraph to start
  for (char c = ' '; this->inputFile.peek() != '\n' && this->inputFile.tellg() != 0; this->inputFile.unget()) { }
  // When found the newline, skip it
  if (this->inputFile.peek() == '\n') this->inputFile.get();

  // Load data into batch
  while (true) {
    char c = this->inputFile.get();
    if (this->inputFile.eof()) break;
    result.emplace_back(GPUMatrix::from(oh.encode(c)));
  }

  result.shrink_to_fit();
  return result;
}

void DataLoader::show(std::ostream &outputStream) {
  outputStream << "Dataset from " << this->getPath() << " of "
               << this->getDatasetSize() << " characters ("
               << this->oh.getCharacterAmount() << " unique)\n";
}
