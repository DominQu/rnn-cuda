#include "CPUMatrix.hpp"
#include "linalg/common.hpp"
#include <cstddef>
#include <random>

CPUMatrix::CPUMatrix(const MatrixSize &size) {
  this->cpuData = new MatrixValType[size.total];
  this->size = size;
}

CPUMatrix::CPUMatrix(const CPUMatrix &copied) : CPUMatrix(copied.getSize()) {
  for (std::size_t i = 0; i < this->size.total; i++) {
    this->cpuData[i] = copied.cpuData[i];
  }
}

CPUMatrix::CPUMatrix(CPUMatrix &&moved) {
  this->cpuData = moved.cpuData;
  this->size = moved.size;
  moved.cpuData = nullptr;
}

CPUMatrix::~CPUMatrix() {
  if (this->cpuData != nullptr) {
    delete[] this->cpuData;
    this->cpuData = nullptr;
  }
}

CPUMatrix::CPUMatrix(const MatrixSize &size, const MatrixValType val)
    : CPUMatrix(size) {
  for (std::size_t i = 0; i < this->size.total; i++) {
    this->cpuData[i] = val;
  }
}

CPUMatrix
CPUMatrix::from(const std::vector<std::vector<MatrixValType>> &other) {
  CPUMatrix result(MatrixSize(other.size(), other[0].size()));

  for (std::size_t i = 0; i < result.size.total; i++) {
    result.cpuData[i] = other[i / other[0].size()][i % other[0].size()];
  }

  return result;
}

CPUMatrix CPUMatrix::random(const MatrixSize size, const MatrixValType min,
                            const MatrixValType max) {
  CPUMatrix result(size);

  static std::random_device dev;
  static std::mt19937 gen(dev());
  std::uniform_real_distribution<MatrixValType> distribution(min, max);

  for (std::size_t i = 0; i < size.total; i++) {
    result.cpuData[i] = distribution(gen);
  };

  return result;
}

MatrixValType &CPUMatrix::at(const std::size_t y, const std::size_t x) {
  return this->cpuData[this->coordsFrom(y, x)];
}

const MatrixValType CPUMatrix::at(const std::size_t y,
                                  const std::size_t x) const {
  return this->cpuData[this->coordsFrom(y, x)];
}

bool CPUMatrix::operator==(const CPUMatrix &other) const {
  if (this->getSize() != other.getSize())
    return false;

  for (auto i = 0; i < this->getSize().total; i++) {
    if (this->cpuData[i] != other.cpuData[i])
      return false;
  }

  return true;
}

void CPUMatrix::show(std::ostream &outStream) const {
  for (std::size_t y = 0; y < this->getSize().height; y++) {
    for (std::size_t x = 0; x < this->getSize().width; x++) {
      outStream << this->cpuData[y * this->getSize().width + x];
      if (x + 1 != this->getSize().width)
        outStream << ";";
    }
    outStream << "\n";
  }
}

void CPUMatrix::serialize(std::ostream &output) const {
  output.write(reinterpret_cast<const char *>(&(this->size)),
               sizeof(MatrixSize) / sizeof(char));
  output.write(reinterpret_cast<const char *>(this->cpuData),
               (sizeof(MatrixValType) * this->getSize().total) / sizeof(char));
}

void CPUMatrix::deSerialize(std::istream &input) {
  MatrixSize size;
  input.read(reinterpret_cast<char *>(&size),
             sizeof(MatrixSize) / sizeof(char));

  if (size.width != this->getSize().width ||
      size.height != this->getSize().height ||
      size.total != this->getSize().total)
    throw InvalidMatrixSize("Matrix provided in the stream, has invalid size");

  input.read(reinterpret_cast<char *>(this->cpuData),
             (sizeof(MatrixValType) * this->getSize().total) / sizeof(char));
}

CPUMatrix CPUMatrix::multiply(const MatrixValType scalar) const {
  CPUMatrix result(this->getSize());

  this->multiply(scalar, result);

  return result;
}

void CPUMatrix::multiply(const MatrixValType scalar, CPUMatrix &result) const {
  for (std::size_t i = 0; i < this->getSize().total; i++) {
    result.cpuData[i] = this->cpuData[i] * scalar;
  }
}

CPUMatrix CPUMatrix::multiply(const CPUMatrix &other) const {
  CPUMatrix result(MatrixSize(this->getSize().height, other.getSize().width));

  this->multiply(other, result);

  return result;
}

void CPUMatrix::multiply(const CPUMatrix &other, CPUMatrix &result) const {
  // TODO: Add tests for sizes

  for (std::size_t y = 0; y < result.getSize().height; y++) {
    for (std::size_t x = 0; x < result.getSize().width; x++) {
      MatrixValType val = 0;
      for (std::size_t k = 0; k < this->getSize().width; k++) {
        val += this->cpuData[this->coordsFrom(y, k)] *
               other.cpuData[other.coordsFrom(k, x)];
      }
      result.cpuData[result.coordsFrom(y, x)] = val;
    }
  }
}

CPUMatrix CPUMatrix::add(const MatrixValType scalar) const {
  CPUMatrix result(this->getSize());

  this->add(scalar, result);

  return result;
}

void CPUMatrix::add(const MatrixValType scalar, CPUMatrix &result) const {
  for (std::size_t i = 0; i < this->getSize().total; i++) {
    result.cpuData[i] = this->cpuData[i] + scalar;
  }
}

CPUMatrix CPUMatrix::add(const CPUMatrix &other) const {
  CPUMatrix result(this->getSize());

  this->add(other, result);

  return result;
}

void CPUMatrix::add(const CPUMatrix &other, CPUMatrix &result) const {
  for (std::size_t y = 0; y < result.getSize().height; y++) {
    for (std::size_t x = 0; x < result.getSize().width; x++) {
      result.cpuData[result.coordsFrom(y, x)] =
          this->cpuData[this->coordsFrom(y, x)] +
          other.cpuData[other.coordsFrom(y, x)];
    }
  }
}

CPUMatrix CPUMatrix::transpose() const {
  CPUMatrix result(MatrixSize(this->getSize().width, this->getSize().height));

  this->transpose(result);

  return result;
}

void CPUMatrix::transpose(CPUMatrix &result) const {
  for (std::size_t y = 0; y < result.getSize().height; y++) {
    for (std::size_t x = 0; x < result.getSize().width; x++) {
      result.cpuData[result.coordsFrom(y, x)] =
          this->cpuData[this->coordsFrom(x, y)];
    }
  }
}

int CPUMatrix::argmax() const {
  MatrixValType maxval = 0;
  int index = -1;
  for(int i = 0; i < this->getSize().height; i++) {
    if(this->at(i, 0) > maxval) {
      maxval = this->at(i,0);
      index = i;
    }
  }
  if(index == -1) {
    std::cout << "Invalid max index!";
    return 0;
  }
  return index;
}