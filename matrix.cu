#include "cuda.hpp"
#include "matrix.hpp"

/* Contructor and destructor */

__global__ void init_gpu(MatrixValType *matrix, MatrixSize size,
                         const MatrixValType val) {
  const auto i = (blockIdx.x * blockDim.x + threadIdx.x);

  if (i >= size.total)
    return;

  matrix[i] = val;
}

Matrix::~Matrix() { CUDA_CALL(cudaFree(this->gpuData)); }

Matrix::Matrix(const MatrixSize &size) : size(size) {
  CUDA_CALL(
      cudaMalloc(&this->gpuData, this->size.total * sizeof(MatrixValType)));
}

Matrix::Matrix(const MatrixSize &size, const MatrixValType val) : Matrix(size) {
  init_gpu<<<this->groupSize(), this->threadSize()>>>(this->gpuData, this->size,
                                                      val);
  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaDeviceSynchronize());
}

Matrix::Matrix(const Matrix &copied) : size(copied.size) {
  MatrixValType *copied_gpu_data;

  cudaMalloc(&copied_gpu_data, copied.size.total * sizeof(MatrixValType));
  cudaMemcpy(copied_gpu_data, copied.gpuData,
             this->size.total * sizeof(MatrixValType),
             cudaMemcpyDeviceToDevice);

  this->gpuData = copied_gpu_data;
}

/* Misc */

CPUMatrix Matrix::toCPU() const {
  CPUMatrix matrix;
  matrix.reserve(this->size.height);

  for (int y = 0; y < size.height; y++) {
    matrix.push_back(std::vector<MatrixValType>());
    matrix[y].resize(size.width);

    cudaMemcpy(&(*matrix[y].begin()), &gpuData[y * size.width],
               size.width * sizeof(MatrixValType), cudaMemcpyDeviceToHost);
  }

  return matrix;
}

Matrix Matrix::fromCPU(const CPUMatrix &input) {
  Matrix m(MatrixSize(input.size(), input[0].size()));

  cudaMalloc(&m.gpuData, m.size.total * sizeof(MatrixValType));

  for (std::size_t y = 0; y < input.size(); y++) {
    cudaMemcpy(&m.gpuData[y * m.size.width], &(*input[y].begin()),
               m.size.width * sizeof(MatrixValType), cudaMemcpyHostToDevice);
  }

  return m;
}

// TODO: This should be moved to CPUMatrix function
void Matrix::show() const {
  const auto matrix = this->toCPU();

  for (std::size_t i = 0; i < matrix.size(); i++) {
    for (std::size_t j = 0; j < matrix[0].size(); j++) {
      std::cout << " " << matrix[i][j] << " ";
    }
    std::cout << "\n";
  }
}

/* Adding */

__global__ void add_gpu(MatrixValType *in, MatrixValType *other,
                        MatrixValType *out, const MatrixSize size) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= size.total)
    return;

  out[i] = in[i] + other[i];
}

void Matrix::add(const Matrix &other, Matrix &result) const {
  if (result.size.width != this->size.width ||
      other.size.width != this->size.width)
    throw new InvalidMatrixSize("Input Matrix width is not valid");

  if (result.size.height != this->size.height ||
      other.size.height != this->size.height)
    throw new InvalidMatrixSize("Input Matrix height is not valid");

  add_gpu<<<size.total / 32 + 1, 32>>>(this->gpuData, other.gpuData,
                                       result.gpuData, this->size);

  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaDeviceSynchronize());
}

Matrix Matrix::add(const Matrix &other) const {
  Matrix out(MatrixSize(this->size.height, this->size.width));
  this->add(other, out);
  return out;
}

__global__ void add_gpu(MatrixValType *in, MatrixValType scalar,
                        MatrixValType *out, const MatrixSize size) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= size.total)
    return;

  out[i] = in[i] + scalar;
}

void Matrix::add(const MatrixValType scalar, Matrix &result) const {
  if (result.size.width != this->size.width)
    throw new InvalidMatrixSize("Result Matrix width is not valid");

  if (result.size.height != this->size.height)
    throw new InvalidMatrixSize("Result Matrix height is not valid");

  add_gpu<<<size.total / 32 + 1, 32>>>(this->gpuData, scalar, result.gpuData,
                                       this->size);

  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaDeviceSynchronize());
}

Matrix Matrix::add(const MatrixValType scalar) const {
  Matrix out(MatrixSize(this->size.height, this->size.width));
  this->add(scalar, out);
  return out;
}

/* Multiplication */

__global__ void multiply_gpu(MatrixValType *in, const MatrixValType scalar,
                             MatrixValType *out, const MatrixSize size) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= size.total)
    return;

  out[i] = in[i] * scalar;
}

void Matrix::multiply(const MatrixValType scalar, Matrix &out) const {
  if (this->size.height != out.size.height)
    throw new InvalidMatrixSize(
        "Current matrix height does not match result matrix height");

  if (this->size.width != out.size.width)
    throw new InvalidMatrixSize(
        "Current matrix width does not match result matrix width");

  multiply_gpu<<<size.total / 32 + 1, 32>>>(this->gpuData, scalar, out.gpuData,
                                            this->size);
  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaDeviceSynchronize());
}

Matrix Matrix::multiply(const MatrixValType scalar) const {
  Matrix out(this->size);
  this->multiply(scalar, out);
  return out;
}

__global__ void multiply_gpu(const MatrixValType *in1, const MatrixSize in1Size,
                             const MatrixValType *in2, const MatrixSize in2Size,
                             MatrixValType *out, const MatrixSize outSize) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  std::size_t y = i / outSize.width;
  std::size_t x = i - (y * outSize.width);

  if (i >= outSize.total)
    return;

  MatrixValType tmp = 0;
  
  for (int j = 0; j < in1Size.width; j++) {
    tmp += in1[y * in1Size.width + j] * in2[j * in2Size.width + x];
  }

  out[y * outSize.width + x] = tmp;
}

void Matrix::multiply(const Matrix &other, Matrix &out) const {
  if (this->size.width != other.size.height)
    throw new InvalidMatrixSize(
        "Current matrix width does not match other matrix height");

  if (this->size.height != out.size.height)
    throw new InvalidMatrixSize(
        "Current matrix height does not match result matrix height");

  if (other.size.width != out.size.width)
    throw new InvalidMatrixSize(
        "Other matrix width does not match result matrix width");

  multiply_gpu<<<size.total / 32 + 1, 32>>>(this->gpuHandle(), this->getSize(),
                                            other.gpuHandle(), other.getSize(),
                                            out.gpuData, out.size);
  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaDeviceSynchronize());
}

Matrix Matrix::multiply(const Matrix &other) const {
  Matrix out(MatrixSize(this->size.height, other.size.width));
  this->multiply(other, out);
  return out;
}

/* Transposition */

__global__ void transpose_gpu(const MatrixValType *in, const MatrixSize inSize,
                              MatrixValType *out, const MatrixSize outSize) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  std::size_t y = i / outSize.width;
  std::size_t x = i - (y * outSize.width);

  if (i >= outSize.total)
    return;

  out[y * outSize.width + x] = in[x * inSize.width + y];
}

void Matrix::transpose(Matrix& result) const {
  if (this->getSize().width != result.getSize().height
      ||
      this->getSize().height != result.getSize().width)
    throw new InvalidMatrixSize("Result Matrix does not have proper size");

  transpose_gpu<<<size.total / 32 + 1, 32>>>(
      this->gpuHandle(), this->getSize(), result.gpuHandle(), result.getSize());
  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaDeviceSynchronize());
}

Matrix Matrix::transpose() const {
  Matrix result(MatrixSize(this->size.width, this->size.height));
  this->transpose(result);
  return result;
}
