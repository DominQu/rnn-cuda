#include "cuda.hpp"
#include "matrix.hpp"

__global__
void init_gpu(MatrixValType *matrix, MatrixSize size,
                         const MatrixValType val) {
  const auto i = (blockIdx.x * blockDim.x + threadIdx.x);

  if (i >= size.total)
    return;

  matrix[i] = val;
}

__global__
void multiply_gpu(MatrixValType *matrix, const MatrixSize size,
                             const MatrixValType scalar) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= size.total)
    return;

  matrix[i] = matrix[i] * scalar;
}

__global__
void multiply_gpu(MatrixValType *in1, const MatrixSize in1Size,
                             MatrixValType *in2, const MatrixSize in2Size,
                             MatrixValType *out, const MatrixSize outSize) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  std::size_t y = i / outSize.width;
  std::size_t x = i - (y * outSize.width);

  if (i >= outSize.total)
    return;

  out[y * outSize.height + x] = 0;

  for (int j = 0; j < in1Size.width; j++) {
    out[y * outSize.width + x] +=
        in1[y * in1Size.width + j] * in2[j * in2Size.width + x];
  }
}

__global__
void transpose_gpu(MatrixValType* in, const MatrixSize inSize, MatrixValType* out, const MatrixSize outSize) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  std::size_t y = i / outSize.width;
  std::size_t x = i - (y * outSize.width);

  if (i >= outSize.total)
    return;

  out[y * outSize.width + x] = in[x * inSize.width + y];
}

Matrix::~Matrix() { CUDA_CALL(cudaFree(this->gpuData)); }

Matrix::Matrix(const MatrixSize &size) : size(size) {
  CUDA_CALL(cudaMalloc(&this->gpuData, size.total * sizeof(MatrixValType)));
}

Matrix::Matrix(const MatrixSize &size, const MatrixValType val) : Matrix(size) {
  init_gpu<<<size.total / 32 + 1, 32>>>(this->gpuData, size, val);
  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaDeviceSynchronize());
}

Matrix::Matrix(const Matrix &copied) : size(copied.size) {
  MatrixValType *copied_gpu_data;
  cudaMalloc(&copied_gpu_data, copied.size.total * sizeof(MatrixValType));
  cudaMemcpy(copied_gpu_data, copied.gpuData,
             size.total * sizeof(MatrixValType), cudaMemcpyDeviceToDevice);
  this->gpuData = copied_gpu_data;
}

Matrix Matrix::transpose() const {
  Matrix result(MatrixSize(size.width, size.height));

  transpose_gpu<<<size.total / 32 + 1, 32>>>(this->gpuData, size, result.gpu_handle(), result.getSize());
  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaDeviceSynchronize());

  return result;
}

void Matrix::multiply(const MatrixValType scalar) {
  multiply_gpu<<<size.total / 32 + 1, 32>>>(this->gpuData, this->size, scalar);
  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaDeviceSynchronize());
}

Matrix Matrix::multiply(const Matrix &other) const {
  if (size.width != other.size.height)
    throw new InvalidMatrixSize(
        "Current matrix width does not match other matrix height");

  Matrix out(MatrixSize(size.height, other.size.width));

  multiply_gpu<<<size.total / 32 + 1, 32>>>(gpuData, size, other.gpuData,
                                            other.size, out.gpuData, out.size);
  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaDeviceSynchronize());

  return out;
}

void Matrix::multiply(const Matrix &other, Matrix &out) const {
  if (size.width != other.size.height)
    throw new InvalidMatrixSize(
        "Current matrix width does not match other matrix height");

  if (size.height != out.size.height)
    throw new InvalidMatrixSize(
        "Current matrix height does not match result matrix height");

  if (other.size.width != out.size.width)
    throw new InvalidMatrixSize(
        "Other matrix width does not match result matrix width");

  multiply_gpu<<<size.total / 32 + 1, 32>>>(gpuData, size, other.gpuData,
                                            other.size, out.gpuData, out.size);
  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaDeviceSynchronize());
}

CPUMatrix Matrix::toCPU() const {
  CPUMatrix matrix;
  matrix.reserve(size.height);

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

void Matrix::show() const {
  const auto matrix = this->toCPU();

  for (std::size_t i = 0; i < matrix.size(); i++) {
    for (std::size_t j = 0; j < matrix[0].size(); j++) {
      std::cout << " " << matrix[i][j] << " ";
    }
    std::cout << "\n";
  }
}
