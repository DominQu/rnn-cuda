#include "matrix.hpp"
#include <cstddef>
#include <iostream>

__global__
void init(MatrixValType* matrix, const MatrixSize size, const MatrixValType val) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= size.total) return;
  
  matrix[i] = val;
}

Matrix::~Matrix() {
  CUDA_CALL(cudaFree(this->gpuData));
}

Matrix::Matrix(const MatrixSize& size) : size(size) {
  CUDA_CALL(cudaMalloc(&this->gpuData, size.total * sizeof(MatrixValType)));
}

Matrix::Matrix(const MatrixSize& size, const MatrixValType val) : Matrix(size) {
  init<<<(size.height * size.width) / 32 + 1, 32>>>(this->gpuData, this->size, val);
  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaDeviceSynchronize());
}

void Matrix::show() const {
  MatrixValType* val = new MatrixValType[size.total];
  cudaMemcpy(val, this->gpuData, size.total * sizeof(MatrixValType), cudaMemcpyDeviceToHost);

  for (int i = 0; i < size.height; i++) {
    for (int j = 0; j < size.height; j++) {
      std::cout << " " << val[i*size.height + j] << " ";
    }
    std::cout << "\n";
  }
}
