#include "CCEloss.hpp"

__global__ void dcceforward(const MatrixValType *input, MatrixValType *result, int size) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if ( i < size) {
    result[i] = logf(input[i]);
  }
}

MatrixValType CCEloss::forward(const GPUMatrix &input,const GPUMatrix &label) {
  GPUMatrix crossentropy = GPUMatrix::like(input);
  int ThreadsPerSM = input.getSize().total % 1024;
  int SMs = input.getSize().total / 1024 + 1;
  dcceforward<<<SMs, ThreadsPerSM>>>(input.gpuHandle(), crossentropy.gpuHandle(), input.getSize().total);
  crossentropy.syncGPU();
  int index = -1;
  for(int i = 0; i < label.getSize().height; i++) {
    if( label.toCPU().at(i,0) == 1) {
      index = i;
      break;
    }
  }
  if(index == -1) {
    std::cout << "Wrong label" << std::endl;
    return 0;
  }
  return crossentropy.toCPU().at(index,0);
}

GPUMatrix CCEloss::backward(const GPUMatrix &softmax, const GPUMatrix &label) {
    return softmax.add(label.multiply(-1));
}