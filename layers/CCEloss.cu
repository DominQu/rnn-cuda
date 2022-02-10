#include "CCEloss.hpp"

__global__ void dcceforward(const MatrixValType *input, MatrixValType *result, int size) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if ( i < size) {
    result[i] = -1 * logf(input[i]);
    // printf("Current loss %f", result[i]);
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

std::vector<MatrixValType> CCEloss::forward(const std::vector<GPUMatrix> &softmax,
                                            const GPUMatrix &label, 
                                            const std::vector<GPUMatrix> &batch) {
  std::vector<MatrixValType> result;
  int ThreadsPerSM = (softmax[0]).getSize().total % 1024;
  int SMs = (softmax[0]).getSize().total / 1024 + 1;
  for(int i = 0; i < batch.size() - 1; i++) {
    GPUMatrix crossentropy = GPUMatrix::like(softmax[i]);
    dcceforward<<<SMs, ThreadsPerSM>>>((softmax[i]).gpuHandle(), crossentropy.gpuHandle(), (softmax[i]).getSize().total);
    crossentropy.syncGPU();
    int index = -1;
    for(int ind = 0; ind < batch[i+1].getSize().height; ind++) {
      if( batch[i+1].toCPU().at(ind,0) == 1) {
        index = ind;
        break;
      }
    }
    if(index == -1) {
      std::cout << "Wrong label" << std::endl;
    }
    result.push_back(crossentropy.toCPU().at(index,0));
  }
  /// Loss for last element of the sequence
  GPUMatrix crossentropy = GPUMatrix::like(softmax.back());
  dcceforward<<<SMs, ThreadsPerSM>>>((softmax.back()).gpuHandle(), crossentropy.gpuHandle(), (softmax.back()).getSize().total);
  crossentropy.syncGPU();
  int index = -1;
  for(int ind = 0; ind < label.getSize().height; ind++) {
    if( label.toCPU().at(ind,0) == 1) {
      index = ind;
      break;
    }
  }
  if(index == -1) {
    std::cout << "Wrong label" << std::endl;
  }
  result.push_back(crossentropy.toCPU().at(index,0));

  return result;
}

GPUMatrix CCEloss::backward(const GPUMatrix &softmax, const GPUMatrix &label) {
    return softmax.add(label.multiply(-1));
}

std::vector<GPUMatrix> CCEloss::backward(std::vector<GPUMatrix> &softmax,
                                GPUMatrix &label, 
                                std::vector<GPUMatrix> &batch) {
  std::vector<GPUMatrix> result;
  for(int i = 0; i < softmax.size()-1; i++) {
    result.push_back(softmax[i].add(batch[i+1].multiply(-1)));
  }
  result.push_back(softmax.back().add(label.multiply(-1)));
  return result;
}