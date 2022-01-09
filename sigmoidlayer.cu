#include "cuda.hpp"
#include "sigmoidlayer.hpp"

__device__ MatrixValType dsigmoid(MatrixValType x) {
  return (MatrixValType)1.0 / ( (MatrixValType)1.0 + exp(-x));
}

__global__ void dsigmoidlayerforward(MatrixValType *input, MatrixValType *output, MatrixSize input_size) {

  const auto i = (blockIdx.x * blockDim.x + threadIdx.x);

  if (i < input_size.total) {
    output[i] = dsigmoid(input[i]);
  }
    
}

Matrix& SigmoidLayer::forward(Matrix& input) {
  this->input = input;
  this->output = Matrix(input.getSize());
  
  dsigmoidlayerforward<<<input.groupSize(), input.threadSize()>>>(input.gpuData,
                                                                  output.gpuData,
                                                                  input.size);
  CUDA_CALL(cudaGetLastError());

  return output;
}

