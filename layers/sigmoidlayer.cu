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
  //TODO: fix return because it is a reference to local variable
  Matrix output(input.getSize());

  dim3 num_threads = 32;
  dim3 num_blocks = num_threads.x / input.getSize().total + 1;
  dsigmoidlayerforward<<<num_blocks, num_threads>>>(input.gpuHandle(),
                                                    output.gpuHandle(),
                                                    input.getSize());
  CUDA_CALL(cudaGetLastError());

  return output;
}

