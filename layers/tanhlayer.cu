#include "cuda.hpp"
#include "tanhlayer.hpp"

__global__ void dtanhlayerforward(MatrixValType *input, MatrixValType *output, MatrixSize input_size) {
  const auto i = (blockIdx.x * blockDim.x + threadIdx.x);

  if (i < input_size.total) {
    output[i] = tanhf(input[i]);
  }
}

Matrix& TanhLayer::forward(Matrix& input) {
  Matrix output(input.getSize());

  dim3 num_threads = 32;
  dim3 num_blocks = num_threads.x / input.getSize().total + 1;
  dtanhlayerforward<<<num_blocks, num_threads>>>(input.gpuHandle(),
                                                 output.gpuHandle(),
                                                 input.getSize());
  CUDA_CALL(cudaGetLastError());

  return output;
}