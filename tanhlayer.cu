#include "cuda.hpp"
#include "tanhlayer.hpp"

__global__ void dtanhlayerforward(MatrixValType *input, MatrixValType *output, MatrixSize input_size) {
  const auto i = (blockIdx.x * blockDim.x + threadIdx.x);

  if (i < input_size.total) {
    output[i] = tanhf(input[i]);
  }
}

Matrix& TanhLayer::forward(Matrix& input) {
    output = Matrix(input.getSize());

    dtanhlayerforward<<<input.groupSize(), input.threadSize()>>>(input.getData(),
                                                                 output.getData(),
                                                                 input.getSize());
    CUDA_CALL(cudaGetLastError());

    return output;
}