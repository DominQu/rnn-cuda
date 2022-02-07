#include "softmax.hpp"
#include "iomanip"

static int ThreadsPerSM = 0;
static int SMs = 0;

Softmax::Softmax(int input_dim) : input_dim(input_dim) {
  if( SMs == 0) {
      ThreadsPerSM = input_dim % 1024;
      SMs = input_dim / 1024 + 1;
  }
}

__global__ void dsoftmaxforward(const MatrixValType *input, MatrixValType *result, MatrixSize inputsize, MatrixValType *sum) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  *sum = 0;

  if( i < inputsize.total) {
    MatrixValType value = exp(input[i]);
    atomicAdd(sum, value);
    __syncthreads();
    result[i] = exp(input[i]) / *sum;
  }
}

GPUMatrix Softmax::forward(const GPUMatrix &input) {
  GPUMatrix result = GPUMatrix::like(input);
  MatrixValType *sum;
  cudaMalloc(&sum, sizeof(MatrixValType));
  dsoftmaxforward<<<SMs, ThreadsPerSM>>>(input.gpuHandle(), result.gpuHandle(), input.getSize(), sum);
  result.syncGPU();
  cudaFree(sum);
  return result;
}

__global__ void dsoftmaxbackward(const MatrixValType *forward_result, MatrixValType *result, int size) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    int row = i / size;
    int col = i % size;

    if( i < size*size) {
        if ( row == col) {
            result[i] = forward_result[row] * (1 - forward_result[col]);
        }
        else {
            result[i] = -1 * forward_result[row] * forward_result[col];
        }
    }
}
/// Not to use
GPUMatrix Softmax::backward(const GPUMatrix &upstream, const GPUMatrix &forward_result) {
//   GPUMatrix softmax_gradient(MatrixSize(this->input_dim, this->input_dim), 0);
//   int sms = this->input_dim * this->input_dim / ThreadsPerSM + 1;
//   dsoftmaxbackward<<<sms, ThreadsPerSM>>>(forward_result.gpuHandle(), softmax_gradient.gpuHandle(), this->input_dim);

//   GPUMatrix result = softmax_gradient.multiply(upstream);
//   return result;
  return upstream;
}