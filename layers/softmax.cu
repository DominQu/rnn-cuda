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

__global__ void dnormalize(const MatrixValType *input,MatrixValType *result, MatrixSize inputsize) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    MatrixValType max = 0;
    if(i < inputsize.total) {
      
      for(int t = 0; t < inputsize.total; t++) {
          if(input[t] > max) {
              max = input[t];
          }
      }
      __syncthreads();
      result[i] = input[i] - max;
    }
}

GPUMatrix Softmax::normalize(const GPUMatrix &input) {
  GPUMatrix result = GPUMatrix::like(input);
  int ThreadsPerSM = input.getSize().total % 1024;
  int SMs = input.getSize().total / 1024 + 1;
  dnormalize<<<SMs, ThreadsPerSM>>>(input.gpuHandle(), result.gpuHandle(), input.getSize());
  result.syncGPU();
  return result;
}

GPUMatrix Softmax::forward(const GPUMatrix &input) {
  GPUMatrix result = this->normalize(input);
  MatrixValType *sum;
  cudaMalloc(&sum, sizeof(MatrixValType));
  dsoftmaxforward<<<SMs, ThreadsPerSM>>>(result.gpuHandle(), result.gpuHandle(), input.getSize(), sum);
  result.syncGPU();
  cudaFree(sum);
  return result;
}

std::vector<GPUMatrix> Softmax::forward(const std::vector<GPUMatrix> &input, bool return_sequence) {
  std::vector<GPUMatrix> output;
  for(auto &logits : input) {
    GPUMatrix result = this->normalize(logits);
    MatrixValType *sum;
    cudaMalloc(&sum, sizeof(MatrixValType));
    dsoftmaxforward<<<SMs, ThreadsPerSM>>>(result.gpuHandle(), result.gpuHandle(), logits.getSize(), sum);
    result.syncGPU();
    cudaFree(sum);
    output.push_back(result);
  }
  return output;
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