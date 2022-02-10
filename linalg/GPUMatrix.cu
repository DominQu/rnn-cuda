#include "CPUMatrix.hpp"
#include "GPUMatrix.hpp"

#include <cstddef>
#include <cstdio>
#include <initializer_list>

#define CUDA_CALL(x)                                                           \
  {                                                                            \
    cudaError_t cuda_error__ = (x);                                            \
    if (cuda_error__)                                                          \
      std::cout << "CUDA error: " #x " returned "                              \
                << cudaGetErrorString(cuda_error__) << std::endl;              \
  }

/* Global GPU variables */
static int ThreadsPerSM = 0;
static int SMs = 0;

/* Contructor and destructor */

__global__ void loadVars() {}

__global__ void init_gpu(MatrixValType *matrix, MatrixSize size,
                         const MatrixValType val) {
  for (auto mult = 1;; mult++) {
    const auto i = ((blockIdx.x + 1) * mult - 1) * blockDim.x + threadIdx.x;

    if (i >= size.total)
      return;

    matrix[i] = val;
  }
}

GPUMatrix::~GPUMatrix() {
  if (this->gpuData != nullptr) {
    CUDA_CALL(cudaFree(this->gpuData));
    this->gpuData = nullptr;
  }
}

GPUMatrix::GPUMatrix(const MatrixSize &size) : size(size) {
  if (SMs == 0) {
    cudaDeviceGetAttribute(&ThreadsPerSM, cudaDevAttrMaxThreadsPerBlock, 0);
    cudaDeviceGetAttribute(&SMs, cudaDevAttrMultiProcessorCount, 0);
  }

  CUDA_CALL(
      cudaMalloc(&this->gpuData, this->size.total * sizeof(MatrixValType)));
}

GPUMatrix::GPUMatrix(const MatrixSize &size, const MatrixValType val)
    : GPUMatrix(size) {
  this->syncGPU();
  init_gpu<<<SMs, ThreadsPerSM>>>(this->gpuData, this->size, val);
}

GPUMatrix::GPUMatrix(const GPUMatrix &copied) : GPUMatrix(copied.size) {
  copied.syncGPU();
  cudaMemcpy(this->gpuData, copied.gpuData,
             this->size.total * sizeof(MatrixValType),
             cudaMemcpyDeviceToDevice);
}

GPUMatrix::GPUMatrix(GPUMatrix &&moved) {
  this->gpuData = moved.gpuData;
  this->size = moved.size;
  moved.gpuData = nullptr;
}

/* Misc */

CPUMatrix GPUMatrix::toCPU() const {
  CPUMatrix matrix(this->getSize());

  cudaMemcpy(matrix.cpuHandle(), this->gpuData,
             this->getSize().total * sizeof(MatrixValType),
             cudaMemcpyDeviceToHost);

  return matrix;
}

GPUMatrix GPUMatrix::from(
    const std::initializer_list<std::initializer_list<MatrixValType>> &input) {
  GPUMatrix m(MatrixSize(input.size(), input.begin()[0].size()));

  m.syncGPU();

  for (std::size_t y = 0; y < input.size(); y++) {
    cudaMemcpy(&m.gpuData[y * m.size.width], input.begin()[y].begin(),
               m.size.width * sizeof(MatrixValType), cudaMemcpyHostToDevice);
  }

  return m;
}

GPUMatrix GPUMatrix::from(const CPUMatrix &input) {
  GPUMatrix m(input.getSize());

  m.syncGPU();
  CUDA_CALL(cudaMemcpy(m.gpuData, input.cpuHandle(),
                       m.getSize().total * sizeof(MatrixValType),
                       cudaMemcpyHostToDevice));

  return m;
}

void GPUMatrix::syncGPU() const {
  // TODO: Device synchronize should be called on stream
  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaDeviceSynchronize());
}

/* Random init */

__device__ unsigned int a = 1337;
__device__ unsigned int b = 777;
__device__ unsigned int globalSeed = 0;

__device__ MatrixValType random(const int seed, const MatrixValType min,
                                const MatrixValType max) {
  const float factor = 100;

  MatrixValType r = ((globalSeed + seed) * a + b) %
                    static_cast<unsigned int>((max - min) * factor);

  r /= factor;

  return r;
}

__global__ void random_gpu(MatrixValType *out, const MatrixSize size) {
  for (auto mult = 1;; mult++) {
    const auto i = ((blockIdx.x + 1) * mult - 1) * blockDim.x + threadIdx.x;

    if (i >= size.total)
      return;

    out[i] = random(i + 1, 0, 1);
  }
}

__global__ void shuffleSeed_gpu() { globalSeed = globalSeed * b + a; }

GPUMatrix GPUMatrix::random(const MatrixSize &size) {
  GPUMatrix result(size);

  result.syncGPU();
  random_gpu<<<SMs, ThreadsPerSM>>>(result.gpuHandle(), result.getSize());

  result.syncGPU();
  shuffleSeed_gpu<<<1, 1>>>();

  return result;
}

/* Adding */

__global__ void add_gpu(MatrixValType *in, MatrixValType *other,
                        MatrixValType *out, const MatrixSize size) {
  for (auto mult = 1;; mult++) {
    const auto i = ((blockIdx.x + 1) * mult - 1) * blockDim.x + threadIdx.x;

    if (i >= size.total)
      return;

    out[i] = in[i] + other[i];
  }
}

void GPUMatrix::add(const GPUMatrix &other, GPUMatrix &result) const {
  if (result.size.width != this->size.width ||
      other.size.width != this->size.width)
    throw new InvalidMatrixSize("Input GPUMatrix width is not valid");

  if (result.size.height != this->size.height ||
      other.size.height != this->size.height)
    throw new InvalidMatrixSize("Input GPUMatrix height is not valid");

  this->syncGPU();
  add_gpu<<<SMs, ThreadsPerSM>>>(this->gpuData, other.gpuData, result.gpuData,
                                 this->size);
}

GPUMatrix GPUMatrix::add(const GPUMatrix &other) const {
  GPUMatrix out(MatrixSize(this->size.height, this->size.width));
  this->add(other, out);
  return out;
}

__global__ void add_gpu(MatrixValType *in, MatrixValType scalar,
                        MatrixValType *out, const MatrixSize size) {
  for (auto mult = 1;; mult++) {
    const auto i = ((blockIdx.x + 1) * mult - 1) * blockDim.x + threadIdx.x;

    if (i >= size.total)
      return;

    out[i] = in[i] + scalar;
  }
}

void GPUMatrix::add(const MatrixValType scalar, GPUMatrix &result) const {
  if (result.size.width != this->size.width)
    throw new InvalidMatrixSize("Result GPUMatrix width is not valid");

  if (result.size.height != this->size.height)
    throw new InvalidMatrixSize("Result GPUMatrix height is not valid");

  this->syncGPU();
  add_gpu<<<SMs, ThreadsPerSM>>>(this->gpuData, scalar, result.gpuData,
                                 this->size);
}

GPUMatrix GPUMatrix::add(const MatrixValType scalar) const {
  GPUMatrix out(MatrixSize(this->size.height, this->size.width));
  this->add(scalar, out);
  return out;
}

/* Multiplication */

__global__ void multiply_gpu(MatrixValType *in, const MatrixValType scalar,
                             MatrixValType *out, const MatrixSize size) {
  for (auto mult = 1;; mult++) {
    const auto i = ((blockIdx.x + 1) * mult - 1) * blockDim.x + threadIdx.x;

    if (i >= size.total)
      return;

    out[i] = in[i] * scalar;
  }
}

void GPUMatrix::multiply(const MatrixValType scalar, GPUMatrix &out) const {
  if (this->size.height != out.size.height)
    throw new InvalidMatrixSize(
        "Current matrix height does not match result matrix height");

  if (this->size.width != out.size.width)
    throw new InvalidMatrixSize(
        "Current matrix width does not match result matrix width");

  this->syncGPU();
  multiply_gpu<<<SMs, ThreadsPerSM>>>(this->gpuData, scalar, out.gpuData,
                                      this->size);
}

GPUMatrix GPUMatrix::multiply(const MatrixValType scalar) const {
  GPUMatrix out(this->size);
  this->multiply(scalar, out);
  return out;
}

__global__ void multiply_gpu(const MatrixValType *in1, const MatrixSize in1Size,
                             const MatrixValType *in2, const MatrixSize in2Size,
                             MatrixValType *out, const MatrixSize outSize) {
  for (auto mult = 1;; mult++) {
    const auto i = ((blockIdx.x + 1) * mult - 1) * blockDim.x + threadIdx.x;

    std::size_t y = i / outSize.width;
    std::size_t x = i - (y * outSize.width);

    if (i >= outSize.total)
      return;

    MatrixValType tmp = 0;

    for (int j = 0; j < in1Size.width; j++) {
      tmp += in1[y * in1Size.width + j] * in2[j * in2Size.width + x];
    }

    out[y * outSize.width + x] = tmp;
  }
}

void GPUMatrix::multiply(const GPUMatrix &other, GPUMatrix &out) const {
  if (this->size.width != other.size.height)
    throw new InvalidMatrixSize(
        "Current matrix width does not match other matrix height");

  if (this->size.height != out.size.height)
    throw new InvalidMatrixSize(
        "Current matrix height does not match result matrix height");

  if (other.size.width != out.size.width)
    throw new InvalidMatrixSize(
        "Other matrix width does not match result matrix width");

  this->syncGPU();
  multiply_gpu<<<SMs, ThreadsPerSM>>>(this->gpuHandle(), this->getSize(),
                                      other.gpuHandle(), other.getSize(),
                                      out.gpuData, out.size);
}

GPUMatrix GPUMatrix::multiply(const GPUMatrix &other) const {
  GPUMatrix out(MatrixSize(this->size.height, other.size.width));
  this->multiply(other, out);
  return out;
}

/* Transposition */

__global__ void transpose_gpu(const MatrixValType *in, const MatrixSize inSize,
                              MatrixValType *out, const MatrixSize outSize) {
  for (auto mult = 1;; mult++) {
    const auto i = ((blockIdx.x + 1) * mult - 1) * blockDim.x + threadIdx.x;

    std::size_t y = i / outSize.width;
    std::size_t x = i - (y * outSize.width);

    if (i >= outSize.total)
      return;

    out[y * outSize.width + x] = in[x * inSize.width + y];
  }
}

void GPUMatrix::transpose(GPUMatrix &result) const {
  if (this->getSize().width != result.getSize().height ||
      this->getSize().height != result.getSize().width)
    throw new InvalidMatrixSize("Result GPUMatrix does not have proper size");

  this->syncGPU();
  transpose_gpu<<<SMs, ThreadsPerSM>>>(this->gpuHandle(), this->getSize(),
                                       result.gpuHandle(), result.getSize());
}

GPUMatrix GPUMatrix::transpose() const {
  GPUMatrix result(MatrixSize(this->size.width, this->size.height));
  this->transpose(result);
  return result;
}

/* Elementwise multiplication */

__global__ void
multiplyelementwise_gpu(const MatrixValType *in1, const MatrixSize in1Size,
                        const MatrixValType *in2, const MatrixSize in2Size,
                        MatrixValType *out, const MatrixSize outSize) {
  for (auto mult = 1;; mult++) {
    const auto i = ((blockIdx.x + 1) * mult - 1) * blockDim.x + threadIdx.x;

    if (i >= outSize.total) {
      return;
    }

    out[i] = in1[i] * in2[i];
  }
}

void GPUMatrix::multiplyelementwise(const GPUMatrix &other,
                                    GPUMatrix &result) const {
  if (this->getSize().height != other.getSize().height ||
      this->getSize().width != other.getSize().width)
    throw new InvalidMatrixSize(
        "Current matrix dimensions does not match other matrix dimensions");
  if (this->getSize().height != result.getSize().height ||
      this->getSize().width != result.getSize().width)
    throw new InvalidMatrixSize(
        "Current matrix dimensions does not match result matrix dimensions");

  this->syncGPU();
  multiplyelementwise_gpu<<<SMs, ThreadsPerSM>>>(
      this->gpuHandle(), this->getSize(), other.gpuHandle(), other.getSize(),
      result.gpuData, result.size);
}

GPUMatrix GPUMatrix::multiplyelementwise(const GPUMatrix &other) const {
  GPUMatrix result(MatrixSize(this->size.height, this->size.width));
  this->multiplyelementwise(other, result);
  return result;
}

/* Sqrt */
__global__ void sqrt_gpu(const MatrixValType *in, MatrixValType *out,
                         const MatrixSize size) {
  for (auto mult = 1;; mult++) {
    const auto i = ((blockIdx.x + 1) * mult - 1) * blockDim.x + threadIdx.x;

    if (i >= size.total)
      return;

    out[i] = sqrtf(in[i]);
  }
}

void GPUMatrix::sqrt(GPUMatrix &result) const {
  if (this->getSize().height != result.getSize().height ||
      this->getSize().width != result.getSize().width)
    throw new InvalidMatrixSize(
        "Current matrix dimensions does not match result matrix dimensions");

  this->syncGPU();
  sqrt_gpu<<<SMs, ThreadsPerSM>>>(this->gpuHandle(), result.gpuHandle(),
                                  this->getSize());
}

GPUMatrix GPUMatrix::sqrt() const {
  GPUMatrix result = GPUMatrix::like(*this);
  this->sqrt(result);
  return result;
}

/* Elementwise division */

__global__ void divideelementwise_gpu(const MatrixValType *in1,
                                      const MatrixValType *in2,
                                      MatrixValType *out,
                                      const MatrixSize outSize) {
  for (auto mult = 1;; mult++) {
    const auto i = ((blockIdx.x + 1) * mult - 1) * blockDim.x + threadIdx.x;

    if (i >= outSize.total) {
      return;
    }

    out[i] = in1[i] / in2[i];
  }
}

void GPUMatrix::divideelementwise(const GPUMatrix &other,
                                  GPUMatrix &result) const {
  if (this->getSize().height != other.getSize().height ||
      this->getSize().width != other.getSize().width)
    throw new InvalidMatrixSize(
        "Current matrix dimensions does not match other matrix dimensions");
  if (this->getSize().height != result.getSize().height ||
      this->getSize().width != result.getSize().width)
    throw new InvalidMatrixSize(
        "Current matrix dimensions does not match result matrix dimensions");

  this->syncGPU();
  divideelementwise_gpu<<<SMs, ThreadsPerSM>>>(
      this->gpuHandle(), other.gpuHandle(), result.gpuHandle(), result.size);
}

GPUMatrix GPUMatrix::divideelementwise(const GPUMatrix &other) const {
  GPUMatrix result = GPUMatrix::like(*this);
  this->divideelementwise(other, result);
  return result;
}
