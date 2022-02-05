#include "cuda.hpp"
#include "lstmlayer.hpp"

static int ThreadsPerSM = 0;
static int SMs = 0;
  
LstmLayer::LstmLayer(int input_dim, int state_dim, 
                     int timesteps, int weight_min, 
                     int weight_max) 
  : input_dim(input_dim),
    state_dim(state_dim), 
    timesteps(timesteps),
    
    c(std::vector<GPUMatrix>()),
    tanh_c(std::vector<GPUMatrix>()),
    h(std::vector<GPUMatrix>()), 
    f(std::vector<GPUMatrix>()), 
    g(std::vector<GPUMatrix>()), 
    i(std::vector<GPUMatrix>()), 
    o(std::vector<GPUMatrix>()),
    input_weights_f(GPUMatrix::from(CPUMatrix::random(MatrixSize(state_dim, input_dim), 
                                                      weight_min, 
                                                      weight_max))),
    input_weights_g(GPUMatrix::from(CPUMatrix::random(MatrixSize(state_dim, input_dim), 
                                                      weight_min, 
                                                      weight_max))),
    input_weights_i(GPUMatrix::from(CPUMatrix::random(MatrixSize(state_dim, input_dim), 
                                                      weight_min, 
                                                      weight_max))),
    input_weights_o(GPUMatrix::from(CPUMatrix::random(MatrixSize(state_dim, input_dim), 
                                                      weight_min, 
                                                      weight_max))),
    state_weights_f(GPUMatrix::from(CPUMatrix::random(MatrixSize(state_dim, state_dim), 
                                                      weight_min, 
                                                      weight_max))), 
    state_weights_g(GPUMatrix::from(CPUMatrix::random(MatrixSize(state_dim, state_dim), 
                                                      weight_min, 
                                                      weight_max))), 
    state_weights_i(GPUMatrix::from(CPUMatrix::random(MatrixSize(state_dim, state_dim), 
                                                      weight_min, 
                                                      weight_max))), 
    state_weights_o(GPUMatrix::from(CPUMatrix::random(MatrixSize(state_dim, state_dim), 
                                                      weight_min, 
                                                      weight_max))), 
    output_weights(GPUMatrix::from(CPUMatrix::random(MatrixSize(input_dim, state_dim), 
                                                      weight_min, 
                                                      weight_max))) 
                                                     
{
  // Initilize carry and state with zeros
  c.emplace_back(MatrixSize(state_dim, 1), 0);
  h.emplace_back(MatrixSize(state_dim, 1), 0);

  for( int t = 0; t < timesteps; t++) {
    c.emplace_back(MatrixSize(state_dim, 1));
    tanh_c.emplace_back(MatrixSize(state_dim, 1));
    h.emplace_back(MatrixSize(state_dim, 1));
    f.emplace_back(MatrixSize(state_dim, 1));
    g.emplace_back(MatrixSize(state_dim, 1));
    i.emplace_back(MatrixSize(state_dim, 1));
    o.emplace_back(MatrixSize(state_dim, 1));
  }

  // Set number of blocks and threads
  if (SMs == 0) {
    cudaDeviceGetAttribute(&ThreadsPerSM, cudaDevAttrMaxThreadsPerBlock, 0);
    cudaDeviceGetAttribute(&SMs, cudaDevAttrMultiProcessorCount, 0);
  }
}

__device__ MatrixValType dsigmoid(MatrixValType x) {
  return (MatrixValType)1.0 / ( (MatrixValType)1.0 + exp(-x));
}

__global__ void dsigmoidlayerforward(MatrixValType *input, MatrixValType *output, MatrixSize input_size) {

  const auto i = (blockIdx.x * blockDim.x + threadIdx.x);

  if (i < input_size.total) {
    output[i] = dsigmoid(input[i]);
  }
    
}

void LstmLayer::applySigmoid(GPUMatrix &input, GPUMatrix &result) {

  input.syncGPU();
  dsigmoidlayerforward<<<SMs, ThreadsPerSM>>>(input.gpuHandle(),
                                              result.gpuHandle(),
                                              input.getSize());
  CUDA_CALL(cudaGetLastError());
}

__global__ void dtanhlayerforward(MatrixValType *input, MatrixValType *output, MatrixSize input_size) {
  const auto i = (blockIdx.x * blockDim.x + threadIdx.x);

  if (i < input_size.total) {
    output[i] = tanhf(input[i]);
  }
}

void LstmLayer::applyTanh(GPUMatrix &input, GPUMatrix &result) {

  input.syncGPU();
  dtanhlayerforward<<<SMs, ThreadsPerSM>>>(input.gpuHandle(),
                                           result.gpuHandle(),
                                           input.getSize());
  CUDA_CALL(cudaGetLastError());
}

Matrix LstmLayer::forward(std::vector<CPUMatrix> batch) {
  
  for(int t = 0; t < this->timesteps; t++) {
    
    GPUMatrix input = GPUMatrix::from(batch[t]);
    input.syncGPU();
    // input.show(std::cout);
    // this->input_weights_f.show(std::cout);

    /// Multiply input and state with weights
    // TODO: create matrices outside for loop
    GPUMatrix w_x_input_f = this->input_weights_f.multiply(input);
    GPUMatrix w_x_input_g = this->input_weights_g.multiply(input);
    GPUMatrix w_x_input_i = this->input_weights_i.multiply(input);
    GPUMatrix w_x_input_o = this->input_weights_o.multiply(input);

    // std::cout << "wxf" << std::endl;
    // w_x_input_f.show(std::cout);
    // std::cout << "wxg" << std::endl;
    // w_x_input_g.show(std::cout);
    // std::cout << "wxi" << std::endl;
    // w_x_input_i.show(std::cout);
    // std::cout << "wxo" << std::endl;
    // w_x_input_o.show(std::cout);

    // std::cout<< "Multiplied input weights" << std::endl;

    GPUMatrix w_x_state_f = this->state_weights_f.multiply(h[t]);
    GPUMatrix w_x_state_g = this->state_weights_g.multiply(h[t]);
    GPUMatrix w_x_state_i = this->state_weights_i.multiply(h[t]);
    GPUMatrix w_x_state_o = this->state_weights_o.multiply(h[t]);

    // std::cout<< "Multiplied state weights" << std::endl;

    GPUMatrix state_input_f = w_x_state_f.add(w_x_input_f);
    GPUMatrix state_input_g = w_x_state_g.add(w_x_input_g);
    GPUMatrix state_input_i = w_x_state_i.add(w_x_input_i);
    GPUMatrix state_input_o = w_x_state_o.add(w_x_input_o);
    
    // std::cout<< "Added state and input weights" << std::endl;

    // std::cout << "wxf" << std::endl;
    // state_input_f.show(std::cout);
    // std::cout << "wxg" << std::endl;
    // state_input_g.show(std::cout);
    // std::cout << "wxi" << std::endl;
    // state_input_i.show(std::cout);
    // std::cout << "wxo" << std::endl;
    // state_input_o.show(std::cout);


    /// Apply activations specific to each gate
    applySigmoid(state_input_f, f[t]);
    applySigmoid(state_input_i, i[t]);
    applySigmoid(state_input_o, o[t]);
    applyTanh(state_input_g, g[t]);

    // //Logging
    // std::cout<< "Applied activation functions" << std::endl;
    // std::cout << "f" << std::endl;
    // f[t].show(std::cout);
    // std::cout << "g" << std::endl;
    // g[t].show(std::cout);
    // std::cout << "i" << std::endl;
    // i[t].show(std::cout);
    // std::cout << "o" << std::endl;
    // o[t].show(std::cout);
      

    /// Calculate carry and state passed to another iteration
    c[t].multiplyelementwise(f[t], c[t + 1]);
    GPUMatrix i_x_g = i[t].multiplyelementwise(g[t]);
    c[t + 1].add(i_x_g, c[t + 1]);
    applyTanh(c[t + 1], tanh_c[t]);
    o[t].add(tanh_c[t], h[t + 1]); 

    // //Logging
    // std::cout<< "Calculated next state" << std::endl;
    // std::cout << "\nc" << std::endl;
    // c[t+1].show(std::cout);
    // std::cout << "\nh\n";
    // h[t+1].show(std::cout);


  }

  GPUMatrix output = this->output_weights.multiply(h[this->timesteps]);

  return output;
}
