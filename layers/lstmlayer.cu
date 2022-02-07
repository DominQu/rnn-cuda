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
  
  GPUMatrix w_x_input_f(MatrixSize(this->state_dim, 1));
  GPUMatrix w_x_input_g(MatrixSize(this->state_dim, 1));
  GPUMatrix w_x_input_i(MatrixSize(this->state_dim, 1));
  GPUMatrix w_x_input_o(MatrixSize(this->state_dim, 1));

  GPUMatrix w_x_state_f(MatrixSize(this->state_dim, 1));
  GPUMatrix w_x_state_g(MatrixSize(this->state_dim, 1));
  GPUMatrix w_x_state_i(MatrixSize(this->state_dim, 1));
  GPUMatrix w_x_state_o(MatrixSize(this->state_dim, 1));

  GPUMatrix state_input_f(MatrixSize(this->state_dim, 1));
  GPUMatrix state_input_g(MatrixSize(this->state_dim, 1));
  GPUMatrix state_input_i(MatrixSize(this->state_dim, 1));
  GPUMatrix state_input_o(MatrixSize(this->state_dim, 1));
  
  for(int t = 0; t < this->timesteps; t++) {
    
    GPUMatrix input = GPUMatrix::from(batch[t]);
    input.syncGPU();

    /// Multiply input and state with weights
    this->input_weights_f.multiply(input, w_x_input_f);
    this->input_weights_g.multiply(input, w_x_input_g);
    this->input_weights_i.multiply(input, w_x_input_i);
    this->input_weights_o.multiply(input, w_x_input_o);

    this->state_weights_f.multiply(h[t], w_x_state_f);
    this->state_weights_g.multiply(h[t], w_x_state_g);
    this->state_weights_i.multiply(h[t], w_x_state_i);
    this->state_weights_o.multiply(h[t], w_x_state_o);

    w_x_state_f.add(w_x_input_f, state_input_f);
    w_x_state_g.add(w_x_input_g, state_input_g);
    w_x_state_i.add(w_x_input_i, state_input_i);
    w_x_state_o.add(w_x_input_o, state_input_o);

    /// Apply activations specific to each gate
    applySigmoid(state_input_f, f[t]);
    applySigmoid(state_input_i, i[t]);
    applySigmoid(state_input_o, o[t]);
    applyTanh(state_input_g, g[t]);
      
    /// Calculate carry and state passed to another iteration
    c[t].multiplyelementwise(f[t], c[t + 1]);
    GPUMatrix i_x_g = i[t].multiplyelementwise(g[t]);
    c[t + 1].add(i_x_g, c[t + 1]);
    applyTanh(c[t + 1], tanh_c[t]);
    o[t].add(tanh_c[t], h[t + 1]); 


  }

  GPUMatrix output = this->output_weights.multiply(h[this->timesteps]);

  return output;
}

std::vector<GPUMatrix> LstmLayer::backward(GPUMatrix upstream, 
                                           std::vector<CPUMatrix> batch) {
  std::vector<GPUMatrix> gradients;

  GPUMatrix gradient_input_f(this->input_weights_f.getSize(), 0);
  GPUMatrix gradient_input_g(this->input_weights_g.getSize(), 0);
  GPUMatrix gradient_input_i(this->input_weights_i.getSize(), 0);
  GPUMatrix gradient_input_o(this->input_weights_o.getSize(), 0);

  GPUMatrix gradient_state_f(this->state_weights_f.getSize(), 0);
  GPUMatrix gradient_state_g(this->state_weights_g.getSize(), 0);
  GPUMatrix gradient_state_i(this->state_weights_i.getSize(), 0);
  GPUMatrix gradient_state_o(this->state_weights_o.getSize(), 0);

  // Calculate gradient connected with last timestep output weights
  GPUMatrix gradient_output_weights = upstream.multiply(this->h[this->timesteps].transpose());
  if( gradient_output_weights.getSize().height != this->output_weights.getSize().height ||
      gradient_output_weights.getSize().width != this->output_weights.getSize().width) {
        throw new InvalidMatrixSize("Gradient connected with output weights has wrong size");
      }
  // Calculate gradient connected with last timestep h
  GPUMatrix gradient_upstream_h = this->output_weights.transpose().multiply(upstream);
  if( gradient_upstream_h.getSize().height != this->h[this->timesteps].getSize().height ||
      gradient_upstream_h.getSize().width != this->h[this->timesteps].getSize().width) {
        throw new InvalidMatrixSize("Gradient connected with last timestep has wrong size");
      }
  // Initialize gradient connected with c with zeros
  GPUMatrix gradient_upstream_c(this->c[0].getSize(), 0);

  for( int t = this->timesteps - 1; t >= 0; --t) {

    GPUMatrix input = GPUMatrix::from(batch[t]);
    input.syncGPU();
    
    GPUMatrix ones(MatrixSize(this->state_dim, 1), 1);
    ones.syncGPU();

    //Calculate tanh derivative 1 - tanh^2(c)
    GPUMatrix tanh_squared = tanh_c[t].multiplyelementwise(tanh_c[t]);
    GPUMatrix tanh_derivative = ones.add(tanh_squared.multiply(-1));

    // Calculate derivative of c[t]
    // c_derivative = dUpstream * o * (1 - tanh^2(c)) + gradient_upstream_c
    GPUMatrix dUp_dc = gradient_upstream_h.multiplyelementwise(this->o[t]);
    dUp_dc.multiplyelementwise(tanh_derivative, dUp_dc);
    dUp_dc.add(gradient_upstream_c, dUp_dc);
    
    // Calculate gradients connected with f gate weights
    // x = c_derivative * f * (1 - f)
    // dUpstream/dw_input_f = x * input
    // dUpstream/dw_state_f = x * h[t-1] 
    GPUMatrix dUp_df = dUp_dc.multiplyelementwise(this->c[t]);
    dUp_df.multiplyelementwise(this->f[t], dUp_df);
    dUp_df.multiplyelementwise(ones.add(this->f[t].multiply(-1)), dUp_df);
    gradient_input_f.add(dUp_df.multiply(input.transpose()), gradient_input_f);
    gradient_state_f.add(dUp_df.multiply(this->h[t].transpose()), gradient_state_f);

    // Calculate gradients connected with g gate weights
    // x = c_derivative * i * (1 - g^2)
    // dUpstream/dw_input_g = x * input
    // dUpstream/dw_state_g = x * h[t-1]
    GPUMatrix dUp_dg = dUp_dc.multiplyelementwise(this->i[t]);
    GPUMatrix tanh_g_squared = g[t].multiplyelementwise(g[t]);
    dUp_dg.multiplyelementwise(ones.add(tanh_g_squared.multiply(-1)), dUp_dg);
    gradient_input_g.add(dUp_dg.multiply(input.transpose()), gradient_input_g);
    gradient_state_g.add(dUp_dg.multiply(this->h[t].transpose()), gradient_state_g);

    // Calculate gradients connected with i gate weights
    // x = c_derivative * g * i * (1 - i)
    // dUpstream/dw_input_i = x * input
    // dUpstream/dw_state_i = x * h[t-1]
    GPUMatrix dUp_di = dUp_dc.multiplyelementwise(this->g[t]);
    dUp_di.multiplyelementwise(this->i[t], dUp_di);
    dUp_di.multiplyelementwise(ones.add(this->i[t].multiply(-1)), dUp_di);
    gradient_input_i.add(dUp_di.multiply(input.transpose()), gradient_input_i);
    gradient_state_i.add(dUp_di.multiply(this->h[t].transpose()), gradient_state_i);

    // Calculate gradients connected with o gate weights
    //x = dUpstream * tanh(c) * sigmoid(Wo*input) * (1 - sigmoid(Wo*input))
    //dUpstream/dw_input_o = x * input
    //dUpstream/dw_state_o = x * h[t-1]

    GPUMatrix dUp_do = gradient_upstream_h.multiplyelementwise(this->tanh_c[t]);
    dUp_do.multiplyelementwise(o[t], dUp_do);
    dUp_do.multiplyelementwise(ones.add(o[t].multiply(-1)), dUp_do);
    gradient_input_o.add(dUp_do.multiply(input.transpose()), gradient_input_o);
    gradient_state_o.add(dUp_do.multiply(this->h[t].transpose()), gradient_state_o);

    //Update gradient connected with c passed to a timestep before
    dUp_dc.multiplyelementwise(this->f[t], gradient_upstream_c);

    //Update gradient connected with h passed to a timestep before
    this->state_weights_f.transpose().multiply(dUp_df, gradient_upstream_h);
    gradient_upstream_h.add(this->state_weights_g.transpose().multiply(dUp_dg), gradient_upstream_h);
    gradient_upstream_h.add(this->state_weights_i.transpose().multiply(dUp_di), gradient_upstream_h);
    gradient_upstream_h.add(this->state_weights_o.transpose().multiply(dUp_do), gradient_upstream_h);

  }

  gradients.push_back(gradient_input_f);
  gradients.push_back(gradient_input_g);
  gradients.push_back(gradient_input_i);
  gradients.push_back(gradient_input_o);

  gradients.push_back(gradient_state_f);
  gradients.push_back(gradient_state_g);
  gradients.push_back(gradient_state_i);
  gradients.push_back(gradient_state_o);

  gradients.push_back(gradient_output_weights);
  gradients.push_back(gradient_upstream_h);

  return gradients;
}