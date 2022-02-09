#include "recurrent.hpp"
#include "layers/lstmlayer.hpp"
#include "layers/CCEloss.hpp"
#include "layers/softmax.hpp"
#include "optimizer/sgd.hpp"
#include "loader/loader.hpp"
#include "linalg/GPUMatrix.hpp"

Recurrent::Recurrent(int input_size, int state_size, 
                     int timesteps, float random_weights_low, 
                     float random_weights_high, float learning_rate) 
                     : timesteps(timesteps),
                       lstmlayer1({input_size, state_size, timesteps, 
                                   random_weights_low, random_weights_high}), 
                       softmax1({input_size}), 
                       cceloss1({}), 
                       sgd({learning_rate}) { }

Recurrent::Recurrent(int input_size, int state_size, 
                     int timesteps, float learning_rate, std::string filepath) 
                     : timesteps(timesteps),
                       lstmlayer1({input_size, state_size, 
                                   timesteps, filepath}), 
                       softmax1({input_size}), 
                       cceloss1({}), 
                       sgd({learning_rate}) { }

std::vector<float> Recurrent::train(int epochs, DataLoader &dl, int log) {
  std::vector<float> loss;

  for(int epoch = 0; epoch < epochs; epoch++) {
    // get training batch and separate the label
    std::vector<GPUMatrix> batch = dl.getTrainBatch(this->timesteps+1);
    GPUMatrix label = batch.back();
    batch.pop_back();  

    GPUMatrix lstm_output = this->lstmlayer1.forward(batch);
    GPUMatrix softmax_output = this->softmax1.forward(lstm_output);
    MatrixValType cceloss_output = this->cceloss1.forward(softmax_output, label);

    if(epoch % log == 0) {
        std::cout << "Epoch: " << epoch;
        std::cout << ", Current loss: " << cceloss_output;
        std::cout << "\n"; 
    }

    loss.push_back(cceloss_output);

    GPUMatrix cceloss_gradient = cceloss1.backward(softmax_output, label);
    std::vector<GPUMatrix> lstm_gradient = lstmlayer1.backward(cceloss_gradient, batch);
    std::vector<GPUMatrix> optimizer_output = sgd.calculateUpdate(lstm_gradient);
    lstmlayer1.updateWeights(optimizer_output);
  }
  return loss;
}

void Recurrent::generateText(int generated_text_length, DataLoader &dl, std::ostream &stream) {
  // Show text given to the network as input
  std::vector<GPUMatrix> start = dl.getTrainBatch(this->timesteps);
  for( auto i:start) {
      stream << dl.getOneHot().decode(i.toCPU());
  }
  // Generate text of given length
  for( int letter = 0; letter < generated_text_length; letter++) {

    GPUMatrix lstm_output = lstmlayer1.forward(start);  
    GPUMatrix softmax_output = softmax1.forward(lstm_output);
    CPUMatrix probabilities = softmax_output.toCPU();

    int argmax = probabilities.argmax();
    CPUMatrix onehot(probabilities.getSize(), 0);
    onehot.at(argmax, 0) = 1.0f;  
    GPUMatrix nextletter = GPUMatrix::from(onehot);

    // Append network input with predicted letter
    start.push_back(nextletter);
    start = std::vector<GPUMatrix>(start.begin() + 1, start.end());

    // Show generated letters to output stream
    stream << dl.getOneHot().decode(onehot);
  }
  std::cout << "\n"; 
}

void Recurrent::saveModel() {
  std::fstream outFile;
  outFile.open("model1.txt", std::ios::out);
  if (!outFile) {
      std::cout << "Error" << std::endl;
  }
  else {
      this->lstmlayer1.saveWeights(outFile);
      std::cout << "Model saved succesfully" << std::endl;
      outFile.close();
  }
}